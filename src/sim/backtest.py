"""Event-driven backtest orchestration for pure market making."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

import polars as pl

from .execution import ExecutionSimulator
from .market_state import MarketState
from .quoting_logic import AvellanedaStoikovModel


@dataclass
class BacktestResult:
    """Summary of one backtest run."""

    fills: int
    final_inventory: float
    cash: float
    execution: ExecutionSimulator
    marked_pnl: float = 0.0
    last_mid: float = 0.0
    last_spread: float = 0.0
    last_gamma: float = 0.0
    last_kappa: float = 0.0
    equity_curve: list[tuple[datetime, float]] = field(default_factory=list)
    daily_pnl: list[tuple[str, float, float]] = field(default_factory=list)


@dataclass
class BacktestSnapshot:
    """Current state emitted while replaying a backtest."""

    market: MarketState
    quote: object
    last_trade: dict | None
    last_fill: object | None
    execution: ExecutionSimulator

    @property
    def marked_pnl(self) -> float:
        """Cash plus current inventory marked at the market mid-price."""
        return self.execution.position.marked_pnl(self.market.mid_price)

    @property
    def realized_pnl(self) -> float:
        return self.execution.position.realized_pnl

    @property
    def unrealized_pnl(self) -> float:
        return self.execution.position.unrealized_pnl(self.market.mid_price)


class Backtest:
    """Replay BBO and trade events for one market-making instrument."""

    def __init__(
        self,
        model: AvellanedaStoikovModel,
        execution: ExecutionSimulator,
        volatility_window: int = 300,
        interval_seconds: int = 1,
        estimate_kappa: bool = True,
    ) -> None:
        if volatility_window < 3:
            raise ValueError("volatility_window must be at least 3")
        self.model = model
        self.execution = execution
        self.volatility_window = volatility_window
        self.interval_seconds = interval_seconds
        self.estimate_kappa = estimate_kappa
        self._minute_mid: float | None = None
        self._minute_ts: datetime | None = None
        self._kappa_distances: list[float] = []
        self._kappa_fills: list[int] = []
        self._kappa_exposure: list[float] = []
        self._kappa_updates = 0

    def run(
        self,
        bbo: pl.DataFrame,
        trades: pl.DataFrame,
        on_update: Callable[[BacktestSnapshot], None] | None = None,
    ) -> BacktestResult:
        """Replay BBO snapshots and trades in timestamp order.

        The BBO frame must contain ``ts``, ``bid_price``, ``bid_amount``,
        ``ask_price``, and ``ask_amount``. The trades frame must contain
        ``ts``, ``price``, ``amount``, and aggressor ``side``.
        """
        bbo_rows = list(bbo.sort("ts").iter_rows(named=True))
        trade_rows = list(trades.sort("ts").iter_rows(named=True))
        return self.run_rows(bbo_rows, trade_rows, on_update=on_update)

    def run_rows(
        self,
        bbo_rows: list[dict],
        trade_rows: list[dict],
        on_update: Callable[[BacktestSnapshot], None] | None = None,
    ) -> BacktestResult:
        """Replay pre-materialized BBO and trade rows."""
        trade_index = 0
        mid_history: list[float] = []
        last_quote = None
        last_mid = 0.0
        equity_curve: list[tuple[datetime, float]] = []
        daily_pnl: list[tuple[str, float, float]] = []
        day_start_pnl: float | None = None
        current_day: str | None = None
        last_equity_ts: datetime | None = None
        sample_seconds = max(float(self.interval_seconds), 60.0)

        for index, row in enumerate(bbo_rows):
            last_trade = None
            last_fill = None
            mid_price = (row["bid_price"] + row["ask_price"]) / 2.0
            mid_history.append(mid_price)
            recent_prices = mid_history[-self.volatility_window:]
            volatility = MarketState.calculate_volatility(
                recent_prices,
                self.interval_seconds,
            )

            market = MarketState(
                timestamp=row["ts"],
                best_bid=row["bid_price"],
                best_bid_volume=row["bid_amount"],
                best_ask_volume=row["ask_amount"],
                best_ask=row["ask_price"],
                volatility=volatility,
                bid_prices=row.get("bid_prices") or [],
                bid_volumes=row.get("bid_amounts") or [],
                ask_prices=row.get("ask_prices") or [],
                ask_volumes=row.get("ask_amounts") or [],
            )
            self._update_minute_returns(row["ts"], mid_price)
            quote = self.model.calculate_quotes(
                market,
                self.execution.position,
            )
            if quote.bid_enabled or quote.ask_enabled:
                self.execution.post_quote(quote)
            else:
                self.execution.active_quote = None

            next_timestamp = (
                bbo_rows[index + 1]["ts"]
                if index + 1 < len(bbo_rows)
                else None
            )
            bid_fills = 0
            ask_fills = 0
            while trade_index < len(trade_rows):
                trade = trade_rows[trade_index]
                if trade["ts"] < row["ts"]:
                    trade_index += 1
                    continue
                if next_timestamp is not None and trade["ts"] >= next_timestamp:
                    break

                fill = self.execution.process_trade(
                    timestamp=trade["ts"],
                    trade_price=trade["price"],
                    trade_quantity=trade["amount"],
                    trade_side=trade["side"],
                )
                last_trade = trade
                last_fill = fill
                if fill is not None:
                    if fill.side == "buy":
                        bid_fills += 1
                    else:
                        ask_fills += 1
                trade_index += 1

            last_quote = quote
            last_mid = mid_price
            marked = (
                self.execution.position.cash
                + self.execution.position.inventory * mid_price
            )
            day_key = str(row["ts"])[:10]
            if current_day is None:
                current_day = day_key
                day_start_pnl = marked
                equity_curve.append((row["ts"], marked))
                last_equity_ts = row["ts"]
            elif day_key != current_day:
                daily_pnl.append(
                    (current_day, marked - (day_start_pnl or 0.0), marked)
                )
                current_day = day_key
                day_start_pnl = marked
                equity_curve.append((row["ts"], marked))
                last_equity_ts = row["ts"]
            elif (
                last_equity_ts is None
                or (row["ts"] - last_equity_ts).total_seconds() >= sample_seconds
                or index == len(bbo_rows) - 1
            ):
                equity_curve.append((row["ts"], marked))
                last_equity_ts = row["ts"]

            if self.estimate_kappa:
                self._record_kappa_observation(
                    mid_price=mid_price,
                    quote=quote,
                    bid_fills=bid_fills,
                    ask_fills=ask_fills,
                    current_ts=row["ts"],
                    next_ts=next_timestamp,
                )

            if on_update is not None:
                on_update(
                    BacktestSnapshot(
                        market=market,
                        quote=quote,
                        last_trade=last_trade,
                        last_fill=last_fill,
                        execution=self.execution,
                    )
                )

        final_pnl = (
            self.execution.position.cash
            + self.execution.position.inventory * last_mid
        )
        if current_day is not None:
            daily_pnl.append(
                (current_day, final_pnl - (day_start_pnl or 0.0), final_pnl)
            )
        if not equity_curve or equity_curve[-1][1] != final_pnl:
            stamp = bbo_rows[-1]["ts"] if bbo_rows else datetime.utcnow()
            equity_curve.append((stamp, final_pnl))

        return BacktestResult(
            fills=len(self.execution.fills),
            final_inventory=self.execution.position.inventory,
            cash=self.execution.position.cash,
            execution=self.execution,
            marked_pnl=final_pnl,
            last_mid=last_mid,
            last_spread=last_quote.spread if last_quote is not None else 0.0,
            last_gamma=last_quote.gamma if last_quote is not None else 0.0,
            last_kappa=last_quote.kappa if last_quote is not None else 0.0,
            equity_curve=equity_curve,
            daily_pnl=daily_pnl,
        )

    def _update_minute_returns(self, timestamp: datetime, mid_price: float) -> None:
        """Append a 1-minute mid return / close when at least 60 seconds elapsed."""
        if self._minute_ts is None or self._minute_mid is None:
            self._minute_ts = timestamp
            self._minute_mid = mid_price
            return
        elapsed = (timestamp - self._minute_ts).total_seconds()
        if elapsed < 60.0 or self._minute_mid <= 0:
            return
        self.model.params.returns_1m.append(mid_price / self._minute_mid - 1.0)
        del self.model.params.returns_1m[:-10]
        self.model.params.closes_1m.append(mid_price)
        keep = max(int(self.model.params.ma_slow), 25) + 5
        del self.model.params.closes_1m[:-keep]
        self._minute_ts = timestamp
        self._minute_mid = mid_price

    def _record_kappa_observation(
        self,
        mid_price: float,
        quote,
        bid_fills: int,
        ask_fills: int,
        current_ts: datetime,
        next_ts: datetime | None,
    ) -> None:
        """Store quote-distance fill samples and periodically re-estimate kappa."""
        if next_ts is None:
            exposure = float(self.interval_seconds)
        else:
            exposure = max((next_ts - current_ts).total_seconds(), 1e-6)
        bid_distance = max(mid_price - quote.bid, 0.0)
        ask_distance = max(quote.ask - mid_price, 0.0)
        self._kappa_distances.extend([bid_distance, ask_distance])
        self._kappa_fills.extend([bid_fills, ask_fills])
        self._kappa_exposure.extend([exposure, exposure])
        del self._kappa_distances[:-2000]
        del self._kappa_fills[:-2000]
        del self._kappa_exposure[:-2000]
        self._kappa_updates += 1
        if self._kappa_updates % 200 != 0:
            return
        try:
            self.model.estimated_kappa = self.model.kappa_calculation(
                self._kappa_distances,
                self._kappa_fills,
                self._kappa_exposure,
            )
        except ValueError:
            return
