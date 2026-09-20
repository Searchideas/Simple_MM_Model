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
        interval_seconds: float = 1,
        kappa_path: pl.DataFrame | None = None,
        kappa_fallback: float | None = None,
    ) -> None:
        if volatility_window < 3:
            raise ValueError("volatility_window must be at least 3")
        self.model = model
        self.execution = execution
        self.volatility_window = volatility_window
        self.interval_seconds = interval_seconds
        self.kappa_path = kappa_path
        self.kappa_fallback = (
            float(kappa_fallback)
            if kappa_fallback is not None
            else float(model.params.kappa)
        )
        if self.kappa_path is not None and not self.kappa_path.is_empty():
            self.model.estimated_kappa = self.kappa_fallback


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
            if self.kappa_path is not None:
                from .fill_probabilty import kappa_at_time

                self.model.estimated_kappa = kappa_at_time(
                    self.kappa_path,
                    row["ts"],
                    self.kappa_fallback,
                )
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
