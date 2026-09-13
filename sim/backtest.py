"""Event-driven backtest orchestration for pure market making."""

from dataclasses import dataclass
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
        return self.execution.position.cash + (
            self.execution.position.inventory * self.market.mid_price
        )


class Backtest:
    """Replay BBO and trade events for one market-making instrument."""

    def __init__(
        self,
        model: AvellanedaStoikovModel,
        execution: ExecutionSimulator,
        volatility_window: int = 300,
        interval_seconds: int = 1,
    ) -> None:
        if volatility_window < 3:
            raise ValueError("volatility_window must be at least 3")
        self.model = model
        self.execution = execution
        self.volatility_window = volatility_window
        self.interval_seconds = interval_seconds

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
        trade_index = 0
        mid_history: list[float] = []

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
                bid_prices=row["bid_prices"],
                bid_volumes=row["bid_amounts"],
                ask_prices=row["ask_prices"],
                ask_volumes=row["ask_amounts"],
            )
            quote = self.model.calculate_quotes(
                market,
                self.execution.position,
            )
            self.execution.post_quote(quote)

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

        return BacktestResult(
            fills=len(self.execution.fills),
            final_inventory=self.execution.position.inventory,
            cash=self.execution.position.cash,
            execution=self.execution,
        )
