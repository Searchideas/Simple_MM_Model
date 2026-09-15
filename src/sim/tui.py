"""Terminal dashboard for replaying the pure market-making backtest."""

from datetime import datetime
import time
from collections import deque

import msvcrt

from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table

from .backtest import BacktestSnapshot


class BacktestTUI:
    """Render one backtest snapshot at a time in the terminal."""

    def __init__(self, speed: float = 0.0) -> None:
        if speed < 0:
            raise ValueError("speed must be non-negative")
        self.speed = speed
        self._previous_timestamp: datetime | None = None
        self._live: Live | None = None
        self._paused = False
        self._equity_curve: list[float] = []
        self._trade_history: deque[str] = deque(maxlen=8)

    def __enter__(self) -> "BacktestTUI":
        self._live = Live(self.render_message("Starting replay..."), refresh_per_second=10)
        self._live.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._live is not None:
            self._live.stop()

    def update(self, snapshot: BacktestSnapshot) -> None:
        """Render a snapshot and optionally preserve historical timing."""
        self._handle_keyboard()
        while self._paused:
            if self._live is not None:
                self._live.update(self.render(snapshot))
            self._handle_keyboard()
            time.sleep(0.1)

        if self.speed > 0 and self._previous_timestamp is not None:
            elapsed = (
                snapshot.market.timestamp - self._previous_timestamp
            ).total_seconds()
            time.sleep(min(max(elapsed / self.speed, 0.0), 1.0))

        self._previous_timestamp = snapshot.market.timestamp
        self._equity_curve.append(snapshot.marked_pnl)
        self._record_activity(snapshot)
        if self._live is not None:
            self._live.update(self.render(snapshot))

    def render(self, snapshot: BacktestSnapshot) -> Layout:
        """Build the current dashboard."""
        market = snapshot.market
        quote = snapshot.quote
        position = snapshot.execution.position
        total_fees = sum(fill.fee for fill in snapshot.execution.fills)

        state = Table(show_header=False, box=None, padding=(0, 1))
        state.add_column("Name", style="bold cyan")
        state.add_column("Value", justify="right")
        state.add_row("Timestamp", str(market.timestamp))
        state.add_row("Market bid", f"{market.best_bid:.8f} ({market.best_bid_volume:.6f})")
        state.add_row("Market ask", f"{market.best_ask:.8f} ({market.best_ask_volume:.6f})")
        state.add_row("Mid-price", f"{market.mid_price:.8f}")
        state.add_row("Volatility", f"{market.volatility:.8f}")
        state.add_row("Our bid", f"{quote.bid:.8f}")
        state.add_row("Our ask", f"{quote.ask:.8f}")
        state.add_row("Reservation", f"{quote.reservation_price:.8f}")
        state.add_row("Our spread", f"{quote.spread:.8f}")
        state.add_row("Status", "PAUSED" if self._paused else "RUNNING")

        book = Table(show_header=True, box=None, padding=(0, 1))
        book.add_column("Level", style="bold cyan")
        book.add_column("Bid price", justify="right")
        book.add_column("Bid size", justify="right")
        book.add_column("Ask price", justify="right")
        book.add_column("Ask size", justify="right")
        for level, (bid_price, bid_volume, ask_price, ask_volume) in enumerate(
            zip(
                market.bid_prices,
                market.bid_volumes,
                market.ask_prices,
                market.ask_volumes,
            ),
            start=1,
        ):
            book.add_row(
                str(level),
                f"{bid_price:.8f}",
                f"{bid_volume:.6f}",
                f"{ask_price:.8f}",
                f"{ask_volume:.6f}",
            )

        account = Table(show_header=False, box=None, padding=(0, 1))
        account.add_column("Name", style="bold cyan")
        account.add_column("Value", justify="right")
        account.add_row("Fills", str(len(snapshot.execution.fills)))
        account.add_row("Inventory", f"{position.inventory:.8f}")
        account.add_row("Cash", f"{position.cash:.8f}")
        account.add_row("Fees", f"{total_fees:.8f}")
        account.add_row("Marked PnL", f"{snapshot.marked_pnl:.8f}")

        activity = Table(show_header=False, box=None, padding=(0, 1))
        activity.add_column("Trade activity", style="bold cyan")
        for item in self._trade_history:
            activity.add_row(item)
        if not self._trade_history:
            activity.add_row("Waiting for trades...")

        equity = self._render_equity_curve()
        layout = Layout(name="root")
        layout.split_column(
            Layout(name="top", size=12),
            Layout(name="bottom"),
        )
        layout["top"].split_row(
            Layout(Panel(state, title="Market and Quote"), ratio=1),
            Layout(Panel(account, title="Account"), ratio=1),
            Layout(Panel(equity, title="Equity Curve"), ratio=1),
        )
        layout["bottom"].split_row(
            Layout(Panel(book, title="Order Book (25 levels)"), ratio=2),
            Layout(Panel(activity, title="Trades and Fills"), ratio=1),
        )
        return layout

    def _render_equity_curve(self) -> str:
        """Render a compact equity curve without a Rich sparkline dependency."""
        values = self._equity_curve or [0.0]
        values = values[-60:]
        levels = ".:-=+*#@"
        minimum = min(values)
        maximum = max(values)

        if maximum == minimum:
            return levels[0] * len(values)

        return "".join(
            levels[
                min(
                    len(levels) - 1,
                    int(
                        (value - minimum)
                        / (maximum - minimum)
                        * (len(levels) - 1)
                    ),
                )
            ]
            for value in values
        )

    def _record_activity(self, snapshot: BacktestSnapshot) -> None:
        """Add newly observed public trades and our fills to the activity box."""
        if snapshot.last_trade is not None:
            trade = snapshot.last_trade
            self._trade_history.append(
                f"PUBLIC {trade['side']} {trade['amount']:.6f} @ {trade['price']:.8f}"
            )
        if snapshot.last_fill is not None:
            fill = snapshot.last_fill
            self._trade_history.append(
                f"OUR FILL {fill.side} {fill.quantity:.6f} @ {fill.price:.8f}"
            )

    def _handle_keyboard(self) -> None:
        """Toggle pause with spacebar without blocking the replay."""
        while msvcrt.kbhit():
            key = msvcrt.getwch()
            if key == " ":
                self._paused = not self._paused

    def render_message(self, message: str) -> Layout:
        layout = Layout(name="root")
        layout.update(Panel(message, title="Pure Market-Making Replay"))
        return layout

