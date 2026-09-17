"""Terminal dashboard for replaying the pure market-making backtest."""

from datetime import datetime
import time
from collections import deque
from typing import Any

import msvcrt

from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .backtest import BacktestSnapshot
from .sweep import SweepRow


AS_FORMULA = """
r = mid - q_frac*skew*tick + book_shift + ma_shift
d = clip(AS+vol, min_spread/fee_floor, max_spread_ticks*tick)
skew moves center up to +/- skew_ticks (buy/sell tilt)
Flat QUOTE: drop toxic OBI / fade MA side
Pos: WAIT_TP entry+/-fee | TP/FLATTEN join best
"""


class BacktestTUI:
    """Render one backtest snapshot at a time in the terminal."""

    def __init__(
        self,
        speed: float = 0.0,
        *,
        symbol: str | None = None,
        date: str | None = None,
        parameters: Any | None = None,
        maker_fee: float | None = None,
        max_inventory: float | None = None,
        volatility_window: int | None = None,
    ) -> None:
        if speed < 0:
            raise ValueError("speed must be non-negative")
        self.speed = speed
        self.symbol = symbol
        self.date = date
        self.parameters = parameters
        self.maker_fee = maker_fee
        self.max_inventory = max_inventory
        self.volatility_window = volatility_window
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
        state.add_row("Our bid", f"{quote.bid:.8f}" + ("" if getattr(quote, "bid_enabled", True) else " OFF"))
        state.add_row("Our ask", f"{quote.ask:.8f}" + ("" if getattr(quote, "ask_enabled", True) else " OFF"))
        state.add_row("Action", getattr(quote, "action", "QUOTE"))
        state.add_row("Reservation", f"{quote.reservation_price:.8f}")
        state.add_row("VWAP bid", f"{getattr(quote, 'vwap_bid', 0.0):.8f}")
        state.add_row("VWAP ask", f"{getattr(quote, 'vwap_ask', 0.0):.8f}")
        state.add_row("Book shift", f"{getattr(quote, 'book_shift', 0.0):+.8f}")
        state.add_row("MA shift", f"{getattr(quote, 'ma_shift', 0.0):+.8f}")
        state.add_row("MA frac", f"{getattr(quote, 'ma_frac', 0.0):+.3f}")
        state.add_row("VWAP shift", f"{getattr(quote, 'book_vwap_shift', 0.0):+.8f}")
        state.add_row("Size shift", f"{getattr(quote, 'book_size_shift', 0.0):+.8f}")
        state.add_row("Book imb I", f"{getattr(quote, 'book_imbalance', 0.0):+.4f}")
        state.add_row("Our spread", f"{quote.spread:.8f}")
        state.add_row("Status", "PAUSED" if self._paused else "RUNNING")

        book = Table(show_header=True, box=None, padding=(0, 1))
        book.add_column("Level", style="bold cyan")
        book.add_column("Bid price", justify="right")
        book.add_column("Bid size", justify="right")
        book.add_column("Ask price", justify="right")
        book.add_column("Ask size", justify="right")
        tick = getattr(self.parameters, "tick_size", 1e-12) or 1e-12
        for level, (bid_price, bid_volume, ask_price, ask_volume) in enumerate(
            zip(
                market.bid_prices,
                market.bid_volumes,
                market.ask_prices,
                market.ask_volumes,
            ),
            start=1,
        ):
            our_bid = abs(bid_price - quote.bid) <= tick * 0.5 + 1e-15
            our_ask = abs(ask_price - quote.ask) <= tick * 0.5 + 1e-15
            bid_txt = f"{bid_price:.8f}"
            ask_txt = f"{ask_price:.8f}"
            if our_bid:
                bid_txt = f"[bold green]>> {bid_txt}[/]"
            if our_ask:
                ask_txt = f"[bold red]>> {ask_txt}[/]"
            level_txt = str(level)
            if our_bid or our_ask:
                level_txt = f"[bold yellow]{level}[/]"
            book.add_row(
                level_txt,
                bid_txt,
                f"{bid_volume:.6f}",
                ask_txt,
                f"{ask_volume:.6f}",
            )

        account = Table(show_header=False, box=None, padding=(0, 1))
        account.add_column("Name", style="bold cyan")
        account.add_column("Value", justify="right")
        realized = position.realized_pnl
        unrealized = position.unrealized_pnl(market.mid_price)
        account.add_row("Fills", str(len(snapshot.execution.fills)))
        account.add_row("Inventory", f"{position.inventory:.8f}")
        account.add_row("Avg entry", f"{position.avg_entry_price:.8f}")
        account.add_row("Cash", f"{position.cash:.8f}")
        account.add_row("Fees", f"{total_fees:.8f}")
        account.add_row("Realized PnL", f"{realized:.8f}")
        account.add_row("Unrealized PnL", f"{unrealized:.8f}")
        account.add_row("Marked PnL", f"{snapshot.marked_pnl:.8f}")
        account.add_row("R+U check", f"{realized + unrealized:.8f}")

        activity = Table(show_header=False, box=None, padding=(0, 1))
        activity.add_column("Trade activity", style="bold cyan")
        for item in self._trade_history:
            activity.add_row(item)
        if not self._trade_history:
            activity.add_row("Waiting for trades...")

        equity = self._render_equity_curve()
        formula = Text(AS_FORMULA, style="bright_white")
        params = self._render_parameters(snapshot)

        layout = Layout(name="root")
        layout.split_column(
            Layout(name="top", size=14),
            Layout(name="middle", size=14),
            Layout(name="bottom"),
        )
        layout["top"].split_row(
            Layout(Panel(state, title="Market and Quote"), ratio=1),
            Layout(Panel(account, title="Account (R + U = Marked)"), ratio=1),
            Layout(Panel(equity, title="Equity Curve"), ratio=1),
        )
        layout["middle"].split_row(
            Layout(Panel(formula, title="Avellaneda–Stoikov"), ratio=1),
            Layout(Panel(params, title="Parameters"), ratio=1),
        )
        layout["bottom"].split_row(
            Layout(
                Panel(book, title="Order Book (>> = our bid/ask)"),
                ratio=2,
            ),
            Layout(Panel(activity, title="Trades and Fills"), ratio=1),
        )
        return layout

    def _render_parameters(self, snapshot: BacktestSnapshot) -> Table:
        """Show configured model/execution knobs and live quote inputs."""
        params = Table(show_header=False, box=None, padding=(0, 1))
        params.add_column("Name", style="bold cyan")
        params.add_column("Value", justify="right")

        if self.symbol is not None:
            params.add_row("Symbol", self.symbol)
        if self.date is not None:
            params.add_row("Date", self.date)

        model = self.parameters
        quote = snapshot.quote
        if model is not None:
            params.add_row("gamma (base)", f"{model.base_gamma:.6g}")
            params.add_row("gamma_eff", f"{getattr(quote, 'gamma', model.base_gamma):.6g}")
            params.add_row("kappa (base)", f"{model.kappa:.6g}")
            params.add_row("kappa_eff", f"{getattr(quote, 'kappa', model.kappa):.6g}")
            params.add_row("imbalance", f"{getattr(quote, 'imbalance', 0.0):.3f}")
            params.add_row("1m returns", str(len(model.returns_1m)))
            params.add_row("horizon (s)", f"{model.time_horizon:.6g}")
            params.add_row("min_spread", f"{model.min_spread:.8f}")
            params.add_row(
                "max_spread_ticks",
                f"{getattr(model, 'max_spread_ticks', 10):.6g}",
            )
            params.add_row("tick_size", f"{model.tick_size:.8f}")
            params.add_row("skew_ticks", f"{getattr(model, 'max_skew_ticks', 0):.6g}")
            params.add_row(
                "book_skew_ticks",
                f"{getattr(model, 'book_skew_ticks', 0):.6g}",
            )
            params.add_row(
                "book_skew_w",
                f"{getattr(model, 'book_skew_size_weight', 0.75):.3f}",
            )
            params.add_row(
                "ma_fast/slow",
                f"{getattr(model, 'ma_fast', 7)}/{getattr(model, 'ma_slow', 25)}",
            )
            params.add_row(
                "ma_skew_ticks",
                f"{getattr(model, 'ma_skew_ticks', 0):.6g}",
            )
            params.add_row("ma_shift", f"{getattr(quote, 'ma_shift', 0.0):+.8f}")
            params.add_row("ma_frac", f"{getattr(quote, 'ma_frac', 0.0):+.3f}")
            params.add_row("action", getattr(quote, "action", "QUOTE"))
            params.add_row("VWAP bid", f"{getattr(quote, 'vwap_bid', 0.0):.8f}")
            params.add_row("VWAP ask", f"{getattr(quote, 'vwap_ask', 0.0):.8f}")
            params.add_row(
                "book_shift",
                f"{getattr(quote, 'book_shift', 0.0):+.8f}",
            )
            params.add_row(
                "book_imb I",
                f"{getattr(quote, 'book_imbalance', 0.0):+.3f}",
            )
            params.add_row("spread unit", "ticks")

        if self.maker_fee is not None:
            params.add_row("maker_fee", f"{self.maker_fee:.6g}")
        if self.max_inventory is not None:
            params.add_row("max_inventory", f"{self.max_inventory:.6g}")
        if self.volatility_window is not None:
            params.add_row("vol_window", str(self.volatility_window))

        inventory = snapshot.execution.position.inventory
        mid = snapshot.market.mid_price
        vol = snapshot.market.volatility
        horizon = getattr(model, "time_horizon", None) if model is not None else None
        if horizon is not None:
            t_years = horizon / (365 * 24 * 60 * 60)
            params.add_row("q (inventory)", f"{inventory:.8f}")
            params.add_row("σ (ann. vol)", f"{vol:.8f}")
            params.add_row("T (years)", f"{t_years:.8g}")
            params.add_row("mid", f"{mid:.8f}")

        params.add_row("Speed", "max" if self.speed == 0 else f"{self.speed:g}x")
        params.add_row("Controls", "SPACE = pause/resume")
        return params

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


def _read_key() -> str:
    """Read one key, mapping Windows arrows to up/down/enter/quit."""
    key = msvcrt.getwch()
    if key in {"\x00", "\xe0"}:
        extra = msvcrt.getwch()
        if extra == "H":
            return "up"
        if extra == "P":
            return "down"
        if extra == "K":
            return "up"
        if extra == "M":
            return "down"
        return extra
    if key in {"\r", "\n"}:
        return "enter"
    if key in {"q", "Q", "\x1b"}:
        return "quit"
    if key == "k":
        return "up"
    if key == "j":
        return "down"
    if key in {"r", "R"}:
        return "r"
    if key in {"b", "B"}:
        return "b"
    return key


class SweepPicker:
    """First TUI page: pick a parameter combination after the grid finishes."""

    def __init__(
        self,
        *,
        symbol: str,
        date: str,
        every: str,
        tick_size: float,
        horizon: float,
        maker_fee: float,
        volatility_window: int,
    ) -> None:
        self.symbol = symbol
        self.date = date
        self.every = every
        self.tick_size = tick_size
        self.horizon = horizon
        self.maker_fee = maker_fee
        self.volatility_window = volatility_window
        self._index = 0
        self._offset = 0
        self._visible = 16

    def render_progress(
        self,
        index: int,
        total: int,
        combo,
        finished: list[SweepRow],
    ) -> Layout:
        status = (
            f"Running {index}/{total}  "
            f"γ={combo.gamma:g} κ={combo.kappa:g} min={combo.min_spread:g} "
            f"max_sp={combo.max_spread_ticks:g} inv={combo.max_inventory:g} "
            f"skew={combo.skew_ticks:g} book={combo.book_skew_ticks:g} "
            f"w={combo.book_skew_size_weight:g}"
        )
        body = Table(show_header=True, box=None, padding=(0, 1))
        self._add_header(body)
        preview = sorted(finished, key=lambda row: row.marked_pnl, reverse=True)[:8]
        for row_index, row in enumerate(preview, start=1):
            self._add_result_row(body, row_index, row, selected=False)
        if not preview:
            body.add_row("-", "…", "…", "…", "…", "…", "…", "…", "…", "…", "…", "…", "…", "waiting")
        layout = Layout(name="root")
        layout.split_column(
            Layout(
                Panel(
                    f"{self.symbol} {self.date}  resample={self.every}\n{status}",
                    title="Parameter Sweep",
                ),
                size=5,
            ),
            Layout(Panel(body, title="Best so far (marked PnL)")),
        )
        return layout

    def choose(self, rows: list[SweepRow]) -> SweepRow | None:
        """Arrow/j/k to move, Enter opens equity report, q to quit."""
        if not rows:
            return None
        self._index = 0
        with Live(self._render(rows), refresh_per_second=12, screen=True) as live:
            while True:
                if msvcrt.kbhit():
                    action = _read_key()
                    if action == "up":
                        self._index = max(0, self._index - 1)
                    elif action == "down":
                        self._index = min(len(rows) - 1, self._index + 1)
                    elif action == "enter":
                        return rows[self._index]
                    elif action == "quit":
                        return None
                    live.update(self._render(rows))
                time.sleep(0.05)

    def show_report(self, row: SweepRow) -> str:
        """Static equity + daily PnL page. Returns 'replay', 'back', or 'quit'."""
        report = ResultReport(self.symbol, self.date, row)
        with Live(report.render(), refresh_per_second=8, screen=True) as live:
            while True:
                if msvcrt.kbhit():
                    action = _read_key()
                    if action == "enter" or action == "r":
                        return "replay"
                    if action == "b" or action == "quit":
                        # b = back to grid handled by caller; quit exits
                        return "back" if action == "b" else "quit"
                    if action == "up":
                        report.scroll(-1)
                    elif action == "down":
                        report.scroll(1)
                    live.update(report.render())
                time.sleep(0.05)

    def _render(self, rows: list[SweepRow]) -> Layout:
        if self._index < self._offset:
            self._offset = self._index
        if self._index >= self._offset + self._visible:
            self._offset = self._index - self._visible + 1

        table = Table(show_header=True, box=None, padding=(0, 1), expand=True)
        self._add_header(table)
        window = rows[self._offset : self._offset + self._visible]
        for offset, row in enumerate(window):
            absolute = self._offset + offset
            self._add_result_row(table, absolute + 1, row, selected=absolute == self._index)

        chosen = rows[self._index]
        combo = chosen.combo
        detail = Table(show_header=False, box=None, padding=(0, 1))
        detail.add_column("Name", style="bold cyan")
        detail.add_column("Value", justify="right")
        detail.add_row("gamma (risk)", f"{combo.gamma:g}")
        detail.add_row("kappa", f"{combo.kappa:g}")
        detail.add_row("min_spread", f"{combo.min_spread:.8f}")
        detail.add_row("max_spread_ticks", f"{combo.max_spread_ticks:g}")
        detail.add_row("max_inventory", f"{combo.max_inventory:g}")
        detail.add_row("skew_ticks", f"{combo.skew_ticks:g}")
        detail.add_row("book_skew_ticks", f"{combo.book_skew_ticks:g}")
        detail.add_row("book_skew_w", f"{combo.book_skew_size_weight:g}")
        detail.add_row("tick_size", f"{self.tick_size:.8f}")
        detail.add_row("horizon (s)", f"{self.horizon:g}")
        detail.add_row("maker_fee", f"{self.maker_fee:g}")
        detail.add_row("vol_window", str(self.volatility_window))
        if chosen.error:
            detail.add_row("error", chosen.error)
        else:
            result = chosen.result
            detail.add_row("fills", str(result.fills))
            detail.add_row("inventory", f"{result.final_inventory:.8f}")
            detail.add_row("cash", f"{result.cash:.8f}")
            detail.add_row("marked PnL", f"{result.marked_pnl:.8f}")
            detail.add_row("last spread", f"{result.last_spread:.8f}")
            detail.add_row("last gamma_eff", f"{result.last_gamma:.6g}")
            detail.add_row("last kappa_eff", f"{result.last_kappa:.6g}")
            detail.add_row("equity pts", str(len(result.equity_curve)))
            detail.add_row("days", str(len(result.daily_pnl)))

        help_text = (
            f"{self.symbol} {self.date}   resample={self.every}   "
            f"{len(rows)} combinations (sorted by marked PnL)\n"
            "↑/↓ or j/k select   Enter = equity + daily PnL   q quit"
        )
        layout = Layout(name="root")
        layout.split_column(
            Layout(Panel(help_text, title="Choose Parameters"), size=5),
            Layout(name="body"),
        )
        layout["body"].split_row(
            Layout(Panel(table, title="Grid Results"), ratio=3),
            Layout(Panel(detail, title="Selected Combo"), ratio=1),
        )
        return layout

    def _add_header(self, table: Table) -> None:
        table.add_column("#", style="bold cyan", justify="right")
        table.add_column("gamma", justify="right")
        table.add_column("kappa", justify="right")
        table.add_column("min_sp", justify="right")
        table.add_column("max_sp", justify="right")
        table.add_column("max_inv", justify="right")
        table.add_column("skew", justify="right")
        table.add_column("book", justify="right")
        table.add_column("w", justify="right")
        table.add_column("fills", justify="right")
        table.add_column("inventory", justify="right")
        table.add_column("cash", justify="right")
        table.add_column("marked PnL", justify="right")
        table.add_column("last spread", justify="right")

    def _add_result_row(
        self,
        table: Table,
        number: int,
        row: SweepRow,
        *,
        selected: bool,
    ) -> None:
        style = "reverse" if selected else ""
        if row.error:
            values = (
                str(number),
                f"{row.combo.gamma:g}",
                f"{row.combo.kappa:g}",
                f"{row.combo.min_spread:g}",
                f"{row.combo.max_spread_ticks:g}",
                f"{row.combo.max_inventory:g}",
                f"{row.combo.skew_ticks:g}",
                f"{row.combo.book_skew_ticks:g}",
                f"{row.combo.book_skew_size_weight:g}",
                "-",
                "-",
                "-",
                "ERR",
                row.error[:20],
            )
        else:
            result = row.result
            values = (
                str(number),
                f"{row.combo.gamma:g}",
                f"{row.combo.kappa:g}",
                f"{row.combo.min_spread:g}",
                f"{row.combo.max_spread_ticks:g}",
                f"{row.combo.max_inventory:g}",
                f"{row.combo.skew_ticks:g}",
                f"{row.combo.book_skew_ticks:g}",
                f"{row.combo.book_skew_size_weight:g}",
                str(result.fills),
                f"{result.final_inventory:.4f}",
                f"{result.cash:.4f}",
                f"{result.marked_pnl:.4f}",
                f"{result.last_spread:.6f}",
            )
        table.add_row(*values, style=style)


def _ascii_equity(values: list[float], width: int = 72, height: int = 12) -> str:
    """Render a filled ASCII equity chart."""
    if not values:
        return "(no equity samples)"
    width = max(10, width)
    height = max(4, height)
    if len(values) == 1:
        values = values + values
    step = max(1, len(values) // width)
    series = values[::step][:width]
    while len(series) < width:
        series.append(series[-1])
    lo = min(series)
    hi = max(series)
    if hi == lo:
        mid_row = height // 2
        lines = []
        for row in range(height):
            ch = "#" if row == mid_row else " "
            lines.append(f"{lo:10.2f} |" + (ch * width))
        return "\n".join(lines)

    lines: list[str] = []
    for row in range(height):
        level = hi - (hi - lo) * row / (height - 1)
        next_level = hi - (hi - lo) * (row + 1) / (height - 1) if row + 1 < height else lo - 1
        chars: list[str] = []
        for value in series:
            if value >= level:
                chars.append("#")
            elif value >= next_level:
                chars.append(".")
            else:
                chars.append(" ")
        label = f"{level:10.2f}"
        lines.append(f"{label} |" + "".join(chars))
    axis = " " * 11 + "+" + ("-" * width)
    first = str(values[0])
    last = str(values[-1])
    footer = " " * 12 + f"start={series[0]:.2f}   end={series[-1]:.2f}   min={lo:.2f}   max={hi:.2f}"
    return "\n".join(lines + [axis, footer])


class ResultReport:
    """Static equity curve + daily PnL page after picking a grid combo."""

    def __init__(self, symbol: str, date: str, row: SweepRow) -> None:
        self.symbol = symbol
        self.date = date
        self.row = row
        self._daily_offset = 0
        self._visible_days = 14

    def scroll(self, delta: int) -> None:
        days = len(self.row.result.daily_pnl) if not self.row.error else 0
        if days <= self._visible_days:
            self._daily_offset = 0
            return
        self._daily_offset = max(
            0,
            min(days - self._visible_days, self._daily_offset + delta),
        )

    def render(self) -> Layout:
        combo = self.row.combo
        header = (
            f"{self.symbol}  {self.date}\n"
            f"gamma={combo.gamma:g}  kappa={combo.kappa:g}  "
            f"min={combo.min_spread:g}  max_sp={combo.max_spread_ticks:g}  "
            f"inv={combo.max_inventory:g}  "
            f"skew={combo.skew_ticks:g}  book={combo.book_skew_ticks:g}  "
            f"w={combo.book_skew_size_weight:g}\n"
            "Enter/r = live replay   b = back to grid   q = quit   ↑/↓ scroll days"
        )

        if self.row.error:
            body = Panel(f"ERROR: {self.row.error}", title="Result")
            layout = Layout(name="root")
            layout.split_column(
                Layout(Panel(header, title="Combo Report"), size=6),
                Layout(body),
            )
            return layout

        result = self.row.result
        summary = Table(show_header=False, box=None, padding=(0, 1))
        summary.add_column("Name", style="bold cyan")
        summary.add_column("Value", justify="right")
        summary.add_row("fills", str(result.fills))
        summary.add_row("inventory", f"{result.final_inventory:.6f}")
        summary.add_row("cash", f"{result.cash:.4f}")
        pos = result.execution.position
        mid = result.last_mid
        summary.add_row("avg entry", f"{pos.avg_entry_price:.8f}")
        summary.add_row("realized PnL", f"{pos.realized_pnl:.4f}")
        summary.add_row("unrealized PnL", f"{pos.unrealized_pnl(mid):.4f}")
        summary.add_row("marked PnL", f"{result.marked_pnl:.4f}")
        summary.add_row("last spread", f"{result.last_spread:.8f}")
        summary.add_row("equity samples", str(len(result.equity_curve)))
        summary.add_row("trading days", str(len(result.daily_pnl)))

        equity_values = [point[1] for point in result.equity_curve]
        equity_panel = Panel(
            _ascii_equity(equity_values, width=70, height=14),
            title="Full Equity Curve (marked PnL)",
        )

        daily = Table(show_header=True, box=None, padding=(0, 1))
        daily.add_column("Date", style="bold cyan")
        daily.add_column("Day PnL", justify="right")
        daily.add_column("Cum PnL", justify="right")
        window = result.daily_pnl[
            self._daily_offset : self._daily_offset + self._visible_days
        ]
        for day, day_pnl, cum_pnl in window:
            style = "green" if day_pnl >= 0 else "red"
            daily.add_row(day, f"{day_pnl:.4f}", f"{cum_pnl:.4f}", style=style)
        if not result.daily_pnl:
            daily.add_row("-", "-", "-")

        layout = Layout(name="root")
        layout.split_column(
            Layout(Panel(header, title="Combo Report"), size=6),
            Layout(name="mid", size=8),
            Layout(name="bottom"),
        )
        layout["mid"].split_row(
            Layout(Panel(summary, title="Summary"), ratio=1),
            Layout(Panel(daily, title="Daily PnL"), ratio=2),
        )
        layout["bottom"].update(equity_panel)
        return layout

