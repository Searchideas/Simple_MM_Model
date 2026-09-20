"""Part 4: cross-quote backtest risk / attribution metrics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime

from .execution import Fill
from .markout import MarkoutReport
from .market_state import PositionState


def action_family(action: str) -> str:
    """Bucket live-parity action tags into report families."""
    a = (action or "").upper()
    if a.startswith("FLATTEN") or "FLATTEN" in a:
        return "FLATTEN"
    if a.startswith("TRAIL"):
        return "TRAIL"
    if a.startswith("CUT") or a.startswith("HOLD"):
        return "CUT"
    if a.startswith("JOIN") or a in {"TAKE_PROFIT", "WAIT_TP"}:
        return "JOIN_TP"
    if a.startswith("COVER"):
        return "COVER"
    if a.startswith("REDUCE"):
        return "REDUCE"
    if a.startswith("QUOTE"):
        return "QUOTE"
    if a in {"REJECT", "REJECT_CROSS", "FX_STALE"}:
        return "REJECT"
    return a or "OTHER"


@dataclass
class FamilyStats:
    quote_steps: int = 0
    fills: int = 0
    qty: float = 0.0


@dataclass
class CrossQuoteMetrics:
    """Summary risk + attribution for one multi-book run."""

    marked_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    cash: float = 0.0
    final_inventory: float = 0.0
    max_abs_inventory: float = 0.0
    fills: int = 0
    fill_buy_qty: float = 0.0
    fill_sell_qty: float = 0.0
    # Equity / drawdown from sampled equity curve.
    peak_equity: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    # Daily
    n_days: int = 0
    n_win_days: int = 0
    best_day: float = 0.0
    worst_day: float = 0.0
    mean_day: float = 0.0
    # Markout passthrough
    markout: MarkoutReport = field(default_factory=MarkoutReport)
    # Action attribution
    by_family: dict[str, FamilyStats] = field(default_factory=dict)

    def format_lines(self) -> list[str]:
        lines = [
            "metrics:",
            (
                f"  pnl marked={self.marked_pnl:+.6f} "
                f"realized={self.realized_pnl:+.6f} "
                f"unrealized={self.unrealized_pnl:+.6f} cash={self.cash:.6f}"
            ),
            (
                f"  inv final={self.final_inventory:.4f} "
                f"max_abs={self.max_abs_inventory:.4f} "
                f"fills={self.fills} "
                f"buy_qty={self.fill_buy_qty:.4f} sell_qty={self.fill_sell_qty:.4f}"
            ),
            (
                f"  equity peak={self.peak_equity:+.6f} "
                f"max_dd={self.max_drawdown:+.6f} "
                f"max_dd_pct={100.0 * self.max_drawdown_pct:.2f}%"
            ),
        ]
        if self.n_days:
            lines.append(
                f"  daily n={self.n_days} win={self.n_win_days} "
                f"({100.0 * self.n_win_days / self.n_days:.0f}%) "
                f"mean={self.mean_day:+.6f} "
                f"best={self.best_day:+.6f} worst={self.worst_day:+.6f}"
            )
        m = self.markout
        if m.n_scored:
            lines.append(
                f"  markout scored={m.n_scored} "
                f"adverse={m.n_adverse} ({100.0 * m.adverse_rate:.1f}%) "
                f"mean_ticks={m.mean_markout_ticks:+.2f} "
                f"pnl_proxy={m.total_markout_pnl_proxy:+.6f}"
            )
        if self.by_family:
            lines.append("  attribution (quote_steps / fills / qty):")
            order = [
                "QUOTE",
                "COVER",
                "JOIN_TP",
                "CUT",
                "TRAIL",
                "FLATTEN",
                "REDUCE",
                "REJECT",
                "OTHER",
            ]
            keys = [k for k in order if k in self.by_family] + [
                k for k in sorted(self.by_family) if k not in order
            ]
            for k in keys:
                s = self.by_family[k]
                if s.quote_steps == 0 and s.fills == 0:
                    continue
                lines.append(
                    f"    {k:<8} steps={s.quote_steps:7d} "
                    f"fills={s.fills:6d} qty={s.qty:.4f}"
                )
        return lines


def equity_drawdown(
    equity_curve: list[tuple[datetime, float]],
) -> tuple[float, float, float]:
    """Return (peak, max_drawdown_abs ≤ 0, max_drawdown_pct ≥ 0)."""
    if not equity_curve:
        return 0.0, 0.0, 0.0
    peak = equity_curve[0][1]
    max_dd = 0.0
    max_dd_pct = 0.0
    for _, eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = eq - peak
        if dd < max_dd:
            max_dd = dd
            max_dd_pct = (-dd / peak) if peak > 1e-12 else 0.0
    return peak, max_dd, max_dd_pct


def build_metrics(
    *,
    position: PositionState,
    marked_pnl: float,
    last_mid: float,
    fills: list[Fill],
    equity_curve: list[tuple[datetime, float]],
    daily_pnl: list[tuple[str, float, float]],
    markout: MarkoutReport,
    action_steps: Counter[str],
    max_abs_inventory: float,
) -> CrossQuoteMetrics:
    unrealized = 0.0
    if abs(position.inventory) > 1e-15 and position.avg_entry_price > 0 and last_mid > 0:
        unrealized = position.inventory * (last_mid - position.avg_entry_price)
    peak, max_dd, max_dd_pct = equity_drawdown(equity_curve)

    day_pnls = [d for _, d, _ in daily_pnl]
    n_days = len(day_pnls)
    n_win = sum(1 for d in day_pnls if d > 0)
    best = max(day_pnls) if day_pnls else 0.0
    worst = min(day_pnls) if day_pnls else 0.0
    mean_day = (sum(day_pnls) / n_days) if n_days else 0.0

    by_family: dict[str, FamilyStats] = {}
    for action, n in action_steps.items():
        fam = action_family(action)
        by_family.setdefault(fam, FamilyStats()).quote_steps += int(n)

    buy_qty = 0.0
    sell_qty = 0.0
    for fill in fills:
        fam = action_family(getattr(fill, "action", "") or "")
        st = by_family.setdefault(fam, FamilyStats())
        st.fills += 1
        st.qty += float(fill.quantity)
        if fill.side == "buy":
            buy_qty += fill.quantity
        else:
            sell_qty += fill.quantity

    return CrossQuoteMetrics(
        marked_pnl=marked_pnl,
        realized_pnl=float(position.realized_pnl),
        unrealized_pnl=unrealized,
        cash=float(position.cash),
        final_inventory=float(position.inventory),
        max_abs_inventory=float(max_abs_inventory),
        fills=len(fills),
        fill_buy_qty=buy_qty,
        fill_sell_qty=sell_qty,
        peak_equity=peak,
        max_drawdown=max_dd,
        max_drawdown_pct=max_dd_pct,
        n_days=n_days,
        n_win_days=n_win,
        best_day=best,
        worst_day=worst,
        mean_day=mean_day,
        markout=markout,
        by_family=by_family,
    )
