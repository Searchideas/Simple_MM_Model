"""Post-fill mark-out scoring (adverse selection diagnostics)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class _Pending:
    ts: datetime
    side: str
    mid: float
    price: float
    quantity: float


@dataclass
class MarkoutReport:
    n_scored: int = 0
    n_adverse: int = 0
    adverse_rate: float = 0.0
    mean_markout_ticks: float = 0.0
    mean_adverse_ticks: float = 0.0
    total_markout_pnl_proxy: float = 0.0
    total_adverse_pnl_proxy: float = 0.0


@dataclass
class MarkoutTracker:
    """Score own fills by mid move over ``horizon_seconds``.

    Mark-out (ticks), same sign as live ToxicityTracker:
      buy  → (mid_h − mid_fill) / tick
      sell → (mid_fill − mid_h) / tick
    Negative ⇒ adverse.
    """

    tick: float
    horizon_seconds: float = 1.0
    _pending: deque[_Pending] = field(default_factory=deque)
    _markouts: list[float] = field(default_factory=list)
    _adverse: list[bool] = field(default_factory=list)
    _pnl_proxy: list[float] = field(default_factory=list)

    def on_fill(
        self,
        *,
        side: str,
        mid: float,
        price: float,
        quantity: float,
        ts: datetime,
    ) -> None:
        s = side.lower()
        if s not in ("buy", "sell") or mid <= 0 or self.tick <= 0:
            return
        self._pending.append(
            _Pending(ts=ts, side=s, mid=float(mid), price=price, quantity=quantity)
        )

    def on_mid(self, mid: float, ts: datetime) -> int:
        """Resolve pending fills past the horizon. Returns newly scored count."""
        if mid <= 0 or self.tick <= 0:
            return 0
        scored = 0
        while self._pending:
            fill = self._pending[0]
            age = (ts - fill.ts).total_seconds()
            if age < self.horizon_seconds:
                break
            self._pending.popleft()
            self._score(fill, mid)
            scored += 1
        return scored

    def flush(self, mid: float) -> int:
        """Score remaining pending at end of run."""
        if mid <= 0 or not self._pending:
            return 0
        scored = 0
        while self._pending:
            fill = self._pending.popleft()
            self._score(fill, mid)
            scored += 1
        return scored

    def _score(self, fill: _Pending, mid: float) -> None:
        if fill.side == "buy":
            markout = (mid - fill.mid) / self.tick
        else:
            markout = (fill.mid - mid) / self.tick
        adverse = markout < 0.0
        mid_move = mid - fill.mid
        signed = (1.0 if fill.side == "buy" else -1.0) * fill.quantity
        self._markouts.append(markout)
        self._adverse.append(adverse)
        self._pnl_proxy.append(signed * mid_move)

    def report(self) -> MarkoutReport:
        n = len(self._markouts)
        if n == 0:
            return MarkoutReport()
        n_adv = sum(1 for a in self._adverse if a)
        return MarkoutReport(
            n_scored=n,
            n_adverse=n_adv,
            adverse_rate=n_adv / n,
            mean_markout_ticks=sum(self._markouts) / n,
            mean_adverse_ticks=sum(max(0.0, -m) for m in self._markouts) / n,
            total_markout_pnl_proxy=sum(self._pnl_proxy),
            total_adverse_pnl_proxy=sum(
                p for p, a in zip(self._pnl_proxy, self._adverse) if a
            ),
        )
