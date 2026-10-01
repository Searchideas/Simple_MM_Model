"""Incremental L2 order book from Tardis ``incremental_book_L2`` rows."""

from __future__ import annotations

from dataclasses import dataclass, field


def _tick_key(price: float, tick: float) -> float:
    if tick <= 0:
        return price
    return round(round(price / tick) * tick, 10)


@dataclass
class L2Book:
    """Local L2 book: amount is absolute level size; 0 deletes the level."""

    bids: dict[float, float] = field(default_factory=dict)
    asks: dict[float, float] = field(default_factory=dict)
    tick: float = 0.0001

    def clear(self) -> None:
        self.bids.clear()
        self.asks.clear()

    def apply(
        self,
        *,
        side: str,
        price: float,
        amount: float,
        is_snapshot: bool,
        prev_was_snapshot: bool,
    ) -> tuple[float, float]:
        """Apply one row. Returns ``(old_amount, new_amount)`` at that price."""
        if is_snapshot and not prev_was_snapshot:
            self.clear()
        if price <= 0:
            return 0.0, 0.0
        key = _tick_key(price, self.tick)
        book = self.bids if side == "bid" else self.asks
        old = float(book.get(key, 0.0))
        if amount <= 0:
            book.pop(key, None)
            new = 0.0
        else:
            book[key] = float(amount)
            new = float(amount)
        return old, new

    def best_bid(self) -> tuple[float, float]:
        if not self.bids:
            return 0.0, 0.0
        p = max(self.bids)
        return p, self.bids[p]

    def best_ask(self) -> tuple[float, float]:
        if not self.asks:
            return 0.0, 0.0
        p = min(self.asks)
        return p, self.asks[p]

    def mid(self) -> float:
        bb, _ = self.best_bid()
        ba, _ = self.best_ask()
        if bb > 0 and ba > 0:
            return 0.5 * (bb + ba)
        return 0.0

    def microprice(self) -> float:
        """Size-weighted mid. More size on the ask pulls this below mid."""
        bb, bq = self.best_bid()
        ba, aq = self.best_ask()
        if bb <= 0 or ba <= 0 or bq + aq <= 0:
            return 0.0
        return (bb * aq + ba * bq) / (bq + aq)

    def size_ahead(self, *, side: str, our_price: float) -> float:
        """Resting size ahead when joining back of ``our_price``."""
        if our_price <= 0:
            return 0.0
        key = _tick_key(our_price, self.tick)
        ahead = 0.0
        if side == "bid":
            for p, a in self.bids.items():
                if a > 0 and p >= key:
                    ahead += a
        else:
            for p, a in self.asks.items():
                if a > 0 and p <= key:
                    ahead += a
        return ahead
