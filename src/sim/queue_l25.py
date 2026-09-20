"""L25 queue position helpers for maker fill simulation (Part 3B)."""

from __future__ import annotations


def size_ahead_at_price(
    prices: list[float] | tuple[float, ...] | None,
    amounts: list[float] | tuple[float, ...] | None,
    our_price: float,
    *,
    side: str,
    tick: float,
) -> float:
    """Resting size ahead of us when joining ``our_price`` on ``side``.

    - Better prices (more aggressive) must clear before we are reached.
    - Same price: assume we join the **back** of the queue (all size at
      that level is ahead).

    ``side`` is our quote side: ``\"bid\"`` or ``\"ask\"``.
    """
    if our_price <= 0 or tick <= 0:
        return 0.0
    if not prices or not amounts:
        return 0.0

    eps = max(tick * 0.51, 1e-12)
    ahead = 0.0
    n = min(len(prices), len(amounts))
    for i in range(n):
        p = float(prices[i])
        a = float(amounts[i])
        if p <= 0 or a <= 0:
            continue
        if side == "bid":
            # Higher bid is better; same price = ahead of us.
            if p > our_price + eps or abs(p - our_price) <= eps:
                ahead += a
        else:
            # Lower ask is better; same price = ahead of us.
            if p < our_price - eps or abs(p - our_price) <= eps:
                ahead += a
    return ahead


def top_amount(
    prices: list[float] | tuple[float, ...] | None,
    amounts: list[float] | tuple[float, ...] | None,
    *,
    side: str,
    tick: float,
    touch_price: float,
) -> float:
    """Amount at the touch price level (fallback if lists empty)."""
    if touch_price <= 0:
        return 0.0
    if prices and amounts:
        eps = max(tick * 0.51, 1e-12)
        total = 0.0
        n = min(len(prices), len(amounts))
        for i in range(n):
            p = float(prices[i])
            a = float(amounts[i])
            if a > 0 and abs(p - touch_price) <= eps:
                total += a
        if total > 0:
            return total
    return 0.0
