"""MEXC hedge ledger: directional + L25-favorable taker (Part 5)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from .execution import Fill


@dataclass
class HedgeFill:
    timestamp: datetime
    side: str  # buy/sell on MEXC
    price: float  # USDT VWAP
    quantity: float
    fee_usdt: float
    mm_side: str
    mx_bid: float
    mx_ask: float
    basis_bps: float = 0.0
    levels_hit: int = 0


def _as_list(x: object | None) -> list[float]:
    if x is None:
        return []
    if isinstance(x, list):
        return [float(v) for v in x]
    try:
        return [float(v) for v in list(x)]  # type: ignore[arg-type]
    except TypeError:
        return []


def walk_favorable_bids(
    prices: list[float],
    amounts: list[float],
    *,
    bn_mid: float,
    need_qty: float,
) -> tuple[float, float, int]:
    """Hit MEXC bids at/above ``bn_mid`` (sell high). Returns qty, notional, levels."""
    if need_qty <= 0 or bn_mid <= 0:
        return 0.0, 0.0, 0
    left = need_qty
    notional = 0.0
    levels = 0
    n = min(len(prices), len(amounts))
    for i in range(n):
        if left <= 1e-15:
            break
        p, a = float(prices[i]), float(amounts[i])
        if p <= 0 or a <= 0:
            continue
        if p + 1e-12 < bn_mid:
            break  # deeper bids worse; book is sorted best→worse
        take = min(left, a)
        notional += take * p
        left -= take
        levels += 1
    filled = need_qty - left
    return filled, notional, levels


def walk_favorable_asks(
    prices: list[float],
    amounts: list[float],
    *,
    bn_mid: float,
    need_qty: float,
) -> tuple[float, float, int]:
    """Lift MEXC asks at/below ``bn_mid`` (buy low). Returns qty, notional, levels."""
    if need_qty <= 0 or bn_mid <= 0:
        return 0.0, 0.0, 0
    left = need_qty
    notional = 0.0
    levels = 0
    n = min(len(prices), len(amounts))
    for i in range(n):
        if left <= 1e-15:
            break
        p, a = float(prices[i]), float(amounts[i])
        if p <= 0 or a <= 0:
            continue
        if p - 1e-12 > bn_mid:
            break
        take = min(left, a)
        notional += take * p
        left -= take
        levels += 1
    filled = need_qty - left
    return filled, notional, levels


@dataclass
class MexcTakerHedge:
    """Opposite hedge on MEXC SUIUSDT with directional L25 walk.

    - Long BN (MM buy) → sell MEXC only on bid levels **≥ BN USDT mid**
    - Short BN (MM sell) → buy MEXC only on ask levels **≤ BN USDT mid**

    Partial fills allowed when only some depth is favorable.
    ``max_basis_bps`` still skips when |mid basis| is extreme (0=off).
    """

    taker_fee_rate: float = 0.0
    max_basis_bps: float = 0.0
    favorable_only: bool = True
    cash_usdt: float = 0.0
    inventory: float = 0.0
    fills: list[HedgeFill] = field(default_factory=list)
    skipped: int = 0
    skipped_basis: int = 0
    skipped_unfavorable: int = 0
    partial_qty: float = 0.0  # MM qty not hedged (unfavorable / thin)

    def on_mm_fill(
        self,
        mm_fill: Fill,
        *,
        mx_bid: float,
        mx_ask: float,
        mx_mid: float = 0.0,
        bn_usdt_mid: float = 0.0,
        bid_prices: list[float] | None = None,
        bid_amounts: list[float] | None = None,
        ask_prices: list[float] | None = None,
        ask_amounts: list[float] | None = None,
    ) -> HedgeFill | None:
        if mm_fill.quantity <= 0:
            return None
        if not (0 < mx_bid < mx_ask):
            self.skipped += 1
            return None

        mid = mx_mid if mx_mid > 0 else 0.5 * (mx_bid + mx_ask)
        basis_bps = 0.0
        if bn_usdt_mid > 0 and mid > 0:
            basis_bps = (mid / bn_usdt_mid - 1.0) * 1e4
            if self.max_basis_bps > 0 and abs(basis_bps) > self.max_basis_bps:
                self.skipped_basis += 1
                self.partial_qty += float(mm_fill.quantity)
                return None

        need = float(mm_fill.quantity)
        bp = _as_list(bid_prices)
        ba = _as_list(bid_amounts)
        ap = _as_list(ask_prices)
        aa = _as_list(ask_amounts)

        if mm_fill.side == "buy":
            # Short MEXC — only if MEXC ≥ BN
            if self.favorable_only and bn_usdt_mid > 0:
                if bp and ba:
                    qty, notional, levels = walk_favorable_bids(
                        bp, ba, bn_mid=bn_usdt_mid, need_qty=need
                    )
                elif mx_bid + 1e-12 >= bn_usdt_mid:
                    qty, notional, levels = need, need * mx_bid, 1
                else:
                    qty, notional, levels = 0.0, 0.0, 0
            else:
                qty, notional, levels = need, need * mx_bid, 1
            if qty <= 1e-15:
                self.skipped_unfavorable += 1
                self.partial_qty += need
                return None
            if qty + 1e-12 < need:
                self.partial_qty += need - qty
            price = notional / qty
            fee = notional * self.taker_fee_rate
            self.cash_usdt += notional - fee
            self.inventory -= qty
            side = "sell"
        elif mm_fill.side == "sell":
            # Long MEXC — only if MEXC ≤ BN
            if self.favorable_only and bn_usdt_mid > 0:
                if ap and aa:
                    qty, notional, levels = walk_favorable_asks(
                        ap, aa, bn_mid=bn_usdt_mid, need_qty=need
                    )
                elif mx_ask - 1e-12 <= bn_usdt_mid:
                    qty, notional, levels = need, need * mx_ask, 1
                else:
                    qty, notional, levels = 0.0, 0.0, 0
            else:
                qty, notional, levels = need, need * mx_ask, 1
            if qty <= 1e-15:
                self.skipped_unfavorable += 1
                self.partial_qty += need
                return None
            if qty + 1e-12 < need:
                self.partial_qty += need - qty
            price = notional / qty
            fee = notional * self.taker_fee_rate
            self.cash_usdt -= notional + fee
            self.inventory += qty
            side = "buy"
        else:
            return None

        hf = HedgeFill(
            timestamp=mm_fill.timestamp,
            side=side,
            price=price,
            quantity=qty,
            fee_usdt=fee,
            mm_side=mm_fill.side,
            mx_bid=mx_bid,
            mx_ask=mx_ask,
            basis_bps=basis_bps,
            levels_hit=levels,
        )
        self.fills.append(hf)
        return hf

    def marked_usdc(self, *, fx_mid: float, mx_mid_usdt: float) -> float:
        if fx_mid <= 0:
            return 0.0
        return (self.cash_usdt + self.inventory * mx_mid_usdt) / fx_mid

    def net_sui(self, bn_inventory: float) -> float:
        return bn_inventory + self.inventory
