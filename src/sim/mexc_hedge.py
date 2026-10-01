"""MEXC hedge ledger: directional L25 + residual clear (hybrid cover)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from .execution import Fill


@dataclass
class HedgeFill:
    timestamp: datetime
    side: str
    price: float
    quantity: float
    fee_usdt: float
    mm_side: str
    mx_bid: float
    mx_ask: float
    basis_bps: float = 0.0
    levels_hit: int = 0
    threshold_usdt: float = 0.0


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
    threshold: float,
    need_qty: float,
) -> tuple[float, float, int]:
    """Sell into bids at/above ``threshold``."""
    if need_qty <= 0 or threshold <= 0:
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
        if p + 1e-12 < threshold:
            break
        take = min(left, a)
        notional += take * p
        left -= take
        levels += 1
    return need_qty - left, notional, levels


def walk_favorable_asks(
    prices: list[float],
    amounts: list[float],
    *,
    threshold: float,
    need_qty: float,
) -> tuple[float, float, int]:
    """Buy from asks at/below ``threshold``."""
    if need_qty <= 0 or threshold <= 0:
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
        if p - 1e-12 > threshold:
            break
        take = min(left, a)
        notional += take * p
        left -= take
        levels += 1
    return need_qty - left, notional, levels


@dataclass
class MexcTakerHedge:
    """MEXC hedge with fill-price / mid threshold and residual clearing.

    Long BN → sell MEXC only if bid ≥ threshold_usdt (SUIUSDC_fill × FX).
    Short BN → buy MEXC only if ask ≤ threshold_usdt.
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
    partial_qty: float = 0.0
    residual_clears: int = 0

    def _apply_sell(
        self,
        *,
        qty: float,
        notional: float,
        levels: int,
        ts: datetime,
        mm_side: str,
        mx_bid: float,
        mx_ask: float,
        basis_bps: float,
        threshold: float,
    ) -> HedgeFill | None:
        if qty <= 1e-15:
            return None
        price = notional / qty
        fee = notional * self.taker_fee_rate
        self.cash_usdt += notional - fee
        self.inventory -= qty
        hf = HedgeFill(
            timestamp=ts,
            side="sell",
            price=price,
            quantity=qty,
            fee_usdt=fee,
            mm_side=mm_side,
            mx_bid=mx_bid,
            mx_ask=mx_ask,
            basis_bps=basis_bps,
            levels_hit=levels,
            threshold_usdt=threshold,
        )
        self.fills.append(hf)
        return hf

    def _apply_buy(
        self,
        *,
        qty: float,
        notional: float,
        levels: int,
        ts: datetime,
        mm_side: str,
        mx_bid: float,
        mx_ask: float,
        basis_bps: float,
        threshold: float,
    ) -> HedgeFill | None:
        if qty <= 1e-15:
            return None
        price = notional / qty
        fee = notional * self.taker_fee_rate
        self.cash_usdt -= notional + fee
        self.inventory += qty
        hf = HedgeFill(
            timestamp=ts,
            side="buy",
            price=price,
            quantity=qty,
            fee_usdt=fee,
            mm_side=mm_side,
            mx_bid=mx_bid,
            mx_ask=mx_ask,
            basis_bps=basis_bps,
            levels_hit=levels,
            threshold_usdt=threshold,
        )
        self.fills.append(hf)
        return hf

    def _hedge_sell_qty(
        self,
        need: float,
        *,
        threshold: float,
        mx_bid: float,
        mx_ask: float,
        bid_prices: list[float],
        bid_amounts: list[float],
        ts: datetime,
        mm_side: str,
        basis_bps: float,
    ) -> HedgeFill | None:
        if self.favorable_only and threshold > 0:
            if bid_prices and bid_amounts:
                qty, notional, levels = walk_favorable_bids(
                    bid_prices, bid_amounts, threshold=threshold, need_qty=need
                )
            elif mx_bid + 1e-12 >= threshold:
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
        return self._apply_sell(
            qty=qty,
            notional=notional,
            levels=levels,
            ts=ts,
            mm_side=mm_side,
            mx_bid=mx_bid,
            mx_ask=mx_ask,
            basis_bps=basis_bps,
            threshold=threshold,
        )

    def _hedge_buy_qty(
        self,
        need: float,
        *,
        threshold: float,
        mx_bid: float,
        mx_ask: float,
        ask_prices: list[float],
        ask_amounts: list[float],
        ts: datetime,
        mm_side: str,
        basis_bps: float,
    ) -> HedgeFill | None:
        if self.favorable_only and threshold > 0:
            if ask_prices and ask_amounts:
                qty, notional, levels = walk_favorable_asks(
                    ask_prices, ask_amounts, threshold=threshold, need_qty=need
                )
            elif mx_ask - 1e-12 <= threshold:
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
        return self._apply_buy(
            qty=qty,
            notional=notional,
            levels=levels,
            ts=ts,
            mm_side=mm_side,
            mx_bid=mx_bid,
            mx_ask=mx_ask,
            basis_bps=basis_bps,
            threshold=threshold,
        )

    def on_mm_fill(
        self,
        mm_fill: Fill,
        *,
        mx_bid: float,
        mx_ask: float,
        mx_mid: float = 0.0,
        bn_usdt_mid: float = 0.0,
        fx_mid: float = 0.0,
        bid_prices: list[float] | None = None,
        bid_amounts: list[float] | None = None,
        ask_prices: list[float] | None = None,
        ask_amounts: list[float] | None = None,
    ) -> HedgeFill | None:
        """Hedge one MM fill; threshold = fill_usdc × FX (else BN USDT mid)."""
        if mm_fill.quantity <= 0:
            return None
        if not (0 < mx_bid < mx_ask):
            self.skipped += 1
            return None

        mid = mx_mid if mx_mid > 0 else 0.5 * (mx_bid + mx_ask)
        # Break-even vs SUIUSDC fill in USDT terms.
        if fx_mid > 0 and mm_fill.price > 0:
            threshold = float(mm_fill.price) * float(fx_mid)
        else:
            threshold = float(bn_usdt_mid) if bn_usdt_mid > 0 else mid

        basis_bps = 0.0
        if threshold > 0 and mid > 0:
            basis_bps = (mid / threshold - 1.0) * 1e4
            if self.max_basis_bps > 0 and abs(basis_bps) > self.max_basis_bps:
                self.skipped_basis += 1
                self.partial_qty += float(mm_fill.quantity)
                return None

        need = float(mm_fill.quantity)
        bp, ba = _as_list(bid_prices), _as_list(bid_amounts)
        ap, aa = _as_list(ask_prices), _as_list(ask_amounts)

        if mm_fill.side == "buy":
            return self._hedge_sell_qty(
                need,
                threshold=threshold,
                mx_bid=mx_bid,
                mx_ask=mx_ask,
                bid_prices=bp,
                bid_amounts=ba,
                ts=mm_fill.timestamp,
                mm_side="buy",
                basis_bps=basis_bps,
            )
        if mm_fill.side == "sell":
            return self._hedge_buy_qty(
                need,
                threshold=threshold,
                mx_bid=mx_bid,
                mx_ask=mx_ask,
                ask_prices=ap,
                ask_amounts=aa,
                ts=mm_fill.timestamp,
                mm_side="sell",
                basis_bps=basis_bps,
            )
        return None

    def try_clear_residual(
        self,
        bn_inventory: float,
        *,
        threshold_usdt: float,
        mx_bid: float,
        mx_ask: float,
        mx_mid: float,
        ts: datetime,
        bid_prices: list[float] | None = None,
        bid_amounts: list[float] | None = None,
        ask_prices: list[float] | None = None,
        ask_amounts: list[float] | None = None,
    ) -> HedgeFill | None:
        """Keep monitoring MEXC to flatten net = bn + mx when price is good."""
        if not (0 < mx_bid < mx_ask):
            return None
        net = bn_inventory + self.inventory
        if abs(net) < 1e-12:
            return None
        mid = mx_mid if mx_mid > 0 else 0.5 * (mx_bid + mx_ask)
        thresh = threshold_usdt if threshold_usdt > 0 else mid
        basis_bps = (mid / thresh - 1.0) * 1e4 if thresh > 0 and mid > 0 else 0.0
        bp, ba = _as_list(bid_prices), _as_list(bid_amounts)
        ap, aa = _as_list(ask_prices), _as_list(ask_amounts)

        if net > 1e-12:
            hf = self._hedge_sell_qty(
                net,
                threshold=thresh,
                mx_bid=mx_bid,
                mx_ask=mx_ask,
                bid_prices=bp,
                bid_amounts=ba,
                ts=ts,
                mm_side="residual",
                basis_bps=basis_bps,
            )
        else:
            hf = self._hedge_buy_qty(
                -net,
                threshold=thresh,
                mx_bid=mx_bid,
                mx_ask=mx_ask,
                ask_prices=ap,
                ask_amounts=aa,
                ts=ts,
                mm_side="residual",
                basis_bps=basis_bps,
            )
        if hf is not None:
            self.residual_clears += 1
        return hf

    def marked_usdc(self, *, fx_mid: float, mx_mid_usdt: float) -> float:
        if fx_mid <= 0:
            return 0.0
        return (self.cash_usdt + self.inventory * mx_mid_usdt) / fx_mid

    def net_sui(self, bn_inventory: float) -> float:
        return bn_inventory + self.inventory
