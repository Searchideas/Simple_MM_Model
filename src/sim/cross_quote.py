"""Live-parity helpers: FX convert, maker clamp, inventory, cover/join.

Ported from Binance Streaming ``live_quote.py`` for cross-quote backtests
(Parts 1–2 of docs/CROSS_QUOTE_ROADMAP.md).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .quoting_logic import Quote


REFERENCE_BASIS_BPS = 4.0
MAKER_SAFETY_TICKS = 2.0


def _round_down(price: float, tick: float) -> float:
    return math.floor(price / tick + 1e-12) * tick


def _round_up(price: float, tick: float) -> float:
    return math.ceil(price / tick - 1e-12) * tick


def usdt_to_usdc(price_usdt: float, fx_mid: float, basis_bps: float) -> float:
    """SUIUSDC = SUIUSDT / USDCUSDT × (1 − basis_bps/1e4)."""
    if fx_mid <= 0 or price_usdt <= 0:
        return 0.0
    return price_usdt / fx_mid * (1.0 - basis_bps / 10_000.0)


def clamp_maker_no_cross(
    bid: float,
    ask: float,
    *,
    trade_bid: float,
    trade_ask: float,
    tick: float,
    safety_ticks: float = MAKER_SAFETY_TICKS,
) -> tuple[float, float] | None:
    """Pin quotes on the trade (USDC) book so GTX cannot take liquidity."""
    if tick <= 0 or not (0 < trade_bid < trade_ask):
        return None

    bid = _round_down(bid, tick)
    ask = _round_up(ask, tick)

    safety = safety_ticks * tick
    bid = min(bid, trade_ask - safety)
    ask = max(ask, trade_bid + safety)

    bid = _round_down(bid, tick)
    ask = _round_up(ask, tick)

    if bid >= ask:
        bid = _round_down(trade_bid, tick)
        ask = _round_up(trade_ask, tick)
        if bid >= ask:
            ask = bid + tick

    if not (0 < bid < ask):
        return None
    if bid >= trade_ask or ask <= trade_bid:
        return None
    return bid, ask


def inventory_permissions(
    inventory: float,
    *,
    max_inventory: float,
    effective_order_qty: float,
    soft_inventory_lots: float = 0.0,
) -> tuple[bool, bool]:
    """Allow both sides near flat, then only the inventory-reducing side.

    ``soft_inventory_lots=0`` → both sides at q≈0; one-side as soon as |q|>0.
    ``soft_inventory_lots=1`` → both sides until about one executable lot.
    Uses ``<=`` / ``>=`` so flat inventory is two-sided (live used strict
    inequalities which disabled both sides at q=0).
    """
    if inventory >= max_inventory:
        return False, True
    if inventory <= -max_inventory:
        return True, False

    soft_inventory = max(0.0, soft_inventory_lots) * effective_order_qty
    enable_bid = (
        inventory <= soft_inventory
        and inventory + effective_order_qty <= max_inventory
    )
    enable_ask = (
        inventory >= -soft_inventory
        and inventory - effective_order_qty >= -max_inventory
    )
    return enable_bid, enable_ask


def leveraged_roi(
    inventory: float,
    entry: float,
    mark: float,
    leverage: float,
) -> float:
    """Unrealized ROI as a fraction of margin at ``leverage``."""
    if abs(inventory) < 1e-12 or entry <= 0 or mark <= 0 or leverage <= 0:
        return 0.0
    notional = abs(inventory) * entry
    if notional <= 0:
        return 0.0
    upnl = inventory * (mark - entry)
    return (upnl * leverage) / notional


def join_bbo_for_reduce(
    bid: float,
    ask: float,
    *,
    inventory: float,
    enable_bid: bool,
    enable_ask: bool,
    trade_bid: float,
    trade_ask: float,
    tick: float,
) -> tuple[float, float, bool]:
    """Join touch on the inventory-reducing side (maker, no cross)."""
    if tick <= 0 or not (0 < trade_bid < trade_ask):
        return bid, ask, False

    joined = False
    if enable_ask and not enable_bid and inventory > 0:
        ask = _round_up(trade_ask, tick)
        ask = max(ask, _round_up(trade_bid + tick, tick))
        bid = min(bid, ask - tick)
        joined = True
    elif enable_bid and not enable_ask and inventory < 0:
        bid = _round_down(trade_bid, tick)
        bid = min(bid, _round_down(trade_ask - tick, tick))
        ask = max(ask, bid + tick)
        joined = True

    if not (0 < bid < ask):
        return bid, ask, False
    if bid >= trade_ask or ask <= trade_bid:
        return bid, ask, False
    return bid, ask, joined


def trail_reduce_roi_away(
    bid: float,
    ask: float,
    *,
    inventory: float,
    enable_bid: bool,
    enable_ask: bool,
    trade_bid: float,
    trade_ask: float,
    trade_mid: float,
    tick: float,
    roi_pct: float,
    leverage: float,
) -> tuple[float, float, bool]:
    """Maker reduce order trailing mid by ``roi_pct`` @ ``leverage`` (price frac).

    Short → bid at mid·(1 − roi/lev); long → ask at mid·(1 + roi/lev).
    Recomputed every cycle so the working order moves with the book.
    """
    if tick <= 0 or roi_pct <= 0 or leverage <= 0:
        return bid, ask, False
    if not (0 < trade_bid < trade_ask):
        return bid, ask, False
    mid = trade_mid if trade_mid > 0 else 0.5 * (trade_bid + trade_ask)
    if mid <= 0:
        return bid, ask, False
    price_frac = (roi_pct / 100.0) / leverage
    trailed = False
    if enable_ask and not enable_bid and inventory > 0:
        ask = _round_up(mid * (1.0 + price_frac), tick)
        ask = max(ask, _round_up(trade_bid + tick, tick))
        bid = min(bid, ask - tick)
        trailed = True
    elif enable_bid and not enable_ask and inventory < 0:
        bid = _round_down(mid * (1.0 - price_frac), tick)
        bid = min(bid, _round_down(trade_ask - tick, tick))
        ask = max(ask, bid + tick)
        trailed = True

    if not (0 < bid < ask):
        return bid, ask, False
    if bid >= trade_ask or ask <= trade_bid:
        return bid, ask, False
    return bid, ask, trailed


def cover_at_reservation(
    bid: float,
    ask: float,
    *,
    inventory: float,
    enable_bid: bool,
    enable_ask: bool,
    reservation: float,
    trade_bid: float,
    trade_ask: float,
    tick: float,
) -> tuple[float, float, bool]:
    """Flatten using AS reservation; maker-safe, no BBO hardcode."""
    if tick <= 0 or reservation <= 0 or not (0 < trade_bid < trade_ask):
        return bid, ask, False

    covered = False
    if enable_ask and not enable_bid and inventory > 0:
        ask = _round_up(reservation, tick)
        ask = max(ask, _round_up(trade_bid + tick, tick))
        bid = min(bid, ask - tick)
        covered = True
    elif enable_bid and not enable_ask and inventory < 0:
        bid = _round_down(reservation, tick)
        bid = min(bid, _round_down(trade_ask - tick, tick))
        ask = max(ask, bid + tick)
        covered = True

    if not (0 < bid < ask):
        return bid, ask, False
    if bid >= trade_ask or ask <= trade_bid:
        return bid, ask, False
    return bid, ask, covered


@dataclass
class CrossQuoteParams:
    """Runtime knobs matching live Settings (v1 subset)."""

    reference_basis_bps: float = REFERENCE_BASIS_BPS
    maker_safety_ticks: float = MAKER_SAFETY_TICKS
    fx_max_age_seconds: float = 5.0
    soft_inventory_lots: float = 0.0
    cover_join_roi_pct: float = 5.0
    cover_cut_roi_pct: float = 12.0
    cover_trail_roi_pct: float = 1.0
    cover_join_leverage: float = 10.0
    cover_max_hold_seconds: float = 0.0
    trade_tick: float = 0.0001
    vol_floor: float = 0.0001
    order_size: float = 10.0
    max_inventory: float = 50.0
    hard_flatten_at_max: bool = True


@dataclass
class CrossQuoteDecision:
    """Result of one cross-quote cycle (posted on SUIUSDC)."""

    quote: Quote | None
    action: str
    bid_usdc: float
    ask_usdc: float
    fx_mid: float
    enable_bid: bool
    enable_ask: bool
    size: float
    joined_bbo: bool = False
    cover_tight: bool = False
    cut_exit: bool = False
    trail_cut: bool = False
    hard_flatten: bool = False
    inv_roi: float = 0.0
    basis_residual_bps: float = 0.0


def build_cross_quote(
    *,
    as_quote: Quote,
    inventory: float,
    avg_entry: float,
    trade_bid: float,
    trade_ask: float,
    fx_mid: float,
    params: CrossQuoteParams,
    inventory_open_age_seconds: float = 0.0,
) -> CrossQuoteDecision:
    """AS (USDT) → FX → clamp → inventory / cover gates → working Quote."""
    if fx_mid <= 0:
        return CrossQuoteDecision(
            quote=None,
            action="FX_STALE",
            bid_usdc=0.0,
            ask_usdc=0.0,
            fx_mid=fx_mid,
            enable_bid=False,
            enable_ask=False,
            size=0.0,
        )

    bid_usdc = usdt_to_usdc(as_quote.bid, fx_mid, params.reference_basis_bps)
    ask_usdc = usdt_to_usdc(as_quote.ask, fx_mid, params.reference_basis_bps)
    r_usdc = usdt_to_usdc(
        as_quote.reservation_price, fx_mid, params.reference_basis_bps
    )

    trade_mid = 0.5 * (trade_bid + trade_ask) if trade_bid > 0 and trade_ask > 0 else 0.0
    # Residual: trade mid vs raw SUIUSDT reservation / FX (no basis haircut).
    raw_mid_usdc = (
        (as_quote.reservation_price / fx_mid) if fx_mid > 0 else 0.0
    )
    basis_residual_bps = 0.0
    if trade_mid > 0 and raw_mid_usdc > 0:
        basis_residual_bps = (trade_mid / raw_mid_usdc - 1.0) * 10_000.0

    clamped = clamp_maker_no_cross(
        bid_usdc,
        ask_usdc,
        trade_bid=trade_bid,
        trade_ask=trade_ask,
        tick=params.trade_tick,
        safety_ticks=params.maker_safety_ticks,
    )
    if clamped is None:
        return CrossQuoteDecision(
            quote=None,
            action="REJECT_CROSS",
            bid_usdc=bid_usdc,
            ask_usdc=ask_usdc,
            fx_mid=fx_mid,
            enable_bid=False,
            enable_ask=False,
            size=0.0,
            basis_residual_bps=basis_residual_bps,
        )

    bid_px, ask_px = clamped
    q = inventory
    effective_order_qty = params.order_size
    enable_bid, enable_ask = inventory_permissions(
        q,
        max_inventory=params.max_inventory,
        effective_order_qty=effective_order_qty,
        soft_inventory_lots=params.soft_inventory_lots,
    )

    if q >= params.max_inventory:
        size = min(params.order_size, q)
    elif q <= -params.max_inventory:
        size = min(params.order_size, abs(q))
    else:
        size = params.order_size

    cover_tight = False
    joined_bbo = False
    cut_exit = False
    trail_cut = False
    hard_flatten = False
    mark_px = trade_mid if trade_mid > 0 else 0.5 * (trade_bid + trade_ask)
    entry_px = avg_entry if avg_entry > 0 else mark_px
    inv_roi = leveraged_roi(
        q, entry_px, mark_px, params.cover_join_leverage
    )
    hold_s = inventory_open_age_seconds if abs(q) >= 0.05 else 0.0
    take_profit = abs(q) >= 0.05 and inv_roi >= (
        params.cover_join_roi_pct / 100.0
    )
    cut_thresh = params.cover_cut_roi_pct / 100.0
    # Tiny eps so "exactly 1%" still counts as breach (join), not COVER.
    cut_eps = 1e-6
    cut_loss = (
        abs(q) >= 0.05
        and params.cover_cut_roi_pct > 0
        and inv_roi <= -cut_thresh + cut_eps
    )
    # Strictly worse than the cut threshold → trail mid±(cut/lev); at breach → join BBO.
    deep_cut = cut_loss and inv_roi < -cut_thresh - cut_eps
    hold_timeout = (
        abs(q) >= 0.05
        and params.cover_max_hold_seconds > 0
        and hold_s >= params.cover_max_hold_seconds
    )
    # Rail: at ±max inventory always join BBO to flatten (do not sit COVER).
    at_max = abs(q) >= params.max_inventory - 1e-12
    if params.hard_flatten_at_max and at_max:
        hard_flatten = True
        if q > 0:
            enable_bid, enable_ask = False, True
        else:
            enable_bid, enable_ask = True, False
        bid_px, ask_px, joined_bbo = join_bbo_for_reduce(
            bid_px,
            ask_px,
            inventory=q,
            enable_bid=enable_bid,
            enable_ask=enable_ask,
            trade_bid=trade_bid,
            trade_ask=trade_ask,
            tick=params.trade_tick,
        )
        size = min(params.order_size, abs(q))
    elif take_profit or cut_loss or hold_timeout:
        cut_exit = (cut_loss or hold_timeout) and not take_profit
        if q > 0:
            enable_bid, enable_ask = False, True
        elif q < 0:
            enable_bid, enable_ask = True, False
        if deep_cut and not take_profit and not hold_timeout:
            bid_px, ask_px, trail_cut = trail_reduce_roi_away(
                bid_px,
                ask_px,
                inventory=q,
                enable_bid=enable_bid,
                enable_ask=enable_ask,
                trade_bid=trade_bid,
                trade_ask=trade_ask,
                trade_mid=mark_px,
                tick=params.trade_tick,
                roi_pct=(
                    params.cover_trail_roi_pct
                    if params.cover_trail_roi_pct > 0
                    else params.cover_cut_roi_pct
                ),
                leverage=params.cover_join_leverage,
            )
            if trail_cut:
                size = min(params.order_size, abs(q))
            else:
                # Fallback to join if trail would cross.
                bid_px, ask_px, joined_bbo = join_bbo_for_reduce(
                    bid_px,
                    ask_px,
                    inventory=q,
                    enable_bid=enable_bid,
                    enable_ask=enable_ask,
                    trade_bid=trade_bid,
                    trade_ask=trade_ask,
                    tick=params.trade_tick,
                )
                if joined_bbo:
                    size = min(params.order_size, abs(q))
        else:
            bid_px, ask_px, joined_bbo = join_bbo_for_reduce(
                bid_px,
                ask_px,
                inventory=q,
                enable_bid=enable_bid,
                enable_ask=enable_ask,
                trade_bid=trade_bid,
                trade_ask=trade_ask,
                tick=params.trade_tick,
            )
            if joined_bbo:
                size = min(params.order_size, abs(q))
    else:
        reduce_only = (enable_bid != enable_ask) and abs(q) >= 0.05
        if reduce_only:
            bid_px, ask_px, cover_tight = cover_at_reservation(
                bid_px,
                ask_px,
                inventory=q,
                enable_bid=enable_bid,
                enable_ask=enable_ask,
                reservation=r_usdc,
                trade_bid=trade_bid,
                trade_ask=trade_ask,
                tick=params.trade_tick,
            )
            if cover_tight:
                size = min(params.order_size, abs(q)) if abs(q) > 0 else size

    valid = size > 0 and (enable_bid or enable_ask) and 0 < bid_px < ask_px
    if not valid:
        return CrossQuoteDecision(
            quote=None,
            action="REJECT",
            bid_usdc=bid_usdc,
            ask_usdc=ask_usdc,
            fx_mid=fx_mid,
            enable_bid=enable_bid,
            enable_ask=enable_ask,
            size=0.0,
            hard_flatten=hard_flatten,
            inv_roi=inv_roi,
            basis_residual_bps=basis_residual_bps,
        )

    if hard_flatten and enable_ask and not enable_bid:
        action = "FLATTEN_ASK"
    elif hard_flatten and enable_bid and not enable_ask:
        action = "FLATTEN_BID"
    elif trail_cut and enable_ask and not enable_bid:
        action = "TRAIL_ASK"
    elif trail_cut and enable_bid and not enable_ask:
        action = "TRAIL_BID"
    elif cut_exit and joined_bbo and enable_ask and not enable_bid:
        action = "CUT_ASK"
    elif cut_exit and joined_bbo and enable_bid and not enable_ask:
        action = "CUT_BID"
    elif joined_bbo and enable_ask and not enable_bid:
        action = "JOIN_ASK"
    elif joined_bbo and enable_bid and not enable_ask:
        action = "JOIN_BID"
    elif cover_tight and enable_ask and not enable_bid:
        action = "COVER_ASK"
    elif cover_tight and enable_bid and not enable_ask:
        action = "COVER_BID"
    elif enable_ask and not enable_bid:
        action = "REDUCE_ASK"
    elif enable_bid and not enable_ask:
        action = "REDUCE_BID"
    else:
        action = "QUOTE"

    quote = Quote(
        bid=bid_px,
        ask=ask_px,
        reservation_price=r_usdc,
        spread=ask_px - bid_px,
        gamma=as_quote.gamma,
        kappa=as_quote.kappa,
        action=action,
        bid_enabled=enable_bid,
        ask_enabled=enable_ask,
        tau_seconds=as_quote.tau_seconds,
        vol_spread=as_quote.vol_spread,
        kappa_spread=as_quote.kappa_spread,
    )
    return CrossQuoteDecision(
        quote=quote,
        action=action,
        bid_usdc=bid_usdc,
        ask_usdc=ask_usdc,
        fx_mid=fx_mid,
        enable_bid=enable_bid,
        enable_ask=enable_ask,
        size=size,
        joined_bbo=joined_bbo,
        cover_tight=cover_tight,
        cut_exit=cut_exit,
        trail_cut=trail_cut,
        hard_flatten=hard_flatten,
        inv_roi=inv_roi,
        basis_residual_bps=basis_residual_bps,
    )
