"""Simulated execution for quotes posted by the market-making model."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from .market_state import PositionState
from .queue_l25 import size_ahead_at_price
from .quoting_logic import Quote

# through = any trade that crosses quote (optimistic, phantom fills)
# touch   = only if our quote is at the visible best bid/ask
# queue   = L25 size-ahead must clear before we fill (Part 3B)
FILL_MODES = ("through", "touch", "queue")


@dataclass
class Fill:
    """One simulated maker fill."""

    timestamp: datetime
    side: str
    price: float
    quantity: float
    fee: float
    mid_at_fill: float = 0.0
    action: str = ""


@dataclass
class ExecutionSimulator:
    """Match posted quotes against historical aggressive trades."""

    position: PositionState = field(default_factory=PositionState)
    maker_fee_rate: float = 0.0002
    max_inventory: float = 1.0
    order_size: float = 0.0
    fills: list[Fill] = field(default_factory=list)
    active_quote: Quote | None = None
    fill_mode: str = "touch"
    tick_size: float = 0.0001
    # Remaining size ahead of us at the working bid/ask (queue mode).
    queue_ahead_bid: float = 0.0
    queue_ahead_ask: float = 0.0
    _posted_bid: float = 0.0
    _posted_ask: float = 0.0

    def __post_init__(self) -> None:
        if self.maker_fee_rate < 0:
            raise ValueError("maker_fee_rate cannot be negative")
        if self.max_inventory <= 0:
            raise ValueError("max_inventory must be positive")
        if self.order_size < 0:
            raise ValueError("order_size cannot be negative")
        if self.order_size == 0.0:
            self.order_size = max(self.max_inventory * 0.02, 1e-9)
        mode = self.fill_mode.lower().strip()
        if mode not in FILL_MODES:
            raise ValueError(f"fill_mode must be one of {FILL_MODES}")
        self.fill_mode = mode

    def post_quote(
        self,
        quote: Quote,
        *,
        bid_prices: list[float] | None = None,
        bid_amounts: list[float] | None = None,
        ask_prices: list[float] | None = None,
        ask_amounts: list[float] | None = None,
        best_bid_qty: float = 0.0,
        best_ask_qty: float = 0.0,
    ) -> None:
        """Replace the currently active bid and/or ask quote.

        In ``queue`` mode, recomputes size-ahead from L25 when price changes;
        if price is unchanged, keeps the depleted queue remainder.
        """
        if not quote.bid_enabled and not quote.ask_enabled:
            self.active_quote = None
            self.queue_ahead_bid = 0.0
            self.queue_ahead_ask = 0.0
            self._posted_bid = 0.0
            self._posted_ask = 0.0
            return
        if quote.bid_enabled and quote.bid <= 0:
            raise ValueError("enabled bid must be positive")
        if quote.ask_enabled and quote.ask <= 0:
            raise ValueError("enabled ask must be positive")
        if quote.bid_enabled and quote.ask_enabled and quote.ask <= quote.bid:
            raise ValueError("quote must have bid < ask when both sides are on")

        if self.fill_mode == "queue":
            eps = max(self.tick_size * 0.51, 1e-12)
            if quote.bid_enabled:
                if abs(quote.bid - self._posted_bid) > eps or self._posted_bid <= 0:
                    has_l25 = bool(bid_prices) and bool(bid_amounts)
                    ahead = size_ahead_at_price(
                        bid_prices,
                        bid_amounts,
                        quote.bid,
                        side="bid",
                        tick=self.tick_size,
                    )
                    # Only fall back to BBO qty when L25 lists are missing.
                    # Empty level with L25 present ⇒ ahead=0 (we are first).
                    if not has_l25 and ahead <= 0 and best_bid_qty > 0:
                        ahead = float(best_bid_qty)
                    self.queue_ahead_bid = ahead
                # else keep depleted queue_ahead_bid
                self._posted_bid = quote.bid
            else:
                self.queue_ahead_bid = 0.0
                self._posted_bid = 0.0

            if quote.ask_enabled:
                if abs(quote.ask - self._posted_ask) > eps or self._posted_ask <= 0:
                    has_l25 = bool(ask_prices) and bool(ask_amounts)
                    ahead = size_ahead_at_price(
                        ask_prices,
                        ask_amounts,
                        quote.ask,
                        side="ask",
                        tick=self.tick_size,
                    )
                    if not has_l25 and ahead <= 0 and best_ask_qty > 0:
                        ahead = float(best_ask_qty)
                    self.queue_ahead_ask = ahead
                self._posted_ask = quote.ask
            else:
                self.queue_ahead_ask = 0.0
                self._posted_ask = 0.0

        self.active_quote = quote

    def process_trade(
        self,
        timestamp: datetime,
        trade_price: float,
        trade_quantity: float,
        trade_side: str,
        *,
        best_bid: float | None = None,
        best_ask: float | None = None,
        mid: float | None = None,
    ) -> Fill | None:
        """Match one aggressive trade against the active quote.

        ``trade_side`` is the aggressor side: a sell can hit our bid, while
        a buy can hit our ask.
        """
        if trade_price <= 0 or trade_quantity <= 0:
            raise ValueError("trade price and quantity must be positive")
        if trade_side not in {"buy", "sell"}:
            raise ValueError("trade_side must be 'buy' or 'sell'")
        if self.active_quote is None:
            return None

        quote = self.active_quote
        eps = max(self.tick_size * 0.51, 1e-12)
        remaining = float(trade_quantity)

        if (
            trade_side == "sell"
            and quote.bid_enabled
            and trade_price <= quote.bid + eps
        ):
            if self.fill_mode == "touch":
                if best_bid is None or best_bid <= 0:
                    return None
                if abs(quote.bid - best_bid) > eps:
                    return None
            if self.fill_mode == "queue":
                # Trade at/through our bid: eat size ahead first.
                if trade_price < quote.bid - eps:
                    # Swept through our level — ahead is gone.
                    self.queue_ahead_bid = 0.0
                else:
                    take = min(remaining, self.queue_ahead_bid)
                    self.queue_ahead_bid -= take
                    remaining -= take
                if remaining <= 1e-15 or self.queue_ahead_bid > 1e-15:
                    return None
            side = "buy"
            price = quote.bid
        elif (
            trade_side == "buy"
            and quote.ask_enabled
            and trade_price >= quote.ask - eps
        ):
            if self.fill_mode == "touch":
                if best_ask is None or best_ask <= 0:
                    return None
                if abs(quote.ask - best_ask) > eps:
                    return None
            if self.fill_mode == "queue":
                if trade_price > quote.ask + eps:
                    self.queue_ahead_ask = 0.0
                else:
                    take = min(remaining, self.queue_ahead_ask)
                    self.queue_ahead_ask -= take
                    remaining -= take
                if remaining <= 1e-15 or self.queue_ahead_ask > 1e-15:
                    return None
            side = "sell"
            price = quote.ask
        else:
            return None

        inventory = self.position.inventory
        action = (quote.action or "").upper()
        reduce_only = action in {"WAIT_TP", "TAKE_PROFIT", "FLATTEN"} or any(
            tag in action
            for tag in ("COVER_", "JOIN_", "REDUCE_", "CUT_", "FLATTEN_", "TRAIL_")
        )
        if reduce_only:
            if side == "buy" and inventory >= 0:
                return None
            if side == "sell" and inventory <= 0:
                return None
            room = abs(inventory)
        elif side == "buy":
            room = max(0.0, self.max_inventory - inventory)
        else:
            room = max(0.0, self.max_inventory + inventory)

        qty_avail = remaining if self.fill_mode == "queue" else float(trade_quantity)
        quantity = min(qty_avail, self.order_size, room)
        if quantity <= 0:
            return None

        notional = price * quantity
        fee = notional * self.maker_fee_rate
        self._apply_fill(side=side, price=price, quantity=quantity, fee=fee)

        mid_at = mid if mid and mid > 0 else price
        fill = Fill(
            timestamp,
            side,
            price,
            quantity,
            fee,
            mid_at_fill=mid_at,
            action=str(quote.action or ""),
        )
        self.fills.append(fill)
        return fill

    def _apply_fill(
        self,
        *,
        side: str,
        price: float,
        quantity: float,
        fee: float,
    ) -> None:
        """Update cash, inventory, average entry, and realized PnL."""
        pos = self.position
        signed = quantity if side == "buy" else -quantity
        notional = price * quantity

        if side == "buy":
            pos.buy_quantity += quantity
            pos.buy_notional += notional
            pos.cash -= notional + fee
        else:
            pos.sell_quantity += quantity
            pos.sell_notional += notional
            pos.cash += notional - fee

        pos.realized_pnl -= fee

        if pos.inventory > 0 and signed < 0:
            closed = min(pos.inventory, -signed)
            pos.realized_pnl += closed * (price - pos.avg_entry_price)
            remaining = -signed - closed
            pos.inventory -= closed
            if abs(pos.inventory) < 1e-15:
                pos.inventory = 0.0
                pos.avg_entry_price = 0.0
            if remaining > 0:
                pos.avg_entry_price = price
                pos.inventory = -remaining
        elif pos.inventory < 0 and signed > 0:
            closed = min(-pos.inventory, signed)
            pos.realized_pnl += closed * (pos.avg_entry_price - price)
            remaining = signed - closed
            pos.inventory += closed
            if abs(pos.inventory) < 1e-15:
                pos.inventory = 0.0
                pos.avg_entry_price = 0.0
            if remaining > 0:
                pos.avg_entry_price = price
                pos.inventory = remaining
        else:
            old_abs = abs(pos.inventory)
            if old_abs < 1e-15:
                pos.avg_entry_price = price
                pos.inventory = signed
            else:
                pos.avg_entry_price = (
                    pos.avg_entry_price * old_abs + price * quantity
                ) / (old_abs + quantity)
                pos.inventory += signed
