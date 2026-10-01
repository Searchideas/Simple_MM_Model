"""Event-level queue: L2 book updates + trade-tape partial fills."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .event_book import L2Book, _tick_key
from .execution import Fill
from .market_state import PositionState


@dataclass
class RestingSide:
    enabled: bool = False
    price: float = 0.0
    ahead: float = 0.0
    working_qty: float = 0.0
    # Size ahead at the moment this price was accepted. ``ahead`` shrinks later.
    join_ahead: float = 0.0
    # Features from the quote decision that posted this order.
    context: dict | None = None


class EventL2Queue:
    """Maker queue on a live L2 book with trade partial fills."""

    def __init__(
        self,
        *,
        tick: float,
        order_size: float,
        max_inventory: float,
        maker_fee_rate: float = 0.0,
        position: PositionState | None = None,
    ) -> None:
        self.tick = tick
        self.order_size = order_size
        self.max_inventory = max_inventory
        self.maker_fee_rate = maker_fee_rate
        self.position = position or PositionState()
        self.book = L2Book(tick=tick)
        self.bid = RestingSide()
        self.ask = RestingSide()
        self.fills: list[Fill] = []
        # Parallel to ``fills``: quote features copied at fill time.
        self.fill_context: list[dict | None] = []
        self._prev_snapshot = False

    def on_book_row(
        self,
        *,
        side: str,
        price: float,
        amount: float,
        is_snapshot: bool,
    ) -> None:
        side_l = side.lower().strip()
        if side_l not in ("bid", "ask"):
            self._prev_snapshot = bool(is_snapshot)
            return
        old, new = self.book.apply(
            side=side_l,
            price=price,
            amount=amount,
            is_snapshot=bool(is_snapshot),
            prev_was_snapshot=self._prev_snapshot,
        )
        self._prev_snapshot = bool(is_snapshot)
        decrease = old - new
        if decrease > 1e-15:
            self._burn_ahead_on_cancel(side_l, price, decrease)

    def _burn_ahead_on_cancel(self, side: str, price: float, decrease: float) -> None:
        resting = self.bid if side == "bid" else self.ask
        if not resting.enabled or resting.ahead <= 0:
            return
        key = _tick_key(price, self.tick)
        our = _tick_key(resting.price, self.tick)
        if side == "bid":
            affects = key >= our
        else:
            affects = key <= our
        if affects:
            resting.ahead = max(0.0, resting.ahead - decrease)

    def post(
        self,
        *,
        bid: float,
        ask: float,
        bid_enabled: bool,
        ask_enabled: bool,
        size: float | None = None,
    ) -> None:
        qty = float(size if size is not None else self.order_size)
        eps = max(self.tick * 0.51, 1e-12)

        if bid_enabled and bid > 0:
            if abs(bid - self.bid.price) > eps or not self.bid.enabled:
                self.bid.price = bid
                self.bid.ahead = self.book.size_ahead(side="bid", our_price=bid)
                self.bid.join_ahead = self.bid.ahead
                self.bid.working_qty = qty
                self.bid.context = None
            self.bid.enabled = True
        else:
            self.bid = RestingSide()

        if ask_enabled and ask > 0:
            if abs(ask - self.ask.price) > eps or not self.ask.enabled:
                self.ask.price = ask
                self.ask.ahead = self.book.size_ahead(side="ask", our_price=ask)
                self.ask.join_ahead = self.ask.ahead
                self.ask.working_qty = qty
                self.ask.context = None
            self.ask.enabled = True
        else:
            self.ask = RestingSide()

    def on_trade(
        self,
        *,
        timestamp: datetime,
        price: float,
        amount: float,
        aggressor_side: str,
        mid: float = 0.0,
        action: str = "QUOTE",
    ) -> list[Fill]:
        if amount <= 0 or price <= 0:
            return []
        side = aggressor_side.lower().strip()
        left = float(amount)
        out: list[Fill] = []
        eps = max(self.tick * 0.51, 1e-12)

        if side == "sell" and self.bid.enabled and self.bid.working_qty > 0:
            if price <= self.bid.price + eps:
                fill = self._match_resting(
                    resting=self.bid,
                    our_side="buy",
                    trade_left=left,
                    timestamp=timestamp,
                    mid=mid,
                    action=action,
                )
                if fill is not None:
                    out.append(fill)
                    left -= fill.quantity

        if side == "buy" and self.ask.enabled and self.ask.working_qty > 0:
            if price + eps >= self.ask.price:
                fill = self._match_resting(
                    resting=self.ask,
                    our_side="sell",
                    trade_left=left,
                    timestamp=timestamp,
                    mid=mid,
                    action=action,
                )
                if fill is not None:
                    out.append(fill)
        return out

    def _match_resting(
        self,
        *,
        resting: RestingSide,
        our_side: str,
        trade_left: float,
        timestamp: datetime,
        mid: float,
        action: str,
    ) -> Fill | None:
        if resting.ahead > 1e-15:
            take = min(trade_left, resting.ahead)
            resting.ahead -= take
            trade_left -= take
        if trade_left <= 1e-15 or resting.ahead > 1e-15:
            return None
        if resting.working_qty <= 1e-15:
            return None

        inv = self.position.inventory
        if our_side == "buy":
            room = max(0.0, self.max_inventory - inv)
        else:
            room = max(0.0, self.max_inventory + inv)
        qty = min(trade_left, resting.working_qty, room)
        if qty <= 1e-15:
            return None

        fill_px = resting.price
        fee = fill_px * qty * self.maker_fee_rate
        self._apply_fill(our_side, fill_px, qty, fee)
        resting.working_qty -= qty
        if resting.working_qty <= 1e-15:
            resting.working_qty = 0.0

        mid_at = mid if mid > 0 else self.book.mid() or fill_px
        fill = Fill(
            timestamp,
            our_side,
            fill_px,
            qty,
            fee,
            mid_at_fill=mid_at,
            action=action,
        )
        self.fills.append(fill)
        ctx = dict(resting.context) if resting.context else None
        self.fill_context.append(ctx)
        return fill

    def _apply_fill(
        self, side: str, price: float, quantity: float, fee: float
    ) -> None:
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
