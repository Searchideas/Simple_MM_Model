"""Simulated execution for quotes posted by the market-making model."""

from dataclasses import dataclass, field
from datetime import datetime

from .market_state import PositionState
from .quoting_logic import Quote


@dataclass
class Fill:
    """One simulated maker fill."""

    timestamp: datetime
    side: str
    price: float
    quantity: float
    fee: float


@dataclass
class ExecutionSimulator:
    """Match posted quotes against historical aggressive trades."""

    position: PositionState = field(default_factory=PositionState)
    maker_fee_rate: float = 0.0002
    max_inventory: float = 1.0
    order_size: float = 0.0
    fills: list[Fill] = field(default_factory=list)
    active_quote: Quote | None = None

    def __post_init__(self) -> None:
        if self.maker_fee_rate < 0:
            raise ValueError("maker_fee_rate cannot be negative")
        if self.max_inventory <= 0:
            raise ValueError("max_inventory must be positive")
        if self.order_size < 0:
            raise ValueError("order_size cannot be negative")
        if self.order_size == 0.0:
            self.order_size = max(self.max_inventory * 0.02, 1e-9)

    def post_quote(self, quote: Quote) -> None:
        """Replace the currently active bid and/or ask quote."""
        if not quote.bid_enabled and not quote.ask_enabled:
            self.active_quote = None
            return
        if quote.bid_enabled and quote.bid <= 0:
            raise ValueError("enabled bid must be positive")
        if quote.ask_enabled and quote.ask <= 0:
            raise ValueError("enabled ask must be positive")
        if quote.bid_enabled and quote.ask_enabled and quote.ask <= quote.bid:
            raise ValueError("quote must have bid < ask when both sides are on")
        self.active_quote = quote

    def process_trade(
        self,
        timestamp: datetime,
        trade_price: float,
        trade_quantity: float,
        trade_side: str,
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
        if (
            trade_side == "sell"
            and quote.bid_enabled
            and trade_price <= quote.bid
        ):
            side = "buy"
            price = quote.bid
        elif (
            trade_side == "buy"
            and quote.ask_enabled
            and trade_price >= quote.ask
        ):
            side = "sell"
            price = quote.ask
        else:
            return None

        inventory = self.position.inventory
        # Reduce-only modes: only fill the flattening side, capped by |q|.
        reduce_only = quote.action in {"WAIT_TP", "TAKE_PROFIT", "FLATTEN"}
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

        quantity = min(trade_quantity, self.order_size, room)
        if quantity <= 0:
            return None

        notional = price * quantity
        fee = notional * self.maker_fee_rate
        self._apply_fill(side=side, price=price, quantity=quantity, fee=fee)

        fill = Fill(timestamp, side, price, quantity, fee)
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

        # Fees always hit realized; inventory MTM is separate.
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
