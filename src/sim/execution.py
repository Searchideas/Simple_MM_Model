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
    fills: list[Fill] = field(default_factory=list)
    active_quote: Quote | None = None

    def __post_init__(self) -> None:
        if self.maker_fee_rate < 0:
            raise ValueError("maker_fee_rate cannot be negative")
        if self.max_inventory <= 0:
            raise ValueError("max_inventory must be positive")

    def post_quote(self, quote: Quote) -> None:
        """Replace the currently active bid and ask quote."""
        if quote.bid <= 0 or quote.ask <= quote.bid:
            raise ValueError("quote must have 0 < bid < ask")
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

        if trade_side == "sell" and trade_price <= self.active_quote.bid:
            side = "buy"
            price = self.active_quote.bid
        elif trade_side == "buy" and trade_price >= self.active_quote.ask:
            side = "sell"
            price = self.active_quote.ask
        else:
            return None

        if side == "buy":
            quantity = min(
                trade_quantity,
                max(0.0, self.max_inventory - self.position.inventory),
            )
        else:
            quantity = min(
                trade_quantity,
                max(0.0, self.max_inventory + self.position.inventory),
            )

        if quantity == 0:
            return None

        notional = price * quantity
        fee = notional * self.maker_fee_rate

        if side == "buy":
            self.position.inventory += quantity
            self.position.buy_quantity += quantity
            self.position.buy_notional += notional
            self.position.cash -= notional + fee
        else:
            self.position.inventory -= quantity
            self.position.sell_quantity += quantity
            self.position.sell_notional += notional
            self.position.cash += notional - fee

        fill = Fill(timestamp, side, price, quantity, fee)
        self.fills.append(fill)
        return fill
