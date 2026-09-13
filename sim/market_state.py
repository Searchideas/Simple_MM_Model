"""State objects passed between the market-making backtest components."""

from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

@dataclass
class MarketState:
    """Market data available at one timestamp.

    Attributes:
        timestamp: The time of the market snapshot.
        best_bid: The best bid price in the market.
        best_ask: The best ask price in the market.
        volatility: An estimate of short-term volatility.
        fair_value: Optional fair value or basis information.
    """
    timestamp: datetime
    best_bid: float
    best_bid_volume: float
    best_ask_volume: float
    best_ask: float
    volatility: float
    fair_value: float | None = None
    bid_prices: list[float] = field(default_factory=list)
    bid_volumes: list[float] = field(default_factory=list)
    ask_prices: list[float] = field(default_factory=list)
    ask_volumes: list[float] = field(default_factory=list)

    @property
    def mid_price(self) -> float:
        """Calculate the mid-price from the best bid and ask."""
        return (self.best_bid + self.best_ask) / 2

    def weighted_mid_price(self) -> float:
        """Calculate the volume-weighted mid-price."""
        total_volume = self.best_bid_volume + self.best_ask_volume
        if total_volume == 0:
            return self.mid_price  # Avoid division by zero
        return (
            self.best_bid * self.best_bid_volume
            + self.best_ask * self.best_ask_volume
        ) / total_volume

    @staticmethod
    def calculate_volatility(
        midprices: list[float], interval_seconds: int
    ) -> float:
        """Calculate annualized volatility from equally spaced mid-prices."""
        if len(midprices) < 3:
            return 0.0
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")

        prices = np.asarray(midprices, dtype=float)
        if np.any(prices <= 0):
            raise ValueError("midprices must be positive")

        log_returns = np.diff(np.log(prices))
        periods_per_year = 365 * 24 * 60 * 60 / interval_seconds
        return float(np.std(log_returns, ddof=1) * np.sqrt(periods_per_year))


@dataclass
class PositionState:
    """Trading account state maintained by the execution simulator."""
    inventory: float = 0.0
    buy_notional: float = 0.0
    sell_notional: float = 0.0
    buy_quantity: float = 0.0
    sell_quantity: float = 0.0
    cash: float = 0.0

    @property
    def vwap_buy_price(self) -> float:
        """Calculate the volume-weighted average price for buys."""
        if self.buy_quantity == 0:
            return 0.0
        return self.buy_notional / self.buy_quantity

    @property
    def vwap_sell_price(self) -> float:
        """Calculate the volume-weighted average price for sells."""
        if self.sell_quantity == 0:
            return 0.0
        return self.sell_notional / self.sell_quantity