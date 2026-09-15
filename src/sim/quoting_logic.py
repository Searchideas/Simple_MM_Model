"""Avellaneda-Stoikov quote generation.

Purpose:
    Turn a market state and model parameters into bid and ask quotes.

Responsibilities:
    - Estimate or receive short-horizon volatility.
    - Calculate the reservation price from current inventory.
    - Calculate the optimal spread from risk aversion and fill intensity.
    - Respect tick size, minimum spread, and price bounds.

Model context:
    For the basis strategy, the quoted instrument is XAUTUSDT or PAXGUSDT.
    XAUUSDT is the hedge instrument and should be handled by execution or a
    separate hedge component.

Implementation task:
    Implement the AS equations here. This module should not load files,
    simulate fills, or write results.
"""

from dataclasses import dataclass, field
import math
import numpy as np

from .market_state import MarketState, PositionState


@dataclass
class QuoteParameters:
    """Fixed configuration for the AS model."""

    base_gamma: float
    kappa: float
    symmetrical_bid: float
    symmetrical_ask: float
    order_book_liquidity: float
    time_horizon: float
    min_spread: float
    tick_size: float
    returns_1m: list[float] = field(default_factory=list)

@dataclass
class Quote:
    """Quotes produced by the AS model."""

    bid: float
    ask: float
    reservation_price: float
    spread: float

class AvellanedaStoikovModel:
    """Avellaneda-Stoikov quoting model."""

    def __init__(self, params: QuoteParameters):
        self.params = params

    def gamma_calculation(
        self,
        market_state: MarketState,
        position_state: PositionState,
    ) -> float:
        """Calculate gamma from time horizon and recent return imbalance."""
        base_gamma = self.params.base_gamma
        gamma = base_gamma

        returns = self.params.returns_1m[-10:]
        positive = sum(return_value > 0 for return_value in returns)
        negative = sum(return_value < 0 for return_value in returns)
        imbalance = abs(positive - negative) / 10.0

        return gamma * (1.0 + imbalance)

    def kappa_calculation(
        self,
        quote_distances: list[float],
        fills: list[int],
        exposure_seconds: list[float],
    ) -> float:
        """Estimate kappa from historical quote fills.

        The data must describe quote distance from mid-price, number of fills,
        and the time those quotes were exposed at each distance.
        """
        if not (
            len(quote_distances)
            == len(fills)
            == len(exposure_seconds)
        ):
            raise ValueError("historical inputs must have the same length")
        if len(quote_distances) < 2:
            raise ValueError("at least two observations are required")

        distances = np.asarray(quote_distances, dtype=float)
        fill_counts = np.asarray(fills, dtype=float)
        exposure = np.asarray(exposure_seconds, dtype=float)

        if np.any(distances < 0):
            raise ValueError("quote distances must be non-negative")
        if np.any(fill_counts < 0):
            raise ValueError("fill counts must be non-negative")
        if np.any(exposure <= 0):
            raise ValueError("exposure times must be positive")

        fill_rates = fill_counts / exposure
        valid = fill_rates > 0
        if valid.sum() < 2:
            raise ValueError("at least two positive fill rates are required")

        slope, _ = np.polyfit(
            distances[valid],
            np.log(fill_rates[valid]),
            1,
        )
        kappa = -slope

        if kappa <= 0:
            raise ValueError(
                "estimated kappa is not positive; inspect the fill data"
            )

        return float(kappa)

    def calculate_quotes(
        self,
        market_state: MarketState,
        position_state: PositionState,
    ) -> Quote:
        """Calculate inventory-aware bid and ask quotes."""
        mid_price = market_state.mid_price
        if mid_price <= 0:
            raise ValueError("mid-price must be positive")
        if market_state.volatility < 0:
            raise ValueError("volatility cannot be negative")
        if self.params.kappa <= 0:
            raise ValueError("kappa must be positive")
        if self.params.tick_size <= 0:
            raise ValueError("tick_size must be positive")

        gamma = self.gamma_calculation(market_state, position_state)
        time_years = self.params.time_horizon / (365 * 24 * 60 * 60)
        absolute_volatility = mid_price * market_state.volatility
        variance_time = absolute_volatility**2 * time_years

        reservation_price = (
            mid_price
            - position_state.inventory * gamma * variance_time
        )
        spread = (
            2.0 / gamma * math.log(1.0 + gamma / self.params.kappa)
            + 0.5 * gamma * variance_time
        )
        spread = max(spread, self.params.min_spread)

        bid = self._round_down(reservation_price - spread / 2.0)
        ask = self._round_up(reservation_price + spread / 2.0)

        return Quote(
            bid=bid,
            ask=ask,
            reservation_price=reservation_price,
            spread=ask - bid,
        )

    def _round_down(self, price: float) -> float:
        return math.floor(price / self.params.tick_size) * self.params.tick_size

    def _round_up(self, price: float) -> float:
        return math.ceil(price / self.params.tick_size) * self.params.tick_size

    