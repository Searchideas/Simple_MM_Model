"""Avellaneda-Stoikov quote generation.

Classic AS with T-t = seconds to next funding; κ from fill_probability.
σ² has units (Price)²/second so σ²·(T-t) is in (Price)².
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math

from .market_state import MarketState, PositionState


FUNDING_INTERVAL_HOURS = 8


def seconds_to_next_funding(timestamp: datetime) -> float:
    """Seconds until the next Binance-style 8h funding (00/08/16 UTC)."""
    if timestamp.tzinfo is None:
        ts = timestamp.replace(tzinfo=timezone.utc)
    else:
        ts = timestamp.astimezone(timezone.utc)
    hour_block = (ts.hour // FUNDING_INTERVAL_HOURS) * FUNDING_INTERVAL_HOURS
    boundary = ts.replace(hour=hour_block, minute=0, second=0, microsecond=0)
    next_funding = boundary + timedelta(hours=FUNDING_INTERVAL_HOURS)
    return max((next_funding - ts).total_seconds(), 0.0)


def clamp_kappa_ticks(
    kappa: float,
    kappa_min: float = 0.05,
    kappa_max: float = 0.5,
) -> float:
    """Clamp fitted κ_ticks away from broken near-zero / spike values."""
    k = float(kappa)
    lo = float(kappa_min)
    hi = float(kappa_max)
    if hi < lo:
        lo, hi = hi, lo
    if k <= 0:
        return lo
    return min(hi, max(lo, k))


@dataclass
class QuoteParameters:
    """Fixed configuration for the AS model."""

    base_gamma: float
    kappa: float  # κ_ticks (1/tick); converted to κ_$ inside calculate_quotes
    time_horizon: float
    min_spread: float
    tick_size: float
    max_inventory: float = 1.0
    max_spread_ticks: float = 10.0
    maker_fee_rate: float = 0.0002
    include_kappa_spread: bool = True
    kappa_min: float = 0.05
    kappa_max: float = 0.5
    # Pull the toxic side when touch imbalance is this extreme.
    # Positive OBI means the next mid tends to rise, so the ask is toxic.
    obi_pull_level: float = 0.8


@dataclass
class Quote:
    """Quotes produced by the AS model."""

    bid: float
    ask: float
    reservation_price: float
    spread: float
    gamma: float = 0.0
    kappa: float = 0.0
    action: str = "QUOTE"
    bid_enabled: bool = True
    ask_enabled: bool = True
    tau_seconds: float = 0.0
    vol_spread: float = 0.0
    kappa_spread: float = 0.0
    obi: float = 0.0


class AvellanedaStoikovModel:
    """Avellaneda-Stoikov quoting model."""

    def __init__(self, params: QuoteParameters):
        self.params = params
        self.estimated_kappa: float | None = None

    def active_kappa(self) -> float:
        """Live κ_ticks from fill_probability path when set, else CLI default."""
        raw = (
            self.estimated_kappa
            if self.estimated_kappa is not None
            else self.params.kappa
        )
        return clamp_kappa_ticks(
            float(raw),
            self.params.kappa_min,
            self.params.kappa_max,
        )

    def calculate_quotes(
        self,
        market_state: MarketState,
        position_state: PositionState,
        *,
        enforce_maker: bool = True,
    ) -> Quote:
        """Classic AS quotes with T-t = seconds to next funding.

        When ``enforce_maker`` is False (cross-quote live path), raw AS prices
        are returned in the reference currency; caller clamps on the trade book
        after FX conversion.
        """
        mid_price = market_state.mid_price
        if mid_price <= 0:
            raise ValueError("mid-price must be positive")
        if market_state.volatility < 0:
            raise ValueError("volatility cannot be negative")
        if self.active_kappa() <= 0:
            raise ValueError("kappa must be positive")
        if self.params.tick_size <= 0:
            raise ValueError("tick_size must be positive")

        gamma = self.params.base_gamma
        tick = self.params.tick_size
        inventory = position_state.inventory

        # Calibration stores κ_ticks (1/tick). AS δ is in price dollars:
        #   κ_$ = κ_ticks / tick    so  κ_$ · δ_$  is dimensionless
        kappa_ticks = self.active_kappa()
        kappa_price = kappa_ticks / tick

        # T-t in seconds; σ_price in Price/sqrt(s) → σ²(T-t) in Price²
        tau_seconds = seconds_to_next_funding(market_state.timestamp)
        sigma_price = mid_price * market_state.volatility
        variance_tau = (sigma_price**2) * tau_seconds

        # r = mid - q * γ * σ_price² * (T-t)
        reservation_price = mid_price - inventory * gamma * variance_tau

        # δ = γ σ_price² (T-t) + (2/γ) ln(1+γ/κ_$)   [price units]
        vol_spread = gamma * variance_tau
        kappa_spread = 0.0
        if self.params.include_kappa_spread:
            kappa_spread = (2.0 / gamma) * math.log(1.0 + gamma / kappa_price)
        spread = vol_spread + kappa_spread

        fee_floor = 2.0 * self.params.maker_fee_rate * mid_price + tick
        min_spread = max(self.params.min_spread, fee_floor)
        max_spread = max(min_spread, abs(self.params.max_spread_ticks) * tick)
        spread = max(min_spread, min(spread, max_spread))

        bid = self._round_down(reservation_price - spread / 2.0)
        ask = self._round_up(reservation_price + spread / 2.0)
        if enforce_maker:
            # Direct-symbol quoting: may improve inside the spread, never cross.
            bid = min(bid, market_state.best_ask - tick)
            ask = max(ask, market_state.best_bid + tick)
        if bid >= ask:
            bid = self._round_down(market_state.best_bid)
            ask = self._round_up(market_state.best_ask)
            if bid >= ask:
                ask = bid + tick

        obi = _touch_obi(market_state.best_bid_volume, market_state.best_ask_volume)
        bid_enabled = True
        ask_enabled = True
        action = "QUOTE"
        pull = self.params.obi_pull_level
        if pull > 0 and obi >= pull:
            ask_enabled = False
            action = "PULL_ASK"
        elif pull > 0 and obi <= -pull:
            bid_enabled = False
            action = "PULL_BID"

        return Quote(
            bid=bid,
            ask=ask,
            reservation_price=reservation_price,
            spread=ask - bid,
            gamma=gamma,
            kappa=kappa_ticks,
            action=action,
            bid_enabled=bid_enabled,
            ask_enabled=ask_enabled,
            tau_seconds=tau_seconds,
            vol_spread=vol_spread,
            kappa_spread=kappa_spread,
            obi=obi,
        )

    def _round_down(self, price: float) -> float:
        return math.floor(price / self.params.tick_size) * self.params.tick_size

    def _round_up(self, price: float) -> float:
        return math.ceil(price / self.params.tick_size) * self.params.tick_size


def _touch_obi(bid_volume: float, ask_volume: float) -> float:
    """Level-1 imbalance. Positive when the bid is larger than the ask."""
    total = bid_volume + ask_volume
    if total <= 0:
        return 0.0
    return (bid_volume - ask_volume) / total
