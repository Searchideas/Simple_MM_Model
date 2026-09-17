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
import warnings

import numpy as np

try:
    from numpy.exceptions import RankWarning
except ImportError:  # numpy < 2
    RankWarning = np.RankWarning

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
    max_inventory: float = 1.0
    max_skew_ticks: float = 10.0
    max_spread_ticks: float = 10.0
    book_skew_ticks: float = 3.0
    book_skew_size_weight: float = 0.75
    book_levels: int = 5
    maker_fee_rate: float = 0.0002
    spread_in_ticks: bool = True
    ma_fast: int = 7
    ma_slow: int = 25
    ma_skew_ticks: float = 6.0
    ma_signal_ticks: float = 10.0
    toxic_obi: float = 0.35
    ma_fade: float = 0.4
    flatten_frac: float = 0.5
    returns_1m: list[float] = field(default_factory=list)
    closes_1m: list[float] = field(default_factory=list)

@dataclass
class Quote:
    """Quotes produced by the AS model."""

    bid: float
    ask: float
    reservation_price: float
    spread: float
    gamma: float = 0.0
    kappa: float = 0.0
    imbalance: float = 0.0
    book_imbalance: float = 0.0
    vwap_bid: float = 0.0
    vwap_ask: float = 0.0
    book_shift: float = 0.0
    book_vwap_shift: float = 0.0
    book_size_shift: float = 0.0
    ma_shift: float = 0.0
    ma_frac: float = 0.0
    ma_fast: float = 0.0
    ma_slow: float = 0.0
    action: str = "QUOTE"
    bid_enabled: bool = True
    ask_enabled: bool = True


class AvellanedaStoikovModel:
    """Avellaneda-Stoikov quoting model."""

    def __init__(self, params: QuoteParameters):
        self.params = params
        self.estimated_kappa: float | None = None
        self.last_imbalance: float = 0.0
        self.last_book_imbalance: float = 0.0
        self.last_ma_frac: float = 0.0

    @staticmethod
    def queue_vwap(
        prices: list[float],
        volumes: list[float],
        levels: int = 5,
    ) -> float | None:
        """Volume-weighted average price over the top-N book levels."""
        n = max(1, levels)
        numerator = 0.0
        denominator = 0.0
        for price, volume in zip(prices[:n], volumes[:n]):
            if price > 0 and volume > 0:
                numerator += float(price) * float(volume)
                denominator += float(volume)
        if denominator <= 0:
            return None
        return numerator / denominator

    @staticmethod
    def book_imbalance(
        bid_volumes: list[float],
        ask_volumes: list[float],
        levels: int = 5,
    ) -> float:
        """Top-N size imbalance in [-1, 1]: (Vb - Va) / (Vb + Va)."""
        n = max(1, levels)
        vb = float(sum(bid_volumes[:n]))
        va = float(sum(ask_volumes[:n]))
        total = vb + va
        if total <= 0:
            return 0.0
        return (vb - va) / total

    def book_skew_shift(
        self,
        mid_price: float,
        bid_prices: list[float],
        bid_volumes: list[float],
        ask_prices: list[float],
        ask_volumes: list[float],
    ) -> tuple[float, float, float, float, float, float]:
        """Book skew from blended VWAP microprice + size imbalance.

        Returns ``(shift, I, vwap_bid, vwap_ask, vwap_raw, size_raw)``.

        Live blend: ``shift = clip((1-w)*vwap_raw + w*size_raw, ±max_shift)``
        where ``w = book_skew_size_weight`` (default 0.75 → mostly size I).
        """
        levels = self.params.book_levels
        tick = self.params.tick_size
        max_shift = abs(self.params.book_skew_ticks) * tick
        weight = max(0.0, min(1.0, self.params.book_skew_size_weight))

        vwap_bid = self.queue_vwap(bid_prices, bid_volumes, levels)
        vwap_ask = self.queue_vwap(ask_prices, ask_volumes, levels)
        vb = float(sum(bid_volumes[: max(1, levels)]))
        va = float(sum(ask_volumes[: max(1, levels)]))
        size_imb = self.book_imbalance(bid_volumes, ask_volumes, levels)

        if max_shift <= 0 or vb + va <= 0:
            return 0.0, size_imb, vwap_bid or 0.0, vwap_ask or 0.0, 0.0, 0.0

        size_raw = size_imb * max_shift
        vwap_raw = 0.0
        if (
            vwap_bid is not None
            and vwap_ask is not None
            and vwap_bid > 0
            and vwap_ask > 0
        ):
            r_book = (vwap_bid * va + vwap_ask * vb) / (vb + va)
            d_bid = mid_price - vwap_bid
            d_ask = vwap_ask - mid_price
            vwap_raw = 0.5 * ((r_book - mid_price) + (d_ask - d_bid))

        blended = (1.0 - weight) * vwap_raw + weight * size_raw
        shift = max(-max_shift, min(max_shift, blended))
        return shift, size_imb, vwap_bid or 0.0, vwap_ask or 0.0, vwap_raw, size_raw

    def ma_skew_shift(self) -> tuple[float, float, float, float]:
        """MA7−MA25 momentum skew from 1-minute closes.

        Returns ``(ma_shift, ma_frac, ma_fast, ma_slow)``.
        Needs ``ma_slow`` closes (~25 min) before producing a non-zero shift.
        """
        tick = self.params.tick_size
        fast_n = max(1, int(self.params.ma_fast))
        slow_n = max(fast_n, int(self.params.ma_slow))
        closes = self.params.closes_1m
        if len(closes) < slow_n or tick <= 0 or self.params.ma_skew_ticks <= 0:
            return 0.0, 0.0, 0.0, 0.0

        ma_fast = sum(closes[-fast_n:]) / fast_n
        ma_slow = sum(closes[-slow_n:]) / slow_n
        signal = max(self.params.ma_signal_ticks, 1e-12)
        ma_frac = max(-1.0, min(1.0, (ma_fast - ma_slow) / (signal * tick)))
        ma_shift = ma_frac * self.params.ma_skew_ticks * tick
        self.last_ma_frac = ma_frac
        return ma_shift, ma_frac, ma_fast, ma_slow

    def gamma_calculation(
        self,
        market_state: MarketState,
        position_state: PositionState,
    ) -> float:
        """Calculate gamma from time horizon and recent return imbalance."""
        returns = self.params.returns_1m[-10:]
        positive = sum(return_value > 0 for return_value in returns)
        negative = sum(return_value < 0 for return_value in returns)
        imbalance = abs(positive - negative) / 10.0
        self.last_imbalance = imbalance
        return self.params.base_gamma * (1.0 + imbalance)

    def active_kappa(self) -> float:
        """Live kappa when a fill-based estimate exists, otherwise the CLI default."""
        if self.estimated_kappa is not None:
            return self.estimated_kappa
        return self.params.kappa

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

        distances_valid = distances[valid]
        if float(np.max(distances_valid) - np.min(distances_valid)) < 1e-12:
            raise ValueError("quote distances are constant; cannot estimate kappa")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Polyfit may be poorly conditioned")
            warnings.simplefilter("ignore", RankWarning)
            try:
                slope, _ = np.polyfit(
                    distances_valid,
                    np.log(fill_rates[valid]),
                    1,
                )
            except (np.linalg.LinAlgError, ValueError) as exc:
                raise ValueError("kappa fit is poorly conditioned") from exc
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
        if self.active_kappa() <= 0:
            raise ValueError("kappa must be positive")
        if self.params.tick_size <= 0:
            raise ValueError("tick_size must be positive")

        gamma = self.gamma_calculation(market_state, position_state)
        kappa = self.active_kappa()
        tick = self.params.tick_size
        time_years = self.params.time_horizon / (365 * 24 * 60 * 60)
        absolute_volatility = mid_price * market_state.volatility
        variance_time = absolute_volatility**2 * time_years

        as_term = 2.0 / gamma * math.log(1.0 + gamma / kappa)
        vol_term = 0.5 * gamma * variance_time
        if self.params.spread_in_ticks:
            # as_term is in tick units; vol_term is absolute price → ticks.
            spread = (as_term + vol_term / tick) * tick
        else:
            spread = as_term + vol_term

        fee_floor = 2.0 * self.params.maker_fee_rate * mid_price + tick
        min_spread = max(self.params.min_spread, fee_floor)
        max_spread = max(min_spread, abs(self.params.max_spread_ticks) * tick)
        # Vol/AS still widen quotes, but never beyond max_spread_ticks (fillable).
        spread = max(min_spread, min(spread, max_spread))

        max_inventory = max(self.params.max_inventory, 1e-12)
        inventory = position_state.inventory
        inv_frac = max(-1.0, min(1.0, inventory / max_inventory))
        book_shift, book_imb, vwap_bid, vwap_ask, vwap_raw, size_raw = (
            self.book_skew_shift(
                mid_price,
                market_state.bid_prices,
                market_state.bid_volumes,
                market_state.ask_prices,
                market_state.ask_volumes,
            )
        )
        self.last_book_imbalance = book_imb
        ma_shift, ma_frac, ma_fast, ma_slow = self.ma_skew_shift()
        reservation_price = (
            mid_price
            - inv_frac * self.params.max_skew_ticks * tick
            + book_shift
            + ma_shift
        )

        bid = self._round_down(reservation_price - spread / 2.0)
        ask = self._round_up(reservation_price + spread / 2.0)
        bid = min(bid, market_state.best_bid)
        ask = max(ask, market_state.best_ask)

        bid, ask, action, bid_on, ask_on = self._apply_execution_mode(
            market_state=market_state,
            position_state=position_state,
            bid=bid,
            ask=ask,
            book_imb=book_imb,
            ma_frac=ma_frac,
        )

        return Quote(
            bid=bid,
            ask=ask,
            reservation_price=reservation_price,
            spread=max(0.0, ask - bid) if bid_on and ask_on else spread,
            gamma=gamma,
            kappa=kappa,
            imbalance=self.last_imbalance,
            book_imbalance=book_imb,
            vwap_bid=vwap_bid,
            vwap_ask=vwap_ask,
            book_shift=book_shift,
            book_vwap_shift=vwap_raw,
            book_size_shift=size_raw,
            ma_shift=ma_shift,
            ma_frac=ma_frac,
            ma_fast=ma_fast,
            ma_slow=ma_slow,
            action=action,
            bid_enabled=bid_on,
            ask_enabled=ask_on,
        )

    def _apply_execution_mode(
        self,
        *,
        market_state: MarketState,
        position_state: PositionState,
        bid: float,
        ask: float,
        book_imb: float,
        ma_frac: float,
    ) -> tuple[float, float, str, bool, bool]:
        """Live-style side selection: QUOTE / WAIT_TP / TAKE_PROFIT / FLATTEN."""
        inventory = position_state.inventory
        max_inventory = max(self.params.max_inventory, 1e-12)
        fee = self.params.maker_fee_rate
        entry = position_state.avg_entry_price
        mid = market_state.mid_price
        flat_eps = max(1e-12, max_inventory * 1e-9)

        if abs(inventory) <= flat_eps:
            bid_on, ask_on = True, True
            # Toxic OBI: heavy bid size → drop bid; heavy ask size → drop ask.
            if book_imb >= self.params.toxic_obi:
                bid_on = False
            elif book_imb <= -self.params.toxic_obi:
                ask_on = False
            # Fade MA: uptrend → drop ask; downtrend → drop bid.
            if ma_frac >= self.params.ma_fade:
                ask_on = False
            elif ma_frac <= -self.params.ma_fade:
                bid_on = False
            if not bid_on and not ask_on:
                # Keep the less toxic / trend-aligned side.
                if abs(book_imb) >= abs(ma_frac):
                    bid_on, ask_on = book_imb < 0, book_imb > 0
                else:
                    bid_on, ask_on = ma_frac < 0, ma_frac > 0
            return bid, ask, "QUOTE", bid_on, ask_on

        long = inventory > 0
        green = (
            entry > 0
            and (
                (long and mid >= entry * (1.0 + fee))
                or ((not long) and mid <= entry * (1.0 - fee))
            )
        )
        forced = abs(inventory) >= self.params.flatten_frac * max_inventory

        if forced:
            action = "FLATTEN"
            join = True
        elif green:
            action = "TAKE_PROFIT"
            join = True
        else:
            action = "WAIT_TP"
            join = False

        if long:
            bid_on, ask_on = False, True
            if join:
                ask = market_state.best_ask
            elif entry > 0:
                ask = self._round_up(entry * (1.0 + fee))
                # Patient: stay at entry+fee even if behind best (not join touch).
            return bid, ask, action, bid_on, ask_on

        bid_on, ask_on = True, False
        if join:
            bid = market_state.best_bid
        elif entry > 0:
            bid = self._round_down(entry * (1.0 - fee))
        return bid, ask, action, bid_on, ask_on

    def _round_down(self, price: float) -> float:
        return math.floor(price / self.params.tick_size) * self.params.tick_size

    def _round_up(self, price: float) -> float:
        return math.ceil(price / self.params.tick_size) * self.params.tick_size

    