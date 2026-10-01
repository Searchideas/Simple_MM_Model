"""MEXC-fair → USDCUSDT → SUIUSDC maker (N ticks off) → immediate MEXC B2B hedge.

Fair bid/ask (USDC) = MEXC SUIUSDT bid/ask ÷ USDCUSDT mid.
Post maker on Binance SUIUSDC: bid = fair_bid − N·tick, ask = fair_ask + N·tick,
clamped so we never take the SUIUSDC book. Fills from trade tape (queue/touch).
Every MM fill → opposite taker on MEXC at BBO (always hedge).
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from .execution import ExecutionSimulator
from .markout import MarkoutReport, MarkoutTracker
from .metrics import CrossQuoteMetrics, build_metrics
from .mexc_hedge import MexcTakerHedge
from .quoting_logic import Quote


def _floor_tick(price: float, tick: float) -> float:
    if tick <= 0:
        return price
    return math.floor(price / tick + 1e-12) * tick


def _ceil_tick(price: float, tick: float) -> float:
    if tick <= 0:
        return price
    return math.ceil(price / tick - 1e-12) * tick


@dataclass
class MexcFairResult:
    fills: int
    final_inventory: float
    bn_marked_pnl: float
    hedge_fills: int
    hedge_inventory: float
    hedge_cash_usdt: float
    hedge_marked_usdc: float
    combined_marked_pnl: float
    net_sui: float
    fill_mode: str
    maker_buffer_ticks: float
    reject_cross: int = 0
    fx_stale: int = 0
    markout: MarkoutReport = field(default_factory=MarkoutReport)
    metrics: CrossQuoteMetrics = field(default_factory=CrossQuoteMetrics)
    daily_pnl: list[tuple[str, float, float]] = field(default_factory=list)
    equity_curve: list[tuple[datetime, float]] = field(default_factory=list)
    action_steps: Counter[str] = field(default_factory=Counter)
    last_fair_bid: float = 0.0
    last_fair_ask: float = 0.0
    last_quote_bid: float = 0.0
    last_quote_ask: float = 0.0


@dataclass
class MexcFairParams:
    trade_tick: float = 0.0001
    maker_buffer_ticks: float = 5.0
    order_size: float = 10.0
    max_inventory: float = 50.0
    soft_inventory_lots: float = 0.0
    # Extra passive ticks on each side (desk "bid/ask skew").
    bid_skew_ticks: float = 0.0
    ask_skew_ticks: float = 0.0
    # When long: widen bid; when short: widen ask (per |inv|/order_size lot).
    inv_skew_ticks: float = 0.0
    # Pause quoting if |Δ mid| over lookback exceeds this many ticks.
    fast_move_ticks: float = 8.0
    fast_move_lookback_seconds: float = 2.0
    fast_move_pause_seconds: float = 3.0
    # Pull a side when short-term flow is against it.
    # off | micro | ofi | both | either | bid | ask
    flow_filter: str = "either"


class MexcFairHedgeBacktest:
    """Quote SUIUSDC from MEXC fair; B2B hedge every fill on MEXC."""

    def __init__(
        self,
        execution: ExecutionSimulator,
        hedge: MexcTakerHedge,
        params: MexcFairParams,
        markout_horizon_seconds: float = 1.0,
    ) -> None:
        self.execution = execution
        self.hedge = hedge
        # Always B2B — never skip for "unfavorable".
        self.hedge.favorable_only = False
        self.params = params
        self.markout_horizon_seconds = markout_horizon_seconds

    def _build_quote(
        self,
        *,
        mx_bid: float,
        mx_ask: float,
        fx_mid: float,
        trade_bid: float,
        trade_ask: float,
        inventory: float,
    ) -> Quote | None:
        if not (0 < mx_bid < mx_ask) or fx_mid <= 0:
            return None
        if not (0 < trade_bid < trade_ask):
            return None

        tick = self.params.trade_tick
        buf = self.params.maker_buffer_ticks * tick
        fair_bid = mx_bid / fx_mid
        fair_ask = mx_ask / fx_mid
        # N ticks off MEXC fair (more passive), then maker-clamp to SUIUSDC book.
        raw_bid = _floor_tick(fair_bid - buf, tick)
        raw_ask = _ceil_tick(fair_ask + buf, tick)
        bid = min(raw_bid, _floor_tick(trade_bid, tick))
        ask = max(raw_ask, _ceil_tick(trade_ask, tick))
        if bid <= 0 or ask <= 0 or ask <= bid:
            return None

        soft = max(0.0, self.params.soft_inventory_lots) * self.params.order_size
        enable_bid = (
            inventory <= soft
            and inventory + self.params.order_size <= self.params.max_inventory
        )
        enable_ask = (
            inventory >= -soft
            and inventory - self.params.order_size >= -self.params.max_inventory
        )
        if inventory >= self.params.max_inventory - 1e-12:
            enable_bid, enable_ask = False, True
        elif inventory <= -self.params.max_inventory + 1e-12:
            enable_bid, enable_ask = True, False

        if not enable_bid and not enable_ask:
            return None

        mid = 0.5 * (fair_bid + fair_ask)
        return Quote(
            bid=bid,
            ask=ask,
            reservation_price=mid,
            spread=ask - bid,
            action="QUOTE",
            bid_enabled=enable_bid,
            ask_enabled=enable_ask,
        )

    def run_rows(
        self,
        timeline_rows: list[dict],
        trade_rows: list[dict],
        on_update: Callable | None = None,
    ) -> MexcFairResult:
        trade_index = 0
        markout = MarkoutTracker(
            tick=self.params.trade_tick,
            horizon_seconds=self.markout_horizon_seconds,
        )
        reject_cross = 0
        fx_stale = 0
        action_steps: Counter[str] = Counter()
        equity_curve: list[tuple[datetime, float]] = []
        daily_pnl: list[tuple[str, float, float]] = []
        day_start_pnl: float | None = None
        current_day: str | None = None
        last_equity_ts: datetime | None = None
        max_abs_inventory = 0.0
        last_fair_bid = 0.0
        last_fair_ask = 0.0
        last_quote_bid = 0.0
        last_quote_ask = 0.0
        last_trade_mid = 0.0
        last_fx_mid = 0.0
        last_hedge_mid = 0.0
        sample_seconds = 60.0

        for index, row in enumerate(timeline_rows):
            ts: datetime = row["ts"]
            # ref_* = MEXC SUIUSDT (fair source).
            mx_bid = float(row["ref_bid"])
            mx_ask = float(row["ref_ask"])
            mx_mid = float(row.get("hedge_mid") or row["ref_mid"])
            trade_bid = float(row["trade_bid"])
            trade_ask = float(row["trade_ask"])
            trade_mid = float(row["trade_mid"])
            fx_mid = float(row["fx_mid"])
            last_trade_mid = trade_mid
            last_fx_mid = fx_mid
            last_hedge_mid = mx_mid

            if fx_mid <= 0:
                fx_stale += 1
                self.execution.active_quote = None
                continue

            markout.on_mid(trade_mid, ts)
            inv = self.execution.position.inventory
            quote = self._build_quote(
                mx_bid=mx_bid,
                mx_ask=mx_ask,
                fx_mid=fx_mid,
                trade_bid=trade_bid,
                trade_ask=trade_ask,
                inventory=inv,
            )
            if quote is None:
                reject_cross += 1
                action_steps["REJECT"] += 1
                self.execution.active_quote = None
                self.execution.queue_ahead_bid = 0.0
                self.execution.queue_ahead_ask = 0.0
                self.execution._posted_bid = 0.0
                self.execution._posted_ask = 0.0
            else:
                last_fair_bid = mx_bid / fx_mid
                last_fair_ask = mx_ask / fx_mid
                last_quote_bid = quote.bid
                last_quote_ask = quote.ask
                action_steps[quote.action or "QUOTE"] += 1
                bid_prices = row.get("trade_bid_prices")
                bid_amounts = row.get("trade_bid_amounts")
                ask_prices = row.get("trade_ask_prices")
                ask_amounts = row.get("trade_ask_amounts")
                if bid_prices is not None and not isinstance(bid_prices, list):
                    bid_prices = list(bid_prices)
                if bid_amounts is not None and not isinstance(bid_amounts, list):
                    bid_amounts = list(bid_amounts)
                if ask_prices is not None and not isinstance(ask_prices, list):
                    ask_prices = list(ask_prices)
                if ask_amounts is not None and not isinstance(ask_amounts, list):
                    ask_amounts = list(ask_amounts)
                self.execution.post_quote(
                    quote,
                    bid_prices=bid_prices,
                    bid_amounts=bid_amounts,
                    ask_prices=ask_prices,
                    ask_amounts=ask_amounts,
                    best_bid_qty=float(row.get("trade_bid_qty") or 0.0),
                    best_ask_qty=float(row.get("trade_ask_qty") or 0.0),
                )

            max_abs_inventory = max(
                max_abs_inventory, abs(self.execution.position.inventory)
            )

            next_ts = (
                timeline_rows[index + 1]["ts"]
                if index + 1 < len(timeline_rows)
                else None
            )
            while trade_index < len(trade_rows):
                trade = trade_rows[trade_index]
                if trade["ts"] < ts:
                    trade_index += 1
                    continue
                if next_ts is not None and trade["ts"] >= next_ts:
                    break
                fill = self.execution.process_trade(
                    timestamp=trade["ts"],
                    trade_price=float(trade["price"]),
                    trade_quantity=float(trade["amount"]),
                    trade_side=str(trade["side"]),
                    best_bid=trade_bid,
                    best_ask=trade_ask,
                    mid=trade_mid,
                )
                if fill is not None:
                    markout.on_fill(
                        side=fill.side,
                        mid=fill.mid_at_fill or trade_mid,
                        price=fill.price,
                        quantity=fill.quantity,
                        ts=fill.timestamp,
                    )
                    self.hedge.on_mm_fill(
                        fill,
                        mx_bid=mx_bid,
                        mx_ask=mx_ask,
                        mx_mid=mx_mid,
                        bn_usdt_mid=float(row["ref_mid"]),
                        fx_mid=fx_mid,
                    )
                trade_index += 1

            bn_marked = self.execution.position.marked_pnl(trade_mid)
            hedge_usdc = self.hedge.marked_usdc(fx_mid=fx_mid, mx_mid_usdt=mx_mid)
            marked = bn_marked + hedge_usdc
            day_key = str(ts)[:10]
            if current_day is None:
                current_day = day_key
                day_start_pnl = marked
                equity_curve.append((ts, marked))
                last_equity_ts = ts
            elif day_key != current_day:
                assert day_start_pnl is not None
                daily_pnl.append((current_day, marked - day_start_pnl, marked))
                current_day = day_key
                day_start_pnl = marked
            if (
                last_equity_ts is None
                or (ts - last_equity_ts).total_seconds() >= sample_seconds
            ):
                equity_curve.append((ts, marked))
                last_equity_ts = ts

            if on_update is not None:
                on_update(row)

        bn_marked = self.execution.position.marked_pnl(last_trade_mid)
        hedge_usdc = self.hedge.marked_usdc(
            fx_mid=last_fx_mid, mx_mid_usdt=last_hedge_mid
        )
        combined = bn_marked + hedge_usdc
        if current_day is not None and day_start_pnl is not None:
            daily_pnl.append((current_day, combined - day_start_pnl, combined))

        report = markout.report()
        metrics = build_metrics(
            position=self.execution.position,
            marked_pnl=bn_marked,
            last_mid=last_trade_mid,
            fills=self.execution.fills,
            equity_curve=equity_curve,
            daily_pnl=daily_pnl,
            markout=report,
            action_steps=action_steps,
            max_abs_inventory=max_abs_inventory,
        )
        return MexcFairResult(
            fills=len(self.execution.fills),
            final_inventory=self.execution.position.inventory,
            bn_marked_pnl=bn_marked,
            hedge_fills=len(self.hedge.fills),
            hedge_inventory=self.hedge.inventory,
            hedge_cash_usdt=self.hedge.cash_usdt,
            hedge_marked_usdc=hedge_usdc,
            combined_marked_pnl=combined,
            net_sui=self.hedge.net_sui(self.execution.position.inventory),
            fill_mode=self.execution.fill_mode,
            maker_buffer_ticks=self.params.maker_buffer_ticks,
            reject_cross=reject_cross,
            fx_stale=fx_stale,
            markout=report,
            metrics=metrics,
            daily_pnl=daily_pnl,
            equity_curve=equity_curve,
            action_steps=action_steps,
            last_fair_bid=last_fair_bid,
            last_fair_ask=last_fair_ask,
            last_quote_bid=last_quote_bid,
            last_quote_ask=last_quote_ask,
        )
