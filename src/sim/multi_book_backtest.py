"""Multi-book backtest: AS on SUIUSDT → FX → quote/fill on SUIUSDC.

Parts 1–4 of docs/CROSS_QUOTE_ROADMAP.md (touch fills + markout).
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

import polars as pl

from .cross_quote import CrossQuoteDecision, CrossQuoteParams, build_cross_quote
from .execution import ExecutionSimulator, Fill
from .markout import MarkoutReport, MarkoutTracker
from .market_state import MarketState
from .metrics import CrossQuoteMetrics, build_metrics
from .mexc_hedge import MexcTakerHedge
from .quoting_logic import AvellanedaStoikovModel


@dataclass
class MultiBookResult:
    fills: int
    final_inventory: float
    cash: float
    marked_pnl: float
    last_trade_mid: float
    last_action: str
    last_kappa: float
    last_gamma: float
    reject_cross: int = 0
    fx_stale: int = 0
    mean_basis_residual_bps: float = 0.0
    fill_mode: str = "touch"
    markout: MarkoutReport = field(default_factory=MarkoutReport)
    equity_curve: list[tuple[datetime, float]] = field(default_factory=list)
    daily_pnl: list[tuple[str, float, float]] = field(default_factory=list)
    metrics: CrossQuoteMetrics = field(default_factory=CrossQuoteMetrics)
    execution: ExecutionSimulator | None = None
    # Optional MEXC back-to-back hedge overlay (equity in USDC).
    hedge_fills: int = 0
    hedge_inventory: float = 0.0
    hedge_cash_usdt: float = 0.0
    hedge_skipped: int = 0
    hedge_skipped_basis: int = 0
    hedge_skipped_unfavorable: int = 0
    hedge_partial_qty: float = 0.0
    bn_marked_pnl: float = 0.0
    combined_marked_pnl: float = 0.0
    net_sui: float = 0.0
    hedge: MexcTakerHedge | None = None


@dataclass
class MultiBookSnapshot:
    ts: datetime
    ref_mid: float
    trade_mid: float
    fx_mid: float
    decision: CrossQuoteDecision
    last_fill: Fill | None
    execution: ExecutionSimulator


class MidSampler:
    """1-second mid samples for σ (matches live MidHistory)."""

    def __init__(self, window: int, interval_seconds: float = 1.0) -> None:
        self.window = window
        self.interval_seconds = interval_seconds
        self._mids: deque[float] = deque(maxlen=window)
        self._last_sample_ts: float | None = None

    def on_mid(self, mid: float, ts: datetime) -> None:
        now = ts.timestamp()
        if (
            self._last_sample_ts is None
            or (now - self._last_sample_ts) >= self.interval_seconds
        ):
            self._mids.append(mid)
            self._last_sample_ts = now

    def volatility(self, floor: float) -> float:
        vol = MarketState.calculate_volatility(
            list(self._mids), self.interval_seconds
        )
        return max(vol, floor)


def align_cross_books(
    ref_bbo: pl.DataFrame,
    trade_bbo: pl.DataFrame,
    fx_bbo: pl.DataFrame,
    *,
    include_l25: bool = False,
    hedge_bbo: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """As-of join trade BBO (+ optional L25) + FX mid (+ optional hedge BBO)."""
    ref = ref_bbo.select(
        "ts",
        pl.col("bid_price").alias("ref_bid"),
        pl.col("bid_amount").alias("ref_bid_qty"),
        pl.col("ask_price").alias("ref_ask"),
        pl.col("ask_amount").alias("ref_ask_qty"),
        pl.col("mid").alias("ref_mid"),
    ).sort("ts")
    trade_cols = [
        "ts",
        pl.col("bid_price").alias("trade_bid"),
        pl.col("bid_amount").alias("trade_bid_qty"),
        pl.col("ask_price").alias("trade_ask"),
        pl.col("ask_amount").alias("trade_ask_qty"),
        pl.col("mid").alias("trade_mid"),
    ]
    if include_l25:
        for col in ("bid_prices", "bid_amounts", "ask_prices", "ask_amounts"):
            if col in trade_bbo.columns:
                trade_cols.append(pl.col(col).alias(f"trade_{col}"))
    trade = trade_bbo.select(trade_cols).sort("ts")
    fx = fx_bbo.select(
        "ts",
        pl.col("mid").alias("fx_mid"),
    ).sort("ts")

    out = ref.join_asof(trade, on="ts", strategy="backward")
    out = out.join_asof(fx, on="ts", strategy="backward")
    if hedge_bbo is not None:
        hedge_cols: list = [
            "ts",
            pl.col("bid_price").alias("hedge_bid"),
            pl.col("ask_price").alias("hedge_ask"),
            pl.col("mid").alias("hedge_mid"),
        ]
        if include_l25:
            for col in ("bid_prices", "bid_amounts", "ask_prices", "ask_amounts"):
                if col in hedge_bbo.columns:
                    hedge_cols.append(pl.col(col).alias(f"hedge_{col}"))
        hedge = hedge_bbo.select(hedge_cols).sort("ts")
        out = out.join_asof(hedge, on="ts", strategy="backward")
    return out.drop_nulls().sort("ts")


class MultiBookBacktest:
    """Replay aligned ref/trade/FX books; fill on SUIUSDC trades."""

    def __init__(
        self,
        model: AvellanedaStoikovModel,
        execution: ExecutionSimulator,
        cross_params: CrossQuoteParams,
        volatility_window: int = 120,
        mid_sample_seconds: float = 1.0,
        kappa_path: pl.DataFrame | None = None,
        kappa_fallback: float | None = None,
        markout_horizon_seconds: float = 1.0,
        hedge: MexcTakerHedge | None = None,
    ) -> None:
        self.model = model
        self.execution = execution
        self.cross_params = cross_params
        self.volatility_window = volatility_window
        self.mid_sample_seconds = mid_sample_seconds
        self.kappa_path = kappa_path
        self.kappa_fallback = (
            float(kappa_fallback)
            if kappa_fallback is not None
            else float(model.params.kappa)
        )
        self.markout_horizon_seconds = markout_horizon_seconds
        self.hedge = hedge
        if self.kappa_path is not None and not self.kappa_path.is_empty():
            self.model.estimated_kappa = self.kappa_fallback

    def run(
        self,
        timeline: pl.DataFrame,
        trade_trades: pl.DataFrame,
        on_update: Callable[[MultiBookSnapshot], None] | None = None,
    ) -> MultiBookResult:
        rows = list(timeline.sort("ts").iter_rows(named=True))
        trades = list(trade_trades.sort("ts").iter_rows(named=True))
        return self.run_rows(rows, trades, on_update=on_update)

    def run_rows(
        self,
        timeline_rows: list[dict],
        trade_rows: list[dict],
        on_update: Callable[[MultiBookSnapshot], None] | None = None,
    ) -> MultiBookResult:
        trade_index = 0
        sampler = MidSampler(
            window=self.volatility_window,
            interval_seconds=self.mid_sample_seconds,
        )
        markout = MarkoutTracker(
            tick=self.cross_params.trade_tick,
            horizon_seconds=self.markout_horizon_seconds,
        )
        last_decision: CrossQuoteDecision | None = None
        last_trade_mid = 0.0
        last_fx_mid = 0.0
        last_hedge_mid = 0.0
        reject_cross = 0
        fx_stale = 0
        basis_sum = 0.0
        basis_n = 0
        equity_curve: list[tuple[datetime, float]] = []
        daily_pnl: list[tuple[str, float, float]] = []
        day_start_pnl: float | None = None
        current_day: str | None = None
        last_equity_ts: datetime | None = None
        inventory_open_ts: datetime | None = None
        sample_seconds = 60.0
        action_steps: Counter[str] = Counter()
        max_abs_inventory = 0.0

        for index, row in enumerate(timeline_rows):
            ts: datetime = row["ts"]
            ref_bid = float(row["ref_bid"])
            ref_ask = float(row["ref_ask"])
            ref_mid = float(row["ref_mid"])
            trade_bid = float(row["trade_bid"])
            trade_ask = float(row["trade_ask"])
            trade_mid = float(row["trade_mid"])
            fx_mid = float(row["fx_mid"])
            last_trade_mid = trade_mid
            last_fx_mid = fx_mid
            if row.get("hedge_mid") is not None:
                last_hedge_mid = float(row["hedge_mid"])

            if fx_mid <= 0:
                fx_stale += 1
                self.execution.active_quote = None
                self.execution.queue_ahead_bid = 0.0
                self.execution.queue_ahead_ask = 0.0
                self.execution._posted_bid = 0.0
                self.execution._posted_ask = 0.0
                last_decision = CrossQuoteDecision(
                    quote=None,
                    action="FX_STALE",
                    bid_usdc=0.0,
                    ask_usdc=0.0,
                    fx_mid=fx_mid,
                    enable_bid=False,
                    enable_ask=False,
                    size=0.0,
                )
                continue

            sampler.on_mid(ref_mid, ts)
            markout.on_mid(trade_mid, ts)
            vol = sampler.volatility(self.cross_params.vol_floor)

            if self.kappa_path is not None:
                from .fill_probabilty import kappa_at_time

                self.model.estimated_kappa = kappa_at_time(
                    self.kappa_path,
                    ts,
                    self.kappa_fallback,
                )

            market = MarketState(
                timestamp=ts,
                best_bid=ref_bid,
                best_bid_volume=float(row.get("ref_bid_qty") or 0.0),
                best_ask=ref_ask,
                best_ask_volume=float(row.get("ref_ask_qty") or 0.0),
                volatility=vol,
            )
            as_quote = self.model.calculate_quotes(
                market,
                self.execution.position,
                enforce_maker=False,
            )

            q = self.execution.position.inventory
            if abs(q) < 0.05:
                inventory_open_ts = None
            elif inventory_open_ts is None:
                inventory_open_ts = ts
            hold_s = (
                (ts - inventory_open_ts).total_seconds()
                if inventory_open_ts is not None
                else 0.0
            )

            decision = build_cross_quote(
                as_quote=as_quote,
                inventory=q,
                avg_entry=self.execution.position.avg_entry_price,
                trade_bid=trade_bid,
                trade_ask=trade_ask,
                fx_mid=fx_mid,
                params=self.cross_params,
                inventory_open_age_seconds=hold_s,
            )
            last_decision = decision
            action_steps[decision.action or "OTHER"] += 1
            max_abs_inventory = max(
                max_abs_inventory, abs(self.execution.position.inventory)
            )
            if decision.action == "REJECT_CROSS":
                reject_cross += 1
            if decision.basis_residual_bps:
                basis_sum += decision.basis_residual_bps
                basis_n += 1

            if decision.quote is not None and (
                decision.enable_bid or decision.enable_ask
            ):
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
                    decision.quote,
                    bid_prices=bid_prices,
                    bid_amounts=bid_amounts,
                    ask_prices=ask_prices,
                    ask_amounts=ask_amounts,
                    best_bid_qty=float(row.get("trade_bid_qty") or 0.0),
                    best_ask_qty=float(row.get("trade_ask_qty") or 0.0),
                )
            else:
                self.execution.active_quote = None
                self.execution.queue_ahead_bid = 0.0
                self.execution.queue_ahead_ask = 0.0
                self.execution._posted_bid = 0.0
                self.execution._posted_ask = 0.0

            next_ts = (
                timeline_rows[index + 1]["ts"]
                if index + 1 < len(timeline_rows)
                else None
            )
            last_fill: Fill | None = None
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
                    last_fill = fill
                    markout.on_fill(
                        side=fill.side,
                        mid=fill.mid_at_fill or trade_mid,
                        price=fill.price,
                        quantity=fill.quantity,
                        ts=fill.timestamp,
                    )
                    if self.hedge is not None:
                        hb = row.get("hedge_bid_prices")
                        ha = row.get("hedge_bid_amounts")
                        hap = row.get("hedge_ask_prices")
                        haa = row.get("hedge_ask_amounts")
                        self.hedge.on_mm_fill(
                            fill,
                            mx_bid=float(row.get("hedge_bid") or 0.0),
                            mx_ask=float(row.get("hedge_ask") or 0.0),
                            mx_mid=float(row.get("hedge_mid") or 0.0),
                            bn_usdt_mid=ref_mid,
                            bid_prices=list(hb) if hb is not None else None,
                            bid_amounts=list(ha) if ha is not None else None,
                            ask_prices=list(hap) if hap is not None else None,
                            ask_amounts=list(haa) if haa is not None else None,
                        )
                    if abs(self.execution.position.inventory) >= 0.05:
                        if inventory_open_ts is None:
                            inventory_open_ts = trade["ts"]
                    else:
                        inventory_open_ts = None
                trade_index += 1

            bn_marked = self.execution.position.marked_pnl(trade_mid)
            if self.hedge is not None:
                hx = float(row.get("hedge_mid") or 0.0)
                marked = bn_marked + self.hedge.marked_usdc(
                    fx_mid=fx_mid, mx_mid_usdt=hx
                )
            else:
                marked = bn_marked
            day_key = str(ts)[:10]
            if current_day is None:
                current_day = day_key
                day_start_pnl = marked
                equity_curve.append((ts, marked))
                last_equity_ts = ts
            elif day_key != current_day:
                daily_pnl.append(
                    (current_day, marked - (day_start_pnl or 0.0), marked)
                )
                current_day = day_key
                day_start_pnl = marked
                equity_curve.append((ts, marked))
                last_equity_ts = ts
            elif (
                last_equity_ts is None
                or (ts - last_equity_ts).total_seconds() >= sample_seconds
                or index == len(timeline_rows) - 1
            ):
                equity_curve.append((ts, marked))
                last_equity_ts = ts

            if on_update is not None and last_decision is not None:
                on_update(
                    MultiBookSnapshot(
                        ts=ts,
                        ref_mid=ref_mid,
                        trade_mid=trade_mid,
                        fx_mid=fx_mid,
                        decision=last_decision,
                        last_fill=last_fill,
                        execution=self.execution,
                    )
                )

        markout.flush(last_trade_mid)
        bn_final = self.execution.position.marked_pnl(last_trade_mid)
        if self.hedge is not None:
            final_pnl = bn_final + self.hedge.marked_usdc(
                fx_mid=last_fx_mid, mx_mid_usdt=last_hedge_mid
            )
        else:
            final_pnl = bn_final
        if current_day is not None:
            daily_pnl.append(
                (current_day, final_pnl - (day_start_pnl or 0.0), final_pnl)
            )
        if not equity_curve or equity_curve[-1][1] != final_pnl:
            stamp = timeline_rows[-1]["ts"] if timeline_rows else datetime.utcnow()
            equity_curve.append((stamp, final_pnl))
        max_abs_inventory = max(
            max_abs_inventory, abs(self.execution.position.inventory)
        )
        markout_report = markout.report()
        metrics = build_metrics(
            position=self.execution.position,
            marked_pnl=bn_final,
            last_mid=last_trade_mid,
            fills=self.execution.fills,
            equity_curve=equity_curve,
            daily_pnl=daily_pnl,
            markout=markout_report,
            action_steps=action_steps,
            max_abs_inventory=max_abs_inventory,
        )

        hedge = self.hedge
        return MultiBookResult(
            fills=len(self.execution.fills),
            final_inventory=self.execution.position.inventory,
            cash=self.execution.position.cash,
            marked_pnl=final_pnl,
            last_trade_mid=last_trade_mid,
            last_action=last_decision.action if last_decision else "",
            last_kappa=float(self.model.active_kappa()),
            last_gamma=float(self.model.params.base_gamma),
            reject_cross=reject_cross,
            fx_stale=fx_stale,
            mean_basis_residual_bps=(basis_sum / basis_n) if basis_n else 0.0,
            fill_mode=self.execution.fill_mode,
            markout=markout_report,
            equity_curve=equity_curve,
            daily_pnl=daily_pnl,
            metrics=metrics,
            execution=self.execution,
            hedge_fills=len(hedge.fills) if hedge else 0,
            hedge_inventory=hedge.inventory if hedge else 0.0,
            hedge_cash_usdt=hedge.cash_usdt if hedge else 0.0,
            hedge_skipped=hedge.skipped if hedge else 0,
            hedge_skipped_basis=hedge.skipped_basis if hedge else 0,
            hedge_skipped_unfavorable=hedge.skipped_unfavorable if hedge else 0,
            hedge_partial_qty=hedge.partial_qty if hedge else 0.0,
            bn_marked_pnl=bn_final,
            combined_marked_pnl=final_pnl,
            net_sui=(
                hedge.net_sui(self.execution.position.inventory) if hedge else self.execution.position.inventory
            ),
            hedge=hedge,
        )
