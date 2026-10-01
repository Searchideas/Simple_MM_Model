"""Toxic-fill filter for the fair hedge quote.

A fill is toxic when the locked hedge is at least 1 tick worse than the
Binance fill. Markouts are recorded for the report and are not the label.
Flow features are signed so negative means the market is moving against
the side we are about to quote.
"""

from __future__ import annotations

import bisect
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta

FEATURE_NAMES = [
    "side",
    "buffer_ticks",
    "micro_gap_signed",
    "ofi_100_signed",
    "ofi_500_signed",
    "queue_ratio",
    "basis_ticks",
    "spread_ticks",
]

# Gross hedge-vs-fill edge at or below this (ticks) is toxic.
TOXIC_EDGE_TICKS = -1.0
PULL_PROBA = 0.70
MAX_SKEW = 3

HORIZONS_MS = (
    ("markout_100ms", 100),
    ("markout_500ms", 500),
    ("markout_1s", 1000),
)


class TradeOFI:
    """Signed trade size inside a rolling window. Buys positive, sells negative."""

    def __init__(self, window_us: int) -> None:
        self.window = timedelta(microseconds=window_us)
        self._events: deque[tuple[datetime, float]] = deque()
        self._total = 0.0

    def update(self, ts: datetime, signed_qty: float) -> None:
        self._events.append((ts, signed_qty))
        self._total += signed_qty
        self._trim(ts)

    def value(self, ts: datetime) -> float:
        self._trim(ts)
        return self._total

    def _trim(self, ts: datetime) -> None:
        cutoff = ts - self.window
        while self._events and self._events[0][0] < cutoff:
            _, qty = self._events.popleft()
            self._total -= qty


def signed_trade_qty(aggressor_side: str, amount: float) -> float:
    side = aggressor_side.lower().strip()
    if side == "buy":
        return float(amount)
    if side == "sell":
        return -float(amount)
    return 0.0


def locked_edge_ticks(
    *,
    side: str,
    fill_price: float,
    mx_bid: float,
    mx_ask: float,
    fx: float,
    tick: float,
) -> float | None:
    """Hedge price in quote currency minus our fill, in ticks, from our side.

    Buy fill hedges by selling the bid. Sell fill hedges by buying the ask.
    Positive means the hedge locked a better price than the fill.
    """
    if fx <= 0 or tick <= 0 or fill_price <= 0 or not (0 < mx_bid < mx_ask):
        return None
    if side == "buy":
        edge = mx_bid / fx - fill_price
    elif side == "sell":
        edge = fill_price - mx_ask / fx
    else:
        return None
    return edge / tick


def quote_features(
    *,
    side: int,
    buffer_ticks: float,
    tick: float,
    bid: float,
    ask: float,
    bid_qty: float,
    ask_qty: float,
    queue_ahead: float,
    order_qty: float,
    ofi_100: float,
    ofi_500: float,
    hl_bid: float,
    hl_ask: float,
    fx: float,
) -> dict[str, float]:
    """Features known before a fill. ``side`` is +1 bid (we buy), -1 ask."""
    mid = 0.5 * (bid + ask) if bid > 0 and ask > bid else 0.0
    if bid > 0 and ask > bid and bid_qty + ask_qty > 0:
        micro = (bid * ask_qty + ask * bid_qty) / (bid_qty + ask_qty)
    else:
        micro = mid
    micro_gap = (micro - mid) / tick if tick > 0 and mid > 0 else 0.0
    spread = (ask - bid) / tick if tick > 0 and ask > bid else 0.0
    if fx > 0 and hl_bid > 0 and hl_ask > hl_bid and tick > 0 and mid > 0:
        basis = ((0.5 * (hl_bid + hl_ask)) / fx - mid) / tick
    else:
        basis = 0.0
    qty = order_qty if order_qty > 0 else 1.0
    return {
        "side": float(side),
        "buffer_ticks": float(buffer_ticks),
        "micro_gap": micro_gap,
        "micro_gap_signed": micro_gap * side,
        "ofi_100": ofi_100,
        "ofi_500": ofi_500,
        "ofi_100_signed": ofi_100 * side,
        "ofi_500_signed": ofi_500 * side,
        "queue_ahead": float(queue_ahead),
        "queue_ratio": float(queue_ahead) / qty,
        "basis_ticks": basis,
        "spread_ticks": spread,
    }


def feature_vector(row: dict) -> list[float]:
    return [float(row[name]) for name in FEATURE_NAMES]


def attach_markouts(
    rows: list[dict],
    mid_ts: list[datetime],
    mid_px: list[float],
    tick: float,
) -> None:
    """First mid at or after each horizon. No mid yet → leave the field empty.

    ``mid0`` is the book mid at the fill, not the next mid after it.
    """
    if tick <= 0 or not mid_ts:
        return
    for row in rows:
        mid0 = float(row.get("mid0") or 0.0)
        side = float(row.get("side") or 0.0)
        ts = row.get("ts")
        if mid0 <= 0 or side == 0 or not isinstance(ts, datetime):
            continue
        for name, ms in HORIZONS_MS:
            target = ts + timedelta(milliseconds=ms)
            i = bisect.bisect_left(mid_ts, target)
            if i >= len(mid_px):
                row[name] = None
            else:
                row[name] = (mid_px[i] - mid0) / tick * side


@dataclass
class ToxicityPolicy:
    """Pull a side when P(toxic) > 0.70. Otherwise widen by predicted lost edge."""

    clf: object
    reg: object
    max_skew: int = MAX_SKEW
    pull_proba: float = PULL_PROBA

    def decide(self, feat: dict[str, float]) -> tuple[bool, float]:
        """Return (keep_quoting, extra_passive_ticks)."""
        import numpy as np

        x = np.asarray([feature_vector(feat)], dtype=float)
        proba = self.clf.predict_proba(x)
        classes = list(self.clf.classes_)
        if 1 not in classes:
            p_toxic = 0.0
        else:
            p_toxic = float(proba[0, classes.index(1)])
        if p_toxic > self.pull_proba:
            return False, 0.0
        pred_edge = float(self.reg.predict(x)[0])
        if pred_edge >= 0:
            return True, 0.0
        extra = min(self.max_skew, max(0, int(round(-pred_edge))))
        return True, float(extra)


def train_models(rows: list[dict], train_before: str) -> tuple[ToxicityPolicy | None, dict]:
    """Time-split train. Returns (policy or None, report dict)."""
    import numpy as np
    import pandas as pd
    from xgboost import XGBClassifier, XGBRegressor

    frame = pd.DataFrame(rows)
    report: dict = {"n_rows": int(len(frame))}
    if frame.empty or "edge_ticks" not in frame.columns:
        report["error"] = "no labeled fills"
        return None, report
    frame = frame.dropna(subset=["edge_ticks"]).copy()
    frame["toxic"] = (frame["edge_ticks"] <= TOXIC_EDGE_TICKS).astype(int)
    frame["day"] = frame["day"].astype(str)
    train = frame[frame["day"] < train_before]
    test = frame[frame["day"] >= train_before]
    report["n_train"] = int(len(train))
    report["n_test"] = int(len(test))
    report["train_toxic_rate"] = float(train["toxic"].mean()) if len(train) else 0.0
    report["test_toxic_rate"] = float(test["toxic"].mean()) if len(test) else 0.0
    if len(train) < 50 or train["toxic"].nunique() < 2:
        report["error"] = "train split has one class or too few rows"
        return None, report

    n_toxic = int(train["toxic"].sum())
    n_safe = int(len(train) - n_toxic)
    weight = n_safe / n_toxic if n_toxic else 1.0
    report["scale_pos_weight"] = weight

    x_train = np.asarray(train[FEATURE_NAMES], dtype=float)
    y_train = train["toxic"].to_numpy()
    clf = XGBClassifier(
        max_depth=4,
        n_estimators=300,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=weight,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=1,
        random_state=0,
    )
    clf.fit(x_train, y_train)
    reg = XGBRegressor(
        max_depth=4,
        n_estimators=300,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=1,
        random_state=0,
    )
    reg.fit(x_train, train["edge_ticks"].to_numpy(dtype=float))

    report["importance"] = {
        name: float(score)
        for name, score in zip(FEATURE_NAMES, clf.feature_importances_, strict=True)
    }
    if len(test):
        x_test = np.asarray(test[FEATURE_NAMES], dtype=float)
        pred = clf.predict(x_test).astype(int)
        y = test["toxic"].to_numpy().astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        report["test_precision"] = tp / (tp + fp) if (tp + fp) else None
        report["test_recall"] = tp / (tp + fn) if (tp + fn) else 0.0
        report["test_pred_toxic_rate"] = float(pred.mean()) if len(pred) else 0.0
    policy = ToxicityPolicy(clf=clf, reg=reg)
    return policy, report
