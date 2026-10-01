"""Train a toxic-fill filter on locked hedge edge and replay the later days.

Baseline and filter share the event-L2 fair quote. The split is by day:
train on days before ``--train-before``, score only the days on or after it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.sim.mexc_fair_bt import MexcFairParams
from src.sim.run_event_l2_fair_bt import _intersect_days, run_day
from src.sim.toxicity import HORIZONS_MS, TOXIC_EDGE_TICKS, train_models

MARKOUT_KEYS = [name for name, _ in HORIZONS_MS]


def _params(args: argparse.Namespace) -> MexcFairParams:
    return MexcFairParams(
        trade_tick=args.trade_tick,
        maker_buffer_ticks=args.maker_buffer_ticks,
        order_size=args.order_size,
        max_inventory=args.max_inventory,
        bid_skew_ticks=args.bid_skew_ticks,
        ask_skew_ticks=args.ask_skew_ticks,
        fast_move_ticks=0.0,
        flow_filter="off",
    )


def _run_days(
    days: list[str],
    args: argparse.Namespace,
    *,
    policy: object | None,
    fill_rows: list[dict] | None,
) -> list[dict]:
    params = _params(args)
    out: list[dict] = []
    for day in days:
        row = run_day(
            day=day,
            every=args.every,
            fair_symbol=args.fair_symbol,
            trade_symbol=args.trade_symbol,
            fx_symbol=args.fx_symbol,
            fair_exchange=args.fair_exchange,
            trade_exchange=args.trade_exchange,
            fx_exchange=args.fx_exchange,
            params=params,
            maker_fee=args.maker_fee,
            hedge_taker_fee=args.hedge_taker_fee,
            markout_horizon=1.0,
            policy=policy,
            fill_rows=fill_rows,
        )
        print(
            f"  {day} fills={row['bn_fills']} COMBINED={row['combined']:+.4f} "
            f"BN={row['bn_pnl']:+.4f} hedge={row['mx_pnl']:+.4f} "
            f"pull={row['pull_bid'] + row['pull_ask']}",
            flush=True,
        )
        out.append(row)
    return out


def _sum_pnl(rows: list[dict]) -> dict[str, float]:
    return {
        "combined": sum(r["combined"] for r in rows),
        "bn_pnl": sum(r["bn_pnl"] for r in rows),
        "mx_pnl": sum(r["mx_pnl"] for r in rows),
        "fills": sum(r["bn_fills"] for r in rows),
        "pulls": sum(r["pull_bid"] + r["pull_ask"] for r in rows),
    }


def _fill_stats(rows: list[dict]) -> dict[str, float | None]:
    edges = [float(r["edge_ticks"]) for r in rows if r.get("edge_ticks") is not None]
    out: dict[str, float | None] = {
        "n": float(len(rows)),
        "n_edge": float(len(edges)),
        "mean_edge": (sum(edges) / len(edges)) if edges else None,
        "toxic_rate": (
            sum(1 for e in edges if e <= TOXIC_EDGE_TICKS) / len(edges) if edges else None
        ),
    }
    for key in MARKOUT_KEYS:
        vals = [float(r[key]) for r in rows if r.get(key) is not None]
        out[key] = (sum(vals) / len(vals)) if vals else None
    return out


def _fmt(v: float | None, digits: int = 3) -> str:
    if v is None:
        return "n/a"
    return f"{v:+.{digits}f}"


def _fmt_pct(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{100 * v:.1f}%"


def main() -> None:
    p = argparse.ArgumentParser(description="Toxicity filter vs baseline event-L2 fair hedge")
    p.add_argument("--fair-symbol", default="suiusdt")
    p.add_argument("--trade-symbol", default="suiusdc")
    p.add_argument("--fx-symbol", default="usdcusdt")
    p.add_argument("--fair-exchange", default="hyperliquid")
    p.add_argument("--trade-exchange", default="binance-futures")
    p.add_argument("--fx-exchange", default="binance")
    p.add_argument("--from-date", default="2026-08-26")
    p.add_argument("--to-date", default="2026-09-13")
    p.add_argument("--train-before", default="2026-09-07")
    p.add_argument("--every", default="1s")
    p.add_argument("--maker-buffer-ticks", type=float, default=1.0)
    p.add_argument("--bid-skew-ticks", type=float, default=0.0)
    p.add_argument("--ask-skew-ticks", type=float, default=0.0)
    p.add_argument("--trade-tick", type=float, default=0.0001)
    p.add_argument("--order-size", type=float, default=10.0)
    p.add_argument("--max-inventory", type=float, default=50.0)
    p.add_argument("--maker-fee", type=float, default=0.0)
    p.add_argument("--hedge-taker-fee", type=float, default=0.00045)
    args = p.parse_args()

    _from, _to, days = _intersect_days(
        [
            (args.fair_exchange, args.fair_symbol),
            (args.trade_exchange, args.trade_symbol),
            (args.fx_exchange, args.fx_symbol),
        ],
        None,
        args.from_date,
        args.to_date,
    )
    train_days = [d for d in days if d < args.train_before]
    test_days = [d for d in days if d >= args.train_before]
    if not train_days or not test_days:
        raise SystemExit(
            f"Need days on both sides of {args.train_before}. "
            f"Have {days[0]}..{days[-1]} ({len(days)}d)."
        )
    print(
        f"Toxicity BT {days[0]}..{days[-1]} train<{args.train_before} "
        f"({len(train_days)}d) test {test_days[0]}..{test_days[-1]} ({len(test_days)}d) "
        f"buffer={args.maker_buffer_ticks} size={args.order_size} "
        f"hedge_fee={args.hedge_taker_fee}",
        flush=True,
    )

    print("\n## Baseline (log fills)", flush=True)
    baseline_rows: list[dict] = []
    baseline = _run_days(days, args, policy=None, fill_rows=baseline_rows)
    base_test = [r for r in baseline if r["day"] >= args.train_before]
    base_test_fills = [r for r in baseline_rows if str(r["day"]) >= args.train_before]

    policy, train_report = train_models(baseline_rows, args.train_before)
    print("\n## Baseline fills by side", flush=True)
    for side, name in ((1.0, "buy"), (-1.0, "sell")):
        sub = [
            r
            for r in baseline_rows
            if r.get("side") == side and r.get("edge_ticks") is not None
        ]
        if not sub:
            print(f"  {name}: n=0", flush=True)
            continue
        toxic = sum(1 for r in sub if float(r["edge_ticks"]) <= TOXIC_EDGE_TICKS)
        mean_edge = sum(float(r["edge_ticks"]) for r in sub) / len(sub)
        print(
            f"  {name}: n={len(sub)} toxic={toxic / len(sub):.1%} "
            f"mean_edge={mean_edge:+.3f}",
            flush=True,
        )

    print("\n## Model", flush=True)
    for key in (
        "n_train",
        "n_test",
        "train_toxic_rate",
        "test_toxic_rate",
        "scale_pos_weight",
        "test_precision",
        "test_recall",
        "test_pred_toxic_rate",
        "error",
    ):
        if key in train_report:
            print(f"  {key}={train_report[key]}", flush=True)
    importance = train_report.get("importance") or {}
    if importance:
        ranked = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)
        print(
            "  importance=" + ", ".join(f"{k}:{v:.3f}" for k, v in ranked),
            flush=True,
        )

    if policy is None:
        raise SystemExit("Model was not trained. Baseline log is the only result.")

    print("\n## Toxicity replay (test days only)", flush=True)
    toxic_fills: list[dict] = []
    toxic = _run_days(test_days, args, policy=policy, fill_rows=toxic_fills)

    b = _sum_pnl(base_test)
    t = _sum_pnl(toxic)
    bf = _fill_stats(base_test_fills)
    tf = _fill_stats(toxic_fills)
    lines = [
        f"Test window {test_days[0]}..{test_days[-1]}",
        f"{'metric':<22}{'baseline':>14}{'toxicity':>14}",
        f"{'combined PnL':<22}{b['combined']:+14.4f}{t['combined']:+14.4f}",
        f"{'binance PnL':<22}{b['bn_pnl']:+14.4f}{t['bn_pnl']:+14.4f}",
        f"{'hedge PnL':<22}{b['mx_pnl']:+14.4f}{t['mx_pnl']:+14.4f}",
        f"{'fills':<22}{b['fills']:14.0f}{t['fills']:14.0f}",
        f"{'quote pulls':<22}{b['pulls']:14.0f}{t['pulls']:14.0f}",
        f"{'mean locked edge':<22}{_fmt(bf['mean_edge']):>14}{_fmt(tf['mean_edge']):>14}",
        f"{'toxic fill rate':<22}{_fmt_pct(bf['toxic_rate']):>14}{_fmt_pct(tf['toxic_rate']):>14}",
    ]
    for key in MARKOUT_KEYS:
        lines.append(
            f"{key:<22}{_fmt(bf[key]):>14}{_fmt(tf[key]):>14}"
        )
    delta = t["combined"] - b["combined"]
    lines.append(f"combined delta {delta:+.4f}")
    if delta > 0:
        lines.append("The filter raised test-window combined PnL.")
    else:
        lines.append("The filter did not raise test-window combined PnL.")
    text = "\n".join(lines)
    print("\n" + text, flush=True)
    out = Path("results") / "toxicity_compare.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
