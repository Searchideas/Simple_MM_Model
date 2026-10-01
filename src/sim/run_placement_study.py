"""Where to rest the SUIUSDC limit under the event-level adverse filter.

Sweeps buffer distance on the same fair quote. Each fill keeps the 2s mid
move and the signed order-flow imbalance from the moment the order was posted.
Net edge subtracts the Hyperliquid taker fee, in ticks.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.sim.mexc_fair_bt import MexcFairParams
from src.sim.run_event_l2_fair_bt import _intersect_days, run_day

BUFFERS = (1.0, 3.0, 5.0, 8.0)
# Spread across the 19-day window, including the busiest sweep days.
STUDY_DAYS = (
    "2026-08-26",
    "2026-08-28",
    "2026-08-31",
    "2026-09-02",
    "2026-09-05",
    "2026-09-08",
    "2026-09-11",
    "2026-09-13",
)
FEE = 0.00045
TICK = 0.0001


def _fee_ticks(price: float) -> float:
    if price <= 0 or TICK <= 0:
        return 0.0
    return FEE * price / TICK


def main() -> None:
    _from, _to, available = _intersect_days(
        [
            ("hyperliquid", "suiusdt"),
            ("binance-futures", "suiusdc"),
            ("binance", "usdcusdt"),
        ],
        None,
        "2026-08-26",
        "2026-09-13",
    )
    days = [d for d in STUDY_DAYS if d in available]
    out_path = Path("results") / "placement_study.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Placement study {days[0]}..{days[-1]} n_days={len(days)} "
        f"buffers={list(BUFFERS)} flow=either fast=8 fee={FEE}",
        flush=True,
    )
    with out_path.open("w", encoding="utf-8") as fh:
        for buf in BUFFERS:
            params = MexcFairParams(
                trade_tick=TICK,
                maker_buffer_ticks=buf,
                order_size=10.0,
                max_inventory=50.0,
                fast_move_ticks=8.0,
                fast_move_lookback_seconds=2.0,
                fast_move_pause_seconds=3.0,
                flow_filter="either",
            )
            for day in days:
                rows: list[dict] = []
                result = run_day(
                    day=day,
                    every="1s",
                    fair_symbol="suiusdt",
                    trade_symbol="suiusdc",
                    fx_symbol="usdcusdt",
                    fair_exchange="hyperliquid",
                    trade_exchange="binance-futures",
                    fx_exchange="binance",
                    params=params,
                    maker_fee=0.0,
                    hedge_taker_fee=FEE,
                    markout_horizon=1.0,
                    fill_rows=rows,
                )
                fills = []
                for row in rows:
                    edge = row.get("edge_ticks")
                    price = float(row.get("price") or 0.0)
                    net = None if edge is None else float(edge) - _fee_ticks(price)
                    fills.append(
                        {
                            "side": row.get("side"),
                            "qty": row.get("qty"),
                            "price": price,
                            "edge_ticks": edge,
                            "net_ticks": net,
                            "ofi_100_signed": row.get("ofi_100_signed"),
                            "ofi_500_signed": row.get("ofi_500_signed"),
                            "micro_gap_signed": row.get("micro_gap_signed"),
                            "move_2s_ticks": row.get("move_2s_ticks"),
                            "markout_1s": row.get("markout_1s"),
                        }
                    )
                record = {
                    "day": day,
                    "buffer": buf,
                    "combined": result["combined"],
                    "bn_pnl": result["bn_pnl"],
                    "hedge_pnl": result["mx_pnl"],
                    "fills": result["bn_fills"],
                    "adverse": result["adverse"],
                    "flow_pulls": result["pull_bid"] + result["pull_ask"],
                    "pause_steps": result["pause_steps"],
                    "fill_rows": fills,
                }
                fh.write(json.dumps(record) + "\n")
                fh.flush()
                print(
                    f"  buf={buf:g} {day} fills={result['bn_fills']} "
                    f"COMBINED={result['combined']:+.4f} adv={100*result['adverse']:.1f}%",
                    flush=True,
                )
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
