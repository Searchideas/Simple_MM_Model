"""Part 5: Binance-futures vs MEXC-futures SUIUSDT mid basis (5s grid)."""

from __future__ import annotations

import statistics

import polars as pl

from src.loaders import available_dates, load_bbo_range


def acf(x: list[float], lag: int) -> float:
    n = len(x) - lag
    if n < 10:
        return float("nan")
    m = statistics.fmean(x)
    num = sum((x[i] - m) * (x[i + lag] - m) for i in range(n))
    den = sum((xi - m) ** 2 for xi in x)
    return num / den if den else float("nan")


def main() -> None:
    bn_days = set(available_dates("suiusdt", exchange="binance-futures"))
    mx_days = set(available_dates("suiusdt", exchange="mexc-futures"))
    days = sorted(bn_days & mx_days)
    if not days:
        raise SystemExit("No overlapping dates")
    from_d, to_d = days[0], days[-1]
    print(f"overlap={len(days)} {from_d}..{to_d}", flush=True)

    bn = load_bbo_range(
        "suiusdt", from_d, to_d, every="5s", exchange="binance-futures"
    ).select(
        pl.col("ts"),
        pl.col("mid").alias("bn_mid"),
        pl.col("bid_price").alias("bn_bid"),
        pl.col("ask_price").alias("bn_ask"),
    )
    mx = load_bbo_range(
        "suiusdt", from_d, to_d, every="5s", exchange="mexc-futures"
    ).select(
        pl.col("ts"),
        pl.col("mid").alias("mx_mid"),
        pl.col("bid_price").alias("mx_bid"),
        pl.col("ask_price").alias("mx_ask"),
    )
    out = (
        bn.join_asof(mx.sort("ts"), on="ts", strategy="backward")
        .drop_nulls()
        .with_columns(
            ((pl.col("mx_mid") / pl.col("bn_mid") - 1.0) * 1e4).alias("basis_bps"),
            ((pl.col("bn_ask") - pl.col("bn_bid")) / pl.col("bn_mid") * 1e4).alias(
                "bn_spread_bps"
            ),
            ((pl.col("mx_ask") - pl.col("mx_bid")) / pl.col("mx_mid") * 1e4).alias(
                "mx_spread_bps"
            ),
            ((pl.col("bn_bid") / pl.col("mx_ask") - 1.0) * 1e4).alias(
                "sell_bn_buy_mx_bps"
            ),
            ((pl.col("mx_bid") / pl.col("bn_ask") - 1.0) * 1e4).alias(
                "sell_mx_buy_bn_bps"
            ),
        )
        .sort("ts")
    )
    print(f"rows={out.height}", flush=True)

    s = out.select(
        pl.col("basis_bps").mean().alias("basis_mean"),
        pl.col("basis_bps").std().alias("basis_std"),
        pl.col("basis_bps").median().alias("basis_med"),
        pl.col("basis_bps").quantile(0.05).alias("basis_p05"),
        pl.col("basis_bps").quantile(0.95).alias("basis_p95"),
        pl.col("basis_bps").min().alias("basis_min"),
        pl.col("basis_bps").max().alias("basis_max"),
        pl.col("bn_spread_bps").mean().alias("bn_spr_mean"),
        pl.col("mx_spread_bps").mean().alias("mx_spr_mean"),
        pl.col("sell_bn_buy_mx_bps").mean().alias("arb_sbm_mean"),
        pl.col("sell_mx_buy_bn_bps").mean().alias("arb_smb_mean"),
        (pl.col("sell_bn_buy_mx_bps") > 0).mean().alias("frac_sbm_pos"),
        (pl.col("sell_mx_buy_bn_bps") > 0).mean().alias("frac_smb_pos"),
        (pl.col("sell_bn_buy_mx_bps") > 1).mean().alias("frac_sbm_gt1"),
        (pl.col("sell_mx_buy_bn_bps") > 1).mean().alias("frac_smb_gt1"),
        (pl.col("sell_bn_buy_mx_bps") > 2).mean().alias("frac_sbm_gt2"),
        (pl.col("sell_mx_buy_bn_bps") > 2).mean().alias("frac_smb_gt2"),
    ).to_dicts()[0]

    print("OVERALL (basis = MEXC_mid/BN_mid - 1, bps)", flush=True)
    for k, v in s.items():
        print(f"  {k}: {v:.4f}", flush=True)

    daily = (
        out.group_by(pl.col("ts").dt.date().alias("day"))
        .agg(
            pl.col("basis_bps").mean().alias("mean"),
            pl.col("basis_bps").std().alias("std"),
            pl.col("basis_bps").min().alias("min"),
            pl.col("basis_bps").max().alias("max"),
            (pl.col("sell_bn_buy_mx_bps") > 1).mean().alias("frac_sbm_gt1"),
            (pl.col("sell_mx_buy_bn_bps") > 1).mean().alias("frac_smb_gt1"),
        )
        .sort("day")
    )
    print("DAILY basis_bps:", flush=True)
    for r in daily.iter_rows(named=True):
        print(
            f"  {r['day']}  mean={r['mean']:+.2f} std={r['std']:.2f} "
            f"min={r['min']:+.1f} max={r['max']:+.1f}  "
            f"arb>1bps sbm={100 * r['frac_sbm_gt1']:.1f}% "
            f"smb={100 * r['frac_smb_gt1']:.1f}%",
            flush=True,
        )

    b = out["basis_bps"].to_list()
    print(
        f"basis ACF: lag1(5s)={acf(b, 1):.3f} "
        f"lag12(1m)={acf(b, 12):.3f} "
        f"lag60(5m)={acf(b, 60):.3f}",
        flush=True,
    )
    print(
        "arb keys: sbm=sell Binance buy MEXC; smb=sell MEXC buy Binance "
        "(>0 means theoretical cross-touch edge before fees/latency)",
        flush=True,
    )


if __name__ == "__main__":
    main()
