# Cross-quote / live-parity backtest roadmap

Price on **SUIUSDT**, convert via **USDCUSDT** spot, quote maker-only on **SUIUSDC**.
Optional later: MEXC SUIUSDT hedge (0/0 fees) with basis risk.

## Live flow (parity target)

1. Classic AS on SUIUSDT BBO (+ mid vol, κ)
2. Convert USDT → USDC: `px_usdc = px_usdt / fx_mid * (1 - basis_bps/1e4)`
3. Clamp to SUIUSDC book (GTX / maker safety ticks)
4. Inventory / cover / join / toxicity / fast-move gates
5. Fills only on SUIUSDC

## Parts

| Part | Status | Scope |
|------|--------|--------|
| **0** Spec freeze | planned | Live-parity checklist; v1 vs later knobs |
| **1** Multi-book event engine | **done** | Sync SUIUSDT + USDCUSDT + SUIUSDC; quote on ref BBO; fill on trade trades |
| **2** Port live quoting | **done** | AS `enforce_maker=False`, FX+4bps, clamp, soft-inv, cover/join/cut |
| **3** Fill model ladder | **partial** | A touch/through; **B L25 queue**; **C event-L2** (`run_event_l2_fair_bt`, Python) |
| **4** Metrics | **done** | Markout + equity DD + daily win + action attribution (`metrics.py`) |
| **5** MEXC hedge research | **done** | Basis stats; B2B; hybrid (favorable MEXC else 12% cover + requote when flat) |
| **6** C++ hot path | planned | Only if Python L25/L2 too slow for sweeps |
| **7** Live rollout | planned | Dry-run parity → small live → paper MEXC → tiny hedge |

## Data layout

- `data/parquet/binance-futures/suiusdt|suiusdc/`
- `data/parquet/binance/usdcusdt/` (spot FX; not USDTUSDC)
- `data/parquet/mexc-futures/suiusdt/` (Part 5 hedge research; Tardis id `SUI_USDT`)

## v1 knobs (Parts 1–2)

- ON: AS, FX convert, basis bps, maker clamp, soft inventory, cover/join/cut/hold
- OFF (stub later): toxicity gate, fast-move gate, post-fill cooldown, live κ streaming fit  
  (day κ from `fill_probability` on SUIUSDT is OK)

## Commands

```bash
# Default = production-aligned guards (queue L25, κ clamp, hard flatten, fee0, vol floor)
python -m src.sim.run_cross_quote_bt --date all --every 1s

# MEXC always-B2B (no favorable filter)
python -m src.sim.run_cross_quote_hedge_bt --date all --every 1s --fill-mode queue --no-hedge-favorable

# MEXC fair → FX → SUIUSDC maker (1 tick off) → hit MEXC B2B; both-side PnL
python -m src.sim.run_mexc_fair_hedge_bt --date all --every 1s --fill-mode queue --maker-buffer-ticks 1

# Event-L2 queue + multi-day buffer/skew/fast-move grid
python -m src.sim.run_event_l2_fair_bt --date all --from-date 2026-08-26 --to-date 2026-09-13 --grid --order-size 10 --max-inventory 50

# Part 5 compare unhedged vs hedge rails
python -m src.sim.run_part5_hedge_compare --from-date 2026-08-26 --to-date 2026-09-13 --every 5s --fill-mode touch

# Touch-only (no queue ahead)
python -m src.sim.run_cross_quote_bt --date 2026-09-01 --every 1s --fill-mode touch
```

## Default BT mode (current)

- Fill: **queue** (L25 size-ahead deplete; Part 3B)
- κ: fit on SUIUSDT, clamp **[0.05, 0.5]**
- Hard flatten: join BBO when `|q| ≥ max_inventory`
- `maker_fee=0`, `min_spread=0.0004032`, `vol_floor=1e-4`
- Cover: TP join **5%**; cut **12% @10x** → join BBO at breach; trail **1% @10x** when deeper
- FX basis haircut **4 bps**; maker safety **2 ticks**

## Implementation notes

- `execution.fill_mode`: `queue` (default) | `touch` | `through`
- `queue_l25.size_ahead_at_price` + `ExecutionSimulator` deplete-on-trade
- `align_cross_books(..., include_l25=True)` when `--fill-mode queue`
- `markout.py`: post-fill mid mark-out → adverse rate + pnl proxy
- `metrics.py` (Part 4): realized/unrealized, max|inv|, equity peak/max DD, daily win rate, fill attribution by QUOTE/COVER/JOIN_TP/CUT/FLATTEN
- Smoke / 19d: through was fantasy (+28); touch+κ spike lost (−3.9 @ inv−50); touch+κ clamp+flatten → **+18.5 / inv−27.5**
- Still missing vs live: GTX rejects, REPLACE_TICKS lag, toxicity *gating*
- Live env: `KAPPA_MIN`, `KAPPA_MAX`, `HARD_FLATTEN_AT_MAX`
