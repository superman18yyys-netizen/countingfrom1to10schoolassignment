#!/usr/bin/env python3
"""A-000 runner: oracle ceiling + walk-forward tuning + full-period
report on the expanded USDC universe (6y, 4H).

Answers the question: "how much would $100 have become?"
  1. ORACLE (perfect foresight, optimal sizing): the ceiling.
  2. A-000 (causal rotation allocator): what a real bot achieves.
  3. Benchmarks: buy&hold BTC, buy&hold best-coin, cash yield.

Walk-forward (10 folds): top_k / cash_buffer / swap_margin tuned on
data BEFORE each fold (capped grid), one-shot scored per fold.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from bot.a000 import A000Config, oracle_return, run_a000  # noqa: E402
from bot.data.store import Store  # noqa: E402

PURGE_BARS = 200
MIN_TRAIN = 1500
LONG_BARS = 1900     # pairs with >= this join the walk-forward tuning
GRID = {"core_frac": [0.50, 0.65],
        "top_k": [3, 4, 6],
        "cash_buffer": [0.25],
        "swap_margin": [0.02]}   # 6 trials (core-satellite family)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/universe.db")
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--capital", type=float, default=100.0)
    args = ap.parse_args()

    store = Store(args.db)
    all_data, long_data = {}, {}
    pair_list = [r[0] for r in store.conn.execute(
        "SELECT DISTINCT pair FROM candles WHERE granularity='FOUR_HOUR'")]
    for pair in pair_list:
        df = store.load_candles(pair, "FOUR_HOUR")
        if df is None or len(df) < 400:
            continue
        df = df.dropna()
        all_data[pair] = df
        if len(df) >= LONG_BARS:
            long_data[pair] = df
    if not all_data:
        sys.exit("[a000] no universe data")
    pairs = sorted(all_data)
    long_pairs = sorted(long_data)
    n_common = min(len(d) for d in long_data.values())
    print(f"[a000] universe: {len(pairs)} coins ({len(long_pairs)} with "
          f"long history for tuning) | common: {n_common} bars "
          f"({n_common / 2190:.1f}y)")

    # ---------- 1. ORACLE (full universe) ----------
    closes = {p: all_data[p]["close"] for p in pairs}
    highs = {p: all_data[p]["high"] for p in pairs}
    lows = {p: all_data[p]["low"] for p in pairs}
    o = oracle_return(closes, capital=args.capital)
    print(f"== ORACLE ceiling (perfect foresight, optimal sizing) ==")
    print(f"   ${args.capital:.0f} -> ${o['equity']:,.2f}  "
          f"({o['return_pct']:+,.0f}%) in {o['trades']} trades\n")

    # ---------- 2. WALK-FORWARD TUNING (long-history pairs only) ------
    fold_len = (n_common - MIN_TRAIN) // args.folds
    trials = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    print(f"== A-000 walk-forward ({len(trials)} trials x {args.folds} "
          f"folds, {len(long_pairs)} coins) ==")
    lcloses = {p: long_data[p]["close"] for p in long_pairs}
    lhighs = {p: long_data[p]["high"] for p in long_pairs}
    llows = {p: long_data[p]["low"] for p in long_pairs}
    fold_rows = []
    # timeline positions: common window is the last n_common bars of
    # the union timeline; fold spans slice it
    base = len(sorted(set().union(*[set(d.index) for d in long_data.values()]))) \
        - n_common
    for k in range(args.folds):
        te = n_common - (args.folds - 1 - k) * fold_len
        tr_end = te - fold_len - PURGE_BARS
        if tr_end <= MIN_TRAIN:
            continue
        best_sh, best = -1e9, None
        for combo in trials:
            cfg = A000Config(capital=args.capital, **combo,
                             trade_from=base, trade_to=base + tr_end)
            r = run_a000(lcloses, lhighs, llows, cfg)
            if r.return_pct > best_sh:
                best_sh, best = r.return_pct, combo
        cfg = A000Config(capital=args.capital, **best,
                         trade_from=base + te - fold_len,
                         trade_to=base + te)
        r = run_a000(lcloses, lhighs, llows, cfg)
        fold_rows.append({"fold": k + 1, "params": best,
                          "ret_pct": round(r.return_pct, 2),
                          "dd_pct": round(r.max_dd_pct, 2),
                          "trades": r.trades})
        print(f"  fold {k + 1:2d}: {best} ret={r.return_pct:+8.1f}% "
              f"dd={r.max_dd_pct:.1f}% trades={r.trades}")

    # ---------- 3. FULL-PERIOD REPORT (ENTIRE universe) ---------------
    final_params = fold_rows[-1]["params"] if fold_rows else \
        {"core_frac": 0.65, "top_k": 4, "cash_buffer": 0.25,
         "swap_margin": 0.02}
    full = run_a000(closes, highs, lows,
                    A000Config(capital=args.capital, **final_params))

    # benchmarks
    btc = closes.get("BTC-USDC")
    eth = closes.get("ETH-USDC")
    bh = {}
    for name, c in (("BTC", btc), ("ETH", eth)):
        if c is not None:
            bh[name] = round((c.iloc[-1] / c.iloc[0] - 1) * 100, 1)
    best_coin = max((c.iloc[-1] / c.iloc[0] for c in
                     (long_data[p]["close"] for p in long_pairs)))
    span_years = len(all_data.get("BTC-USDC", next(iter(all_data.values())))) \
        / 2190
    cash_ret = ((1 + 0.045) ** span_years - 1) * 100

    print(f"\n== FULL 6Y REPORT (${args.capital:.0f} start, fees + "
          f"slip on, final params {final_params}) ==")
    print(f"   A-000:   ${full.equity:,.2f}  ({full.return_pct:+,.0f}%)  "
          f"maxDD {full.max_dd_pct:.1f}%  {full.trades} trades  "
          f"fees ${full.fees_paid:,.0f}")
    print(f"   ORACLE:  ${o['equity']:,.2f}  ({o['return_pct']:+,.0f}%)")
    for name, v in bh.items():
        print(f"   buy&hold {name}: {v:+,.0f}%")
    print(f"   buy&hold best coin: {(best_coin - 1) * 100:+,.0f}%")
    print(f"   idle USDC yield: {cash_ret:+.0f}%")
    print(f"   capture rate: {full.return_pct / o['return_pct'] * 100:.1f}% "
          f"of the oracle ceiling")

    out = {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
           "universe": pairs, "capital": args.capital,
           "oracle": o, "fold_results": fold_rows,
           "final_params": final_params,
           "full_period": {"equity": round(full.equity, 2),
                           "return_pct": round(full.return_pct, 2),
                           "max_dd_pct": round(full.max_dd_pct, 2),
                           "trades": full.trades,
                           "fees_paid": round(full.fees_paid, 2)},
           "benchmarks": {"buy_hold": bh,
                          "best_coin_pct": round((best_coin - 1) * 100, 1),
                          "cash_yield_pct": round(cash_ret, 1)}}
    os.makedirs("reports", exist_ok=True)
    with open("reports/a000-results.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print("\n[a000] -> reports/a000-results.json")


if __name__ == "__main__":
    main()
