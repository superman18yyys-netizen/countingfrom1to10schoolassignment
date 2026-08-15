#!/usr/bin/env python3
"""Miss analysis: what did the bots leave on the table, and why?

Maps the opportunity set (every >=8% swing, net of fees) via zigzag
pivots, measures each bot's capture rate, and attributes misses:
signal-blind (raw signal never fired) vs gate-blocked (chassis killed
a fired signal) vs early exit (exited before the swing peak).

Usage: python run_miss_analysis.py [--db data/sim.db] [--pct 0.08]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from bot.backtest.engine import run_backtest  # noqa: E402
from bot.config import BotConfig  # noqa: E402
from bot.data.store import Store  # noqa: E402
from bot.strategies.chassis import ChassisStrategy  # noqa: E402
from bot.strategies.sage import SageStrategy  # noqa: E402
from bot.strategies.lab_ideas import DonchianSage  # noqa: E402
from bot.strategies.lab_ideas2 import MTFTrend, SwingRider, VolTrailExit  # noqa: E402
from bot.trade_gate import set_fee_model  # noqa: E402

BOTS = [
    ("sage", SageStrategy, {"buy_score": 3.0, "sell_score": 0.0}),
    ("donchian_sage", DonchianSage, {"min_score": 2.0, "entry_period": 30}),
    ("mtf_trend", MTFTrend, {"day_sma": 30}),
    ("vol_trail_exit", VolTrailExit, {"atr_mult": 3.0}),
    ("swing_rider", SwingRider, {"surge_pct": 0.05, "atr_mult": 4.5}),
]


def zigzag(close, pct):
    pivots, mode, ext_i, ext_p = [], "init", 0, float(close.iloc[0])
    for i in range(len(close)):
        p = float(close.iloc[i])
        if mode in ("init", "up"):
            if mode == "init":
                if p > ext_p * (1 + pct):
                    mode, ext_i, ext_p = "up", i, p
                    continue
                if p < ext_p:
                    ext_i, ext_p = i, p
            elif p > ext_p:
                ext_i, ext_p = i, p
            elif p <= ext_p * (1 - pct):
                pivots.append((ext_i, ext_p, "hi"))
                mode, ext_i, ext_p = "down", i, p
        if mode == "down":
            if p < ext_p:
                ext_i, ext_p = i, p
            elif p >= ext_p * (1 + pct):
                pivots.append((ext_i, ext_p, "lo"))
                mode, ext_i, ext_p = "up", i, p
    return pivots


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/sim.db")
    ap.add_argument("--pct", type=float, default=0.08)
    args = ap.parse_args()

    cfg = BotConfig.from_yaml()
    set_fee_model(cfg.taker_fee, cfg.slippage)
    store = Store(args.db)

    opps = {}
    total_s, total_net = 0, 0.0
    print(f"=== OPPORTUNITY SET (swings >= {args.pct*100:.0f}%, net of fees) ===")
    for pair in cfg.pairs:
        df = store.load_candles(pair, cfg.granularity)
        if df is None or len(df) < 2000:
            continue
        df = df.dropna()
        segs, pending = [], None
        for i, p, kind in zigzag(df["close"], args.pct):
            if kind == "lo":
                pending = (i, p)
            elif kind == "hi" and pending:
                net = p / pending[1] - 1 - 0.014
                if net > 0.02:
                    segs.append({"lo_i": pending[0], "hi_i": i, "net": net})
                pending = None
        opps[pair] = (df, segs)
        total_s += len(segs)
        total_net += sum(s["net"] for s in segs)
        print(f"  {pair:<10} {len(segs):3d} swings, "
              f"sum {sum(s['net'] for s in segs)*100:+7.1f}%")
    print(f"  TOTAL: {total_s} swings, {total_net*100:+.0f}% summed net\n")

    print("=== CAPTURE + MISS ATTRIBUTION ===")
    print(f"{'bot':<16}{'trades':>7}{'capture%':>10}{'sig-blind':>11}"
          f"{'gate-blk':>10}{'early-exit':>12}{'left%':>7}")
    for name, cls, params in BOTS:
        tot = cap = blind = gated = early = 0
        leaves = []
        for pair, (df, segs) in opps.items():
            raw = cls(dict(params))
            r_raw = run_backtest(df, raw, pair=pair,
                                 position_fraction=cfg.position_fraction)
            r_g = run_backtest(df, ChassisStrategy(cls(dict(params))),
                               pair=pair, position_fraction=cfg.position_fraction,
                               cash_yield_apy=cfg.cash_yield_apy)
            raw_buys = {df.index.get_loc(t.entry_ts) for t in r_raw.trades}
            for seg in segs:
                hit = [t for t in r_g.trades
                       if seg["lo_i"] <= df.index.get_loc(t.entry_ts)
                       <= seg["hi_i"]]
                if hit:
                    cap += 1
                    leaves.append(seg["net"] - max(t.pnl_pct for t in hit))
                    for t in hit:
                        if df.index.get_loc(t.exit_ts) < seg["hi_i"] - 2:
                            early += 1
                            break
                else:
                    if any(seg["lo_i"] <= b <= seg["hi_i"] for b in raw_buys):
                        gated += 1
                    else:
                        blind += 1
                tot += 1
        print(f"{name:<16}{r_g.n_trades:>7}{100*cap/max(tot,1):>9.0f}%"
              f"{blind:>11}{gated:>10}{early:>12}"
              f"{(np.mean(leaves)*100 if leaves else 0):>+7.1f}")


if __name__ == "__main__":
    main()
