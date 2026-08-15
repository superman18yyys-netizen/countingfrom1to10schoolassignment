#!/usr/bin/env python3
"""Elite fleet trainer: 10-fold walk-forward over the LONG history.

Only BTC/ETH carry 6+ years of USDC candles (SOL/DOGE/ADA listed
later) — this trainer uses those two across every regime 2020-2026:
COVID crash, 2021 mania, 2021-22 collapse, 2024 recovery, 2025-26
bear. 10 folds x ~1200 bars each; params tuned ONLY on data before
each fold (capped grids, purge gap), one-shot scoring per fold.

A bot graduates ONLY if: mean fold excess >= +8%, trades >= 6 per
fold average, and >= 7/10 folds positive. That is a much harder bar
than the 5-fold 2y trial.

Usage: python run_elite.py [--db data/sim6y.db] [--folds 10]
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
import pandas as pd  # noqa: E402

from bot.backtest.engine import run_backtest  # noqa: E402
from bot.config import BotConfig  # noqa: E402
from bot.data.store import Store  # noqa: E402
from bot.paper.account import PaperAccount  # noqa: E402
from bot.strategies.chassis import ChassisStrategy  # noqa: E402
from bot.strategies.sage import SageStrategy  # noqa: E402
from bot.strategies.lab_ideas import DonchianSage  # noqa: E402
from bot.strategies.lab_ideas2 import MTFTrend, SwingRider, VolTrailExit  # noqa: E402
from bot.strategies.lab_ideas3 import RatchetRider  # noqa: E402
from bot.trade_gate import set_fee_model  # noqa: E402

PURGE_BARS = 200
MIN_TRAIN_BARS = 1500     # ~8 months of 4H before the first fold
REPLAY_WINDOW = 1200
GRAD_EXCESS = 8.0         # % mean fold excess to graduate
GRAD_TRADES_PER_FOLD = 6
GRAD_FOLDS_WON = 7        # of 10

FLEET = {
    "sage": (SageStrategy, {"buy_score": [2.5, 3.0, 3.5],
                            "sell_score": [0.0, 0.5]}),
    "donchian_sage": (DonchianSage, {"min_score": [1.5, 2.0],
                                     "entry_period": [20, 30]}),
    "mtf_trend": (MTFTrend, {"day_sma": [20, 30, 50]}),
    "vol_trail_exit": (VolTrailExit, {"atr_mult": [2.5, 3.0, 3.5]}),
    "swing_rider": (SwingRider, {"surge_pct": [0.05, 0.06],
                                 "atr_mult": [3.5, 4.5]}),
    "ratchet_rider": (RatchetRider, {"surge_pct": [0.05, 0.06],
                                     "trail_mult": [3.0, 4.5]}),
}


def bt(df, strat, pair, cfg):
    return run_backtest(df, strat, pair=pair, taker_fee=cfg.taker_fee,
                        slippage=cfg.slippage,
                        position_fraction=cfg.position_fraction,
                        capital=cfg.paper_capital,
                        cash_yield_apy=cfg.cash_yield_apy)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/sim6y.db")
    ap.add_argument("--folds", type=int, default=10)
    args = ap.parse_args()

    cfg = BotConfig.from_yaml()
    set_fee_model(cfg.taker_fee, cfg.slippage)
    store = Store(args.db)
    data = {}
    for pair in ("BTC-USDC", "ETH-USDC"):
        df = store.load_candles(pair, cfg.granularity)
        if df is not None and len(df) > MIN_TRAIN_BARS + 1000:
            data[pair] = df.dropna()
    if not data:
        sys.exit("[elite] no long-history data")
    n = min(len(d) for d in data.values())
    print(f"[elite] {len(data)} pairs x {n} bars "
          f"({n*4/365:.1f} years), {args.folds} folds")

    fold_len = (n - MIN_TRAIN_BARS) // args.folds
    board = []
    for name, (cls, grid) in FLEET.items():
        keys = list(grid)
        trials = [dict(zip(keys, v)) for v in itertools.product(*grid.values())]
        print(f"\n[elite] === {name} ({len(trials)} trials x {args.folds} folds) ===")
        rows = []
        for k in range(args.folds):
            te = n - (args.folds - 1 - k) * fold_len
            tr_end = te - fold_len - PURGE_BARS
            if tr_end <= MIN_TRAIN_BARS - PURGE_BARS:
                continue
            best_sh, best = -1e9, None
            for combo in trials:
                sharpes = []
                for pair, df in data.items():
                    d = df.iloc[len(df)-n:tr_end]
                    r = bt(d, ChassisStrategy(cls(dict(combo))), pair, cfg)
                    sharpes.append(r.sharpe)
                if sharpes and np.mean(sharpes) > best_sh:
                    best_sh, best = float(np.mean(sharpes)), combo
            if best is None:
                continue
            exs, trs, rets = [], 0, []
            for pair, df in data.items():
                fold = df.iloc[len(df)-n+te-fold_len:te]
                r = bt(fold, ChassisStrategy(cls(dict(best))), pair, cfg)
                exs.append(r.excess_return)
                rets.append(r.total_return)
                trs += r.n_trades
            rows.append({"fold": k+1, "params": best,
                         "excess": float(np.mean(exs)),
                         "ret": float(np.mean(rets)),
                         "trades": trs})
            print(f"  fold {k+1:2d}: {best} excess={rows[-1]['excess']*100:+6.1f}% "
                  f"ret={rows[-1]['ret']*100:+6.1f}% trades={trs}")
        if not rows:
            continue
        mean_ex = float(np.mean([r["excess"] for r in rows]))
        mean_ret = float(np.mean([r["ret"] for r in rows]))
        tot_tr = sum(r["trades"] for r in rows)
        won = sum(1 for r in rows if r["excess"] > 0)
        grad = (mean_ex * 100 >= GRAD_EXCESS
                and tot_tr / len(rows) >= GRAD_TRADES_PER_FOLD
                and won >= GRAD_FOLDS_WON)
        board.append({"name": name, "mean_excess_pct": round(mean_ex*100, 2),
                      "mean_ret_pct": round(mean_ret*100, 2),
                      "trades": tot_tr, "folds_won": f"{won}/{len(rows)}",
                      "per_fold": rows, "final_params": rows[-1]["params"],
                      "graduated": grad})
        print(f"  => excess={mean_ex*100:+.2f}% ret={mean_ret*100:+.2f}% "
              f"trades/fold={tot_tr/len(rows):.0f} won={won}/{len(rows)} "
              f"GRADUATED={grad}")

    board.sort(key=lambda r: r["mean_excess_pct"], reverse=True)
    print("\n[elite] ============ ELITE LEADERBOARD (10-fold, 6y BTC+ETH) ============")
    for r in board:
        m = " <== ELITE" if r["graduated"] else ""
        print(f"  {r['name']:<16} excess={r['mean_excess_pct']:+6.1f}% "
              f"ret={r['mean_ret_pct']:+6.1f}% won={r['folds_won']} "
              f"trades={r['trades']}{m}")
    os.makedirs("reports", exist_ok=True)
    with open("reports/elite-results.json", "w") as fh:
        json.dump({"generated": datetime.now(timezone.utc).strftime(
                       "%Y-%m-%d %H:%M UTC"),
                   "span": "6y BTC+ETH 4H", "board": board}, fh, indent=1)
    print("[elite] -> reports/elite-results.json")


if __name__ == "__main__":
    main()
