#!/usr/bin/env python3
"""R&D lab v2: walk-forward invention testing.

v1 flaw: one validation span (a bear) couldn't distinguish ideas —
standing aside scored +45% with zero trades. v2 scores every idea
across FIVE sequential walk-forward folds (each fold's params tuned
ONLY on the data before it, with a purge gap), so ideas prove
themselves in up-trends, ranges, AND crashes. The overall winner then
gets a final live-path replay (bar-by-bar, past-as-live) as the
decision-parity proof before promotion.

Overfit guards: capped grids (<= 6 combos/idea), purge gap between
train and test, fold params frozen before seeing the fold.

Usage: python run_lab.py [--db data/sim.db] [--folds 5]
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
from bot.strategies.lab_ideas import (DonchianSage, RangeSniper,  # noqa: E402
                                      SageRS, SeasonalTrend, VolAwakening)
from bot.strategies.lab_ideas2 import (Committee, MTFTrend,  # noqa: E402
                                       TrendPullback, VolTrailExit)
from bot.strategies.sage import SageStrategy  # noqa: E402
from bot.trade_gate import set_fee_model  # noqa: E402

PURGE_BARS = 200
MIN_TRAIN_BARS = 700        # first fold starts after this much history
REPLAY_WINDOW = 1200
SAGE_BASELINE = 42.7
PROMO_EXCESS = 8.0          # % mean walk-forward OOS excess
PROMO_TRADES = 8

IDEAS = {
    # gen 2 — informed by gen-1 verdicts
    "trend_pullback": (TrendPullback, {"rsi_lo": [30, 35, 40]}),
    "committee": (Committee, {"sage_buy": [2.5, 3.0]}),
    "mtf_trend": (MTFTrend, {"day_sma": [20, 30, 50]}),
    "vol_trail_exit": (VolTrailExit, {"atr_mult": [2.5, 3.0, 3.5]}),
    # gen 1 — proven performers kept as the bar to beat
    "donchian_sage": (DonchianSage, {"min_score": [1.0, 1.5, 2.0],
                                     "entry_period": [20, 30]}),
    "sage_v2": (SageStrategy, {"buy_score": [2.5, 3.0, 3.5],
                               "sell_score": [0.0, 0.5]}),
}


def bt(df, strat, pair, cfg):
    return run_backtest(df, strat, pair=pair, taker_fee=cfg.taker_fee,
                        slippage=cfg.slippage,
                        position_fraction=cfg.position_fraction,
                        capital=cfg.paper_capital,
                        cash_yield_apy=cfg.cash_yield_apy)


def replay_tail(df, strat, pair, cfg, cross_df=None, tail_bars=600):
    """Final decision-parity proof: bar-by-bar replay of the tail."""
    va = df.iloc[-tail_bars:]
    acc = PaperAccount(capital=cfg.paper_capital, taker_fee=cfg.taker_fee,
                       slippage=cfg.slippage,
                       position_fraction=cfg.position_fraction,
                       max_positions=cfg.max_positions,
                       cash_yield_apy=cfg.cash_yield_apy)
    closes = df["close"]
    warm = max(strat.warmup_bars(), 250)
    off = len(df) - len(va)
    for i in range(warm, len(va)):
        lo = max(0, off + i - REPLAY_WINDOW + 1)
        window = df.iloc[lo:off + i + 1]
        if cross_df is not None:
            window = window.copy()
            window.attrs["cross_close"] = cross_df["close"].iloc[
                lo:off + i + 1].to_numpy()
        ts = int(df.index[off + i].timestamp())
        try:
            strat.execute(acc, pair, window, float(closes.iloc[off + i]), ts)
        except Exception as exc:  # noqa: BLE001
            print(f"    [replay] bar {i}: {exc}")
    price_now = float(va["close"].iloc[-1])
    eq = acc.equity({pair: price_now})
    bh = price_now / float(va["open"].iloc[0]) - 1.0
    return {"ret_pct": round((eq / cfg.paper_capital - 1) * 100, 2),
            "bh_pct": round(bh * 100, 2),
            "excess_pct": round((eq / cfg.paper_capital - bh - 1) * 100, 2),
            "trades": acc.n_trades}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/sim.db")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    cfg = BotConfig.from_yaml()
    set_fee_model(cfg.taker_fee, cfg.slippage)
    store = Store(args.db)

    data, btc = {}, None
    for pair in cfg.pairs:
        df = store.load_candles(pair, cfg.granularity)
        if df is not None and len(df) > MIN_TRAIN_BARS + 800:
            data[pair] = df.dropna()
    btc = data.get("BTC-USDC")
    if not data:
        sys.exit("[lab] need data in the db first")
    n = min(len(d) for d in data.values())
    print(f"[lab] {len(data)} pairs, {n} bars each, "
          f"{args.folds} walk-forward folds")

    # fold boundaries on the common bar count (data may start earlier;
    # use each pair's tail so folds align by bar position)
    fold_len = (n - MIN_TRAIN_BARS) // args.folds
    if fold_len < 200:
        sys.exit("[lab] not enough bars for that many folds")

    board = []
    for idea, (cls, grid) in IDEAS.items():
        keys = list(grid)
        trials = [dict(zip(keys, v)) for v in itertools.product(*grid.values())]
        print(f"\n[lab] === {idea} ({len(trials)} trials x {args.folds} folds) ===")
        fold_rows = []
        for k in range(args.folds):
            te = n - (args.folds - 1 - k) * fold_len     # fold end
            tr_end = te - fold_len - PURGE_BARS          # train end
            if tr_end <= MIN_TRAIN_BARS - PURGE_BARS:
                continue
            # tune on train span
            best_sh, best = -1e9, None
            for combo in trials:
                sharpes = []
                for pair, df in data.items():
                    d = df.iloc[-n:tr_end]
                    if len(d) < 400:
                        continue
                    r = bt(d, ChassisStrategy(cls(dict(combo))), pair, cfg)
                    sharpes.append(r.sharpe)
                if sharpes and np.mean(sharpes) > best_sh:
                    best_sh, best = float(np.mean(sharpes)), combo
            if best is None:
                continue
            # score frozen combo on the untouched fold
            exs, trs = [], 0
            for pair, df in data.items():
                fold = df.iloc[-n + te - fold_len:te]
                if len(fold) < 100:
                    continue
                if idea == "sage_rs" and btc is not None:
                    fold = fold.copy()
                    btc_al = btc.reindex(fold.index)["close"]
                    # numpy array: pandas can't hold Series in attrs
                    fold.attrs["cross_close"] = btc_al.to_numpy()
                strat = ChassisStrategy(cls(dict(best)))
                r = bt(fold, strat, pair, cfg)
                exs.append(r.excess_return)
                trs += r.n_trades
            if exs:
                fold_rows.append({"fold": k + 1, "params": best,
                                  "mean_excess_pct": round(100 * float(np.mean(exs)), 2),
                                  "trades": trs})
                print(f"  fold {k+1}: {best} "
                      f"excess={fold_rows[-1]['mean_excess_pct']:+.1f}% trades={trs}")
        if not fold_rows:
            continue
        mean_ex = float(np.mean([f["mean_excess_pct"] for f in fold_rows]))
        tot_tr = sum(f["trades"] for f in fold_rows)
        win_folds = sum(1 for f in fold_rows if f["mean_excess_pct"] > 0)
        promotable = mean_ex >= PROMO_EXCESS and tot_tr >= PROMO_TRADES
        board.append({"idea": idea, "folds": fold_rows,
                      "mean_oos_excess_pct": round(mean_ex, 2),
                      "total_trades": tot_tr,
                      "winning_folds": f"{win_folds}/{len(fold_rows)}",
                      "final_params": fold_rows[-1]["params"],
                      "promotable": promotable})

    board.sort(key=lambda r: r["mean_oos_excess_pct"], reverse=True)
    print(f"\n[lab] ========== WALK-FORWARD LEADERBOARD (Sage baseline "
          f"+{SAGE_BASELINE}% single-span) ==========")
    for r in board:
        m = " <== PROMOTE" if r["promotable"] else ""
        print(f"  {r['idea']:<16} OOS={r['mean_oos_excess_pct']:+6.1f}% "
              f"folds_won={r['winning_folds']} trades={r['total_trades']}{m}")

    # decision-parity replay for the champion (if promotable)
    winner = next((r for r in board if r["promotable"]), None)
    if winner:
        idea = winner["idea"]
        cls = IDEAS[idea][0]
        print(f"\n[lab] final live-path replay for {idea} "
              f"(last 600 bars, past-as-live):")
        for pair, df in data.items():
            strat = ChassisStrategy(cls(dict(winner["final_params"])))
            cross_df = btc if idea == "sage_rs" else None
            out = replay_tail(df, strat, pair, cfg, cross_df=cross_df)
            print(f"    {pair}: ret={out['ret_pct']:+.1f}% "
                  f"bh={out['bh_pct']:+.1f}% excess={out['excess_pct']:+.1f}% "
                  f"trades={out['trades']}")
            winner.setdefault("replay", []).append({"pair": pair, **out})

    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "lab-results.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"generated": datetime.now(timezone.utc).strftime(
                       "%Y-%m-%d %H:%M UTC"),
                   "baseline_sage_oos_pct": SAGE_BASELINE,
                   "method": f"walk-forward {args.folds} folds, purge {PURGE_BARS}",
                   "board": board}, fh, indent=1)
    print("[lab] results -> reports/lab-results.json")


if __name__ == "__main__":
    main()
