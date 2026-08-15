#!/usr/bin/env python3
"""Sage expert trainer: tune thresholds on TRAIN span, then prove the
winner on an untouched VALIDATION span replayed BAR-BY-BAR through the
REAL live decision path (strategy.execute + chassis gates + paper
account) — past candles arrive one at a time, exactly like the live
zoo receives them, with the full past window as context.

Overfit guards (Bailey/Lopez de Prado):
  - only 2 thresholds tuned, in a 12-combination grid (capped trials)
  - purge gap of 200 bars between train and validation spans
  - selection objective on TRAIN only; validation is one-shot

Usage:
  python run_expert_train.py [--pairs BTC-USDC,...] [--db data/sim.db]
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
from bot.strategies import REGISTRY  # noqa: E402
from bot.strategies.chassis import ChassisStrategy  # noqa: E402
from bot.strategies.sage import SageStrategy  # noqa: E402
from bot.trade_gate import set_fee_model  # noqa: E402

TRAIN_FRAC = 0.70
PURGE_BARS = 200
REPLAY_WINDOW = 1200      # bars of context per decision (live parity)
GRID = {"buy_score": [1.5, 2.0, 2.5, 3.0],
        "sell_score": [-1.0, 0.0, 1.0]}   # 12 trials total


def replay_live(df: pd.DataFrame, strat, pair: str, cfg) -> PaperAccount:
    """THE honest validation: stream past candles one at a time through
    the real execute() path (chassis + gates + account). Bar i sees
    df[..i] only — decisions identical to what the live zoo would have
    made with the same history."""
    acc = PaperAccount(capital=cfg.paper_capital, taker_fee=cfg.taker_fee,
                       slippage=cfg.slippage,
                       position_fraction=cfg.position_fraction,
                       max_positions=cfg.max_positions,
                       cash_yield_apy=cfg.cash_yield_apy)
    closes = df["close"]
    warm = max(strat.warmup_bars(), 250)
    for i in range(warm, len(df)):
        lo = max(0, i - REPLAY_WINDOW + 1)
        window = df.iloc[lo:i + 1]
        ts = int(df.index[i].timestamp())
        try:
            strat.execute(acc, pair, window, float(closes.iloc[i]), ts)
        except Exception as exc:  # noqa: BLE001 — one bad bar never kills
            print(f"  [replay] bar {i} error: {exc}")
    return acc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/sim.db")
    args = ap.parse_args()

    cfg = BotConfig.from_yaml()
    set_fee_model(cfg.taker_fee, cfg.slippage)
    store = Store(args.db)

    data = {}
    for pair in cfg.pairs:
        df = store.load_candles(pair, cfg.granularity)
        if df is not None and len(df) > 2000:
            data[pair] = df.dropna()
    if not data:
        sys.exit("[expert] need >=2000 bars/pair in the db (run download)")
    print(f"[expert] {len(data)} pairs, "
          f"{min(len(d) for d in data.values())}..{max(len(d) for d in data.values())} bars "
          f"({cfg.granularity})")

    # ---- span split (per pair, with purge gap) ------------------------
    spans = {}
    for pair, df in data.items():
        n = len(df)
        cut = int(n * TRAIN_FRAC)
        spans[pair] = (df.index[:cut - PURGE_BARS], df.index[cut:])

    # ---- capped grid search on TRAIN only ------------------------------
    trials = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    print(f"[expert] {len(trials)} trials (capped) on TRAIN span")
    scored = []
    for t_i, combo in enumerate(trials):
        sharpes, rets = [], []
        for pair, df in data.items():
            tr = df.loc[spans[pair][0]]
            if len(tr) < 400:
                continue
            strat = ChassisStrategy(SageStrategy(dict(combo)))
            r = run_backtest(tr, strat, pair=pair, taker_fee=cfg.taker_fee,
                             slippage=cfg.slippage,
                             position_fraction=cfg.position_fraction,
                             capital=cfg.paper_capital,
                             cash_yield_apy=cfg.cash_yield_apy)
            sharpes.append(r.sharpe)
            rets.append(r.total_return)
        if sharpes:
            scored.append((float(np.mean(sharpes)), float(np.mean(rets)), combo))
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    for s, ret, combo in scored[:5]:
        print(f"  train sharpe={s:+.2f} ret={ret * 100:+.1f}% {combo}")

    best = scored[0][2]
    print(f"\n[expert] WINNER (train): {best}")
    print("[expert] validating on untouched span via LIVE-PATH replay...")

    # ---- one-shot live-path validation ---------------------------------
    rows = []
    for pair, df in data.items():
        va = df.loc[spans[pair][1]]
        if len(va) < 300:
            continue
        strat = ChassisStrategy(SageStrategy(dict(best)))
        acc = replay_live(va, strat, pair, cfg)
        price_now = float(va["close"].iloc[-1])
        eq = acc.equity({pair: price_now})
        bh = price_now / float(va["open"].iloc[0]) - 1.0
        ret = eq / cfg.paper_capital - 1.0
        rows.append({"pair": pair, "return_pct": round(ret * 100, 2),
                     "buy_hold_pct": round(bh * 100, 2),
                     "excess_pct": round((ret - bh) * 100, 2),
                     "trades": acc.n_trades,
                     "fees": round(acc.fee_take, 2),
                     "realized": round(acc.realized_pnl, 2)})
        print(f"  {pair}: ret={ret * 100:+.1f}% bh={bh * 100:+.1f}% "
              f"excess={(ret - bh) * 100:+.1f}% trades={acc.n_trades} "
              f"fees=${acc.fee_take:.0f}")

    mean_ex = float(np.mean([r["excess_pct"] for r in rows])) if rows else 0.0
    tot_tr = sum(r["trades"] for r in rows)
    report = {
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "params": {**SageStrategy.DEFAULTS, **best},
        "grid": [s[2] for s in scored[:5]],
        "validation": rows,
        "mean_excess_pct": round(mean_ex, 2),
        "total_trades": tot_tr,
        "promotable": bool(rows and tot_tr >= 6 and mean_ex >= 0.5),
    }
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "expert-sage.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    print(f"\n[expert] mean OOS excess {mean_ex:+.2f}% | promotable: "
          f"{report['promotable']} -> reports/expert-sage.json")


if __name__ == "__main__":
    main()
