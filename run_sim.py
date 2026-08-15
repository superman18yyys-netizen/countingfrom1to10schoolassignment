#!/usr/bin/env python3
"""Simulation tournament: race every zoo bot on historical windows.

The lab twin of the live repo. Runs the SAME strategies through the
SAME chassis on REAL historical data across a series of windows
(e.g. every 7-day slice of the last 3 months), then ranks bots by
aggregate excess return and exports winners as a champion file the
live repo can promote.

Examples:
  # 1-week windows over the last 90 days (default)
  python run_sim.py

  # explicit range and window
  python run_sim.py --start 2026-05-01 --end 2026-08-01 --window 7 --step 7

  # also evolve param variants (random genomes from PARAM_BOUNDS)
  python run_sim.py --variants 40
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bot.backtest.engine import run_backtest  # noqa: E402
from bot.config import BotConfig  # noqa: E402
from bot.data.fetcher import fetch_candles  # noqa: E402
from bot.data.store import Store  # noqa: E402
from bot.strategies import REGISTRY  # noqa: E402
from bot.strategies.chassis import ChassisStrategy  # noqa: E402
from bot.swarm.genome import PARAM_BOUNDS, make_genome  # noqa: E402
from bot.trade_gate import set_fee_model  # noqa: E402
from bot.zoo.roster import ROSTER  # noqa: E402

MIN_TRADES_TOTAL = 5     # a winner must actually trade
CHAMPION_THRESHOLD = 0.5  # % excess to be promotable


def load_data(cfg: BotConfig, start: datetime, end: datetime,
              db_path: str) -> dict:
    store = Store(db_path)
    out = {}
    for pair in cfg.pairs:
        df = store.load_candles(pair, cfg.granularity,
                                start=int(start.timestamp()),
                                end=int(end.timestamp()))
        if df is None or len(df) < 300:
            try:
                df = fetch_candles(pair, cfg.granularity, start, end)
                if not df.empty:
                    store.upsert_candles(pair, cfg.granularity, df)
            except Exception as exc:  # noqa: BLE001
                print(f"[sim] {pair} unavailable: {exc}")
                continue
        if df is not None and not df.empty:
            out[pair] = df.dropna()
    return out


def make_contestants(variants: int, seed: int = 11) -> list:
    """(id, strategy_name, params) — every zoo bot + optional evolved
    variants sampled from the bounded param spaces."""
    out = [(bid, sname, dict(params)) for bid, sname, params, _ in ROSTER]
    if variants > 0:
        import random
        rng = random.Random(seed)
        seen = set()
        for i in range(variants):
            sname = rng.choice(sorted(PARAM_BOUNDS.keys()))
            g = make_genome(sname, f"var-{i:03d}", rng)
            key = (sname, json.dumps(g.params, sort_keys=True))
            if key in seen:
                continue
            seen.add(key)
            out.append((g.id, sname, dict(g.params)))
    return out


LOOKBACK_BARS = 700   # warmup padding before each window (SMA200, 90d
                      # ATR percentile, 1y context all need history)


def run_tournament(cfg, data, contestants, windows) -> list:
    """One full backtest per (contestant, pair) over the whole range
    (indicators stay warm), then per-window attribution by slicing the
    equity curve — realistic position carry between windows."""
    results = {cid: {"excess": [], "dd": [], "sharpe": [], "trades": 0,
                     "strategy": sname, "params": params}
               for cid, sname, params in contestants}
    w0, w1 = windows[0][0], windows[-1][1]
    for pair, df in data.items():
        print(f"[sim] backtesting {pair} ({len(df)} bars)", flush=True)
        df = df.copy()
        df.attrs["pair"] = pair
        close = df["close"]
        for cid, sname, params in contestants:
            if sname not in REGISTRY:
                continue
            base = REGISTRY[sname](dict(params))
            strat = ChassisStrategy(base)
            r = run_backtest(df, strat, pair=pair,
                             taker_fee=cfg.taker_fee,
                             slippage=cfg.slippage,
                             position_fraction=cfg.position_fraction,
                             capital=cfg.paper_capital,
                             cash_yield_apy=cfg.cash_yield_apy)
            eq = r.equity_curve
            agg = results[cid]
            for a, b in windows:
                a0 = eq.index[eq.index < a]
                a1 = eq.index[eq.index <= b]
                if len(a0) == 0 or len(a1) == 0 or a0[-1] == a1[-1]:
                    continue
                s_ret = eq.loc[a1[-1]] / eq.loc[a0[-1]] - 1.0
                c0 = close.index[close.index < a]
                c1 = close.index[close.index <= b]
                if len(c0) == 0 or len(c1) == 0:
                    continue
                bh = close.loc[c1[-1]] / close.loc[c0[-1]] - 1.0
                agg["excess"].append(s_ret - bh)
                sl = eq.loc[a0[-1]:a1[-1]]
                peak = sl.cummax()
                dd = ((sl - peak) / peak).min()
                agg["dd"].append(float(dd) if dd == dd else 0.0)
                rets = sl.pct_change().dropna()
                if len(rets) > 2 and rets.std() > 0:
                    agg["sharpe"].append(float(rets.mean() / rets.std()))
                n = sum(1 for t in r.trades if a < t.exit_ts <= b)
                agg["trades"] += n
    return results


def summarize(results) -> list:
    rows = []
    for cid, r in results.items():
        if not r["excess"]:
            continue
        rows.append({
            "id": cid, "strategy": r["strategy"], "params": r["params"],
            "trades": r["trades"],
            "mean_excess_pct": round(100 * float(np.mean(r["excess"])), 2),
            "median_excess_pct": round(100 * float(np.median(r["excess"])), 2),
            "worst_excess_pct": round(100 * float(np.min(r["excess"])), 2),
            "mean_dd_pct": round(100 * float(np.mean(r["dd"])), 2),
            "mean_sharpe": round(float(np.mean(r["sharpe"])), 3),
            "eligible": r["trades"] >= MIN_TRADES_TOTAL,
        })
    rows.sort(key=lambda x: (x["eligible"], x["mean_excess_pct"],
                             x["median_excess_pct"]), reverse=True)
    return rows


def write_outputs(rows, windows, out_dir) -> None:
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    # markdown board
    lines = [f"# Simulation tournament {stamp}", "",
             f"windows: {windows[0][0]:%Y-%m-%d} .. {windows[-1][1]:%Y-%m-%d} "
             f"({len(windows)} windows)",
             "", "| rank | bot | strategy | trades | mean excess% | median% | worst% | mean dd% | sharpe | eligible |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        lines.append(
            f"| {i} | {r['id']} | {r['strategy']} | {r['trades']} | "
            f"{r['mean_excess_pct']:+.1f} | {r['median_excess_pct']:+.1f} | "
            f"{r['worst_excess_pct']:+.1f} | {r['mean_dd_pct']:.1f} | "
            f"{r['mean_sharpe']:.2f} | {'Y' if r['eligible'] else '-'} |")
    board = os.path.join(out_dir, f"tournament-{stamp}.md")
    with open(board, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    # full results + champion candidates for live-repo promotion
    full = os.path.join(out_dir, f"tournament-{stamp}.json")
    with open(full, "w", encoding="utf-8") as fh:
        json.dump({"generated": stamp,
                   "windows": [[str(a), str(b)] for a, b in windows],
                   "rows": rows}, fh, indent=1)
    winners = [r for r in rows
               if r["eligible"] and r["mean_excess_pct"] >= CHAMPION_THRESHOLD]
    if winners:
        top = winners[0]
        champ = {"promotable": True,
                 "updated_at": datetime.now(timezone.utc).strftime(
                     "%Y-%m-%d %H:%M UTC"),
                 "threshold_excess_pct": CHAMPION_THRESHOLD,
                 "champion": {"strategy": top["strategy"],
                              "params": top["params"],
                              "excess_pct": top["mean_excess_pct"],
                              "eligible": True},
                 "runner_up": [{"strategy": w["strategy"], "params": w["params"],
                                "excess_pct": w["mean_excess_pct"]}
                               for w in winners[1:4]]}
        with open(os.path.join(out_dir, "champions.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(champ, fh, indent=1)
    print(f"[sim] board -> {board}")
    print("\n".join(lines[:14]))


def main() -> None:
    ap = argparse.ArgumentParser(description="Simulation tournament")
    ap.add_argument("--start", default=None, help="YYYY-MM-DD (default: 90d ago)")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD (default: now)")
    ap.add_argument("--window", type=int, default=7, help="window days")
    ap.add_argument("--step", type=int, default=None,
                    help="step days (default: = window)")
    ap.add_argument("--variants", type=int, default=0,
                    help="random param variants to also race")
    ap.add_argument("--db", default="data/sim.db")
    args = ap.parse_args()

    cfg = BotConfig.from_yaml()
    set_fee_model(cfg.taker_fee, cfg.slippage)
    end = (datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
           if args.end else datetime.now(timezone.utc))
    start = (datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
             if args.start else end - timedelta(days=90))
    step = timedelta(days=args.step or args.window)
    windows, s = [], start
    while s + timedelta(days=args.window) <= end:
        windows.append((s, s + timedelta(days=args.window)))
        s += step
    if not windows:
        ap.error("no complete windows in range")

    print(f"[sim] loading {cfg.granularity} data "
          f"{start - timedelta(days=120):%Y-%m-%d}..{end:%Y-%m-%d} "
          f"(with warmup lookback)")
    data = load_data(cfg, start - timedelta(days=120), end, args.db)
    if not data:
        sys.exit("[sim] no data available")
    print(f"[sim] {len(data)} pairs, {sum(len(d) for d in data.values())} bars")

    contestants = make_contestants(args.variants)
    print(f"[sim] {len(contestants)} contestants, {len(windows)} windows")
    results = run_tournament(cfg, data, contestants, windows)
    rows = summarize(results)
    write_outputs(rows, windows, cfg.out_dir)


if __name__ == "__main__":
    main()
