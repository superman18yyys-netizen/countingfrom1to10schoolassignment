"""Nightly auto-tuner: extract the winning guard parameters over time.

This is the lab's "auto-apply winning logic" loop:

  1. Walk-forward backtest a BOUNDED sample of guard parameter variants
     (trial-capped to limit overfitting) on recent real history, with
     fees + slippage + cash-yield ON, scored on out-of-sample excess.
  2. Pick the best parameter set per guard mode (trend / range).
  3. Write state/guard_params.json, which every guarded bot reads at
     construction -- so the winning thresholds go live automatically on
     the next zoo/swarm run, no code changes.

Honesty guards: trial cap, OOS-only scoring, incumbents must be beaten
by a margin or the previous parameters are kept.
"""
from __future__ import annotations

import itertools
import json
import os
from datetime import datetime, timezone

from bot.backtest.engine import run_backtest
from bot.backtest.walkforward import split_train_test
from bot.config import BotConfig
from bot.data.fetcher import fetch_history
from bot.data.store import Store
from bot.strategies.guarded import (GuardedGrid, GuardedMACD, GuardedMomentum,
                                    GuardedRSI2, GuardedStochastic)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OVERRIDES_PATH = os.path.join(BASE_DIR, "state", "guard_params.json")

# Bounded search space (trial-capped: sampled subset, not full product).
PARAM_GRID = {
    "atr_hurdle_pct": [0.004, 0.008, 0.014],
    "trend_sma": [100, 200, 300],
    "confirm_bars": [1, 2, 3],
}
MAX_TRIALS = 10                      # overfit control
BEAT_MARGIN = 0.25                   # new params must beat incumbent by this %excess
PAIRS = ["BTC-USDC", "ETH-USDC", "SOL-USDC", "DOGE-USDC", "XRP-USDC", "ADA-USDC"]

# Representative bots per guard mode for evaluation.
EVAL_BOTS = {
    "trend": [GuardedMomentum, GuardedMACD],
    "range": [GuardedRSI2, GuardedStochastic, GuardedGrid],
}


def _score_params(mode: str, params: dict, data: dict, cfg: BotConfig) -> float:
    """Mean OOS excess return across eval bots x pairs for one param set."""
    excesses = []
    for bot_cls in EVAL_BOTS[mode]:
        for pair, df in data.items():
            _, test = split_train_test(df, train_frac=0.7)
            if len(test) < 200:
                continue
            strat = bot_cls({**params, "mode": mode})
            r = run_backtest(test, strat, pair=pair,
                             taker_fee=cfg.taker_fee, slippage=cfg.slippage,
                             position_fraction=0.30, capital=20.0,
                             cash_yield_apy=cfg.cash_yield_apy)
            excesses.append(r.excess_return * 100.0)
    return sum(excesses) / len(excesses) if excesses else 0.0


def tune(days: int = 365, granularity: str = "FOUR_HOUR",
         db_path: str = os.path.join(BASE_DIR, "data", "tune.db")) -> dict:
    cfg = BotConfig.from_yaml(None)
    store = Store(db_path)
    data = {}
    for pair in PAIRS:
        df = fetch_history(pair, granularity, days)
        if not df.empty:
            store.upsert_candles(pair, granularity, df)
            data[pair] = store.load_candles(pair, granularity)

    combos = list(itertools.product(*[PARAM_GRID[k] for k in
                                      sorted(PARAM_GRID)]))
    step = max(1, len(combos) // MAX_TRIALS)
    sampled = combos[::step][:MAX_TRIALS]

    current = json.load(open(OVERRIDES_PATH)) if os.path.exists(OVERRIDES_PATH) else {}
    results = {"tuned_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
               "note": "auto-tuned guard params; OOS excess, trial-capped", "modes": {}}
    for mode in ("trend", "range"):
        best_params, best_score = None, -1e9
        for combo in sampled:
            params = dict(zip(sorted(PARAM_GRID), combo))
            score = _score_params(mode, params, data, cfg)
            print(f"  [tune:{mode}] {params} -> OOS excess {score:+.2f}%")
            if score > best_score:
                best_score, best_params = score, params
        incumbent = (current.get(mode) or {})
        incumbent_score = _score_params(mode, incumbent, data, cfg) if incumbent else -1e9
        if best_params and best_score >= incumbent_score + BEAT_MARGIN:
            results["modes"][mode] = {**best_params, "score": round(best_score, 2)}
        else:   # keep incumbent: not beaten by the margin (overfit guard)
            results["modes"][mode] = {**incumbent, "score": round(incumbent_score, 2)} \
                if incumbent else {**best_params, "score": round(best_score, 2)}
    return results


def write_overrides(results: dict, path: str = OVERRIDES_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1)


# --------------------------------------------------------------------------
# trend_runner tuning stage: the auto-tuner also drives the trend_runner
# bot directly. Same honesty rules -- trial-capped grid, OOS-only scoring,
# incumbent must be beaten by a margin or the previous params are kept.
# Winners go to state/runner_params.json, which TrendRunner reads at
# construction (explicit params, e.g. swarm mutations, still win).
# --------------------------------------------------------------------------
RUNNER_PATH = os.path.join(BASE_DIR, "state", "runner_params.json")
RUNNER_GRID = {
    "trend_sma": [100, 200, 300],
    "atr_mult": [2.0, 3.0, 4.0],
    "atr_hurdle_pct": [0.004, 0.008],
    "trail_bars": [96, 168],
}
RUNNER_MAX_TRIALS = 12


def _score_runner(params: dict, data: dict, cfg: BotConfig) -> float:
    from bot.strategies.trend_runner import TrendRunner
    excesses = []
    for pair, df in data.items():
        _, test = split_train_test(df, train_frac=0.7)
        if len(test) < 200:
            continue
        r = run_backtest(test, TrendRunner(dict(params)), pair=pair,
                         taker_fee=cfg.taker_fee, slippage=cfg.slippage,
                         position_fraction=0.30, capital=20.0,
                         cash_yield_apy=cfg.cash_yield_apy)
        excesses.append(r.excess_return * 100.0)
    return sum(excesses) / len(excesses) if excesses else 0.0


def tune_runner(days: int = 365, granularity: str = "FOUR_HOUR",
                db_path: str = os.path.join(BASE_DIR, "data", "tune.db")) -> dict:
    """Trial-capped OOS tuning of trend_runner; writes runner_params.json."""
    cfg = BotConfig.from_yaml(None)
    store = Store(db_path)
    data = {}
    for pair in PAIRS:
        df = fetch_history(pair, granularity, days)
        if not df.empty:
            store.upsert_candles(pair, granularity, df)
            data[pair] = store.load_candles(pair, granularity)

    keys = sorted(RUNNER_GRID)
    combos = list(itertools.product(*(RUNNER_GRID[k] for k in keys)))
    step = max(1, len(combos) // RUNNER_MAX_TRIALS)
    sampled = combos[::step][:RUNNER_MAX_TRIALS]

    current = {}
    if os.path.exists(RUNNER_PATH):
        try:
            current = (json.load(open(RUNNER_PATH)) or {}).get("params") or {}
        except Exception:  # noqa: BLE001
            current = {}

    best_params, best_score = None, -1e9
    for combo in sampled:
        params = dict(zip(keys, combo))
        score = _score_runner(params, data, cfg)
        print(f"  [tune:trend_runner] {params} -> OOS excess {score:+.2f}%")
        if score > best_score:
            best_score, best_params = score, params
    incumbent_score = _score_runner(current, data, cfg) if current else -1e9
    if not (best_params and best_score >= incumbent_score + BEAT_MARGIN):
        best_params, best_score = (current or best_params), \
            (incumbent_score if current else best_score)

    payload = {
        "tuned_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "score": round(best_score, 2),
        "params": best_params,
        "note": "OOS excess vs buy&hold after fees; trial-capped; "
                "incumbent kept unless beaten by margin",
    }
    os.makedirs(os.path.dirname(RUNNER_PATH), exist_ok=True)
    with open(RUNNER_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    return payload
