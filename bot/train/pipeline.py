"""Resumable, time-budgeted training controller.

Orchestrates the full training loop that runs as a chain of GitHub
Actions jobs:

    data -> features (in-memory) -> walk-forward CV -> final fit + holdout gate

Persistence split:
  * candles DB          -> data/train_cache/ (restored from Actions cache)
  * CV fold results     -> checkpoint JSON (committed to git)
  * promoted config     -> state/deployed_model.json (committed to git)

Every phase checks the wall-clock deadline and stops cleanly with margin
so the caller can commit the checkpoint and re-dispatch the next job.
Finished CV folds (config, pair, fold) are skipped on resume.

The gate is always evaluated on the *current* trailing holdout, and the
incumbent config is re-fit on that same holdout for a fair comparison.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from bot.backtest.engine import run_backtest
from bot.config import GRANULARITY_SECONDS
from bot.data.fetcher import fetch_candles
from bot.data.store import Store
from bot.strategies.ml_trend import MLTrendStrategy
from bot.train.checkpoint import Checkpoint, load_checkpoint, utcnow
from bot.train.features import FEATURES_VERSION, build_features

HOLD_FRAC = 0.20
MIN_TRADES_GATE = 8
MIN_PAIRS = 2
LEAD_BARS = 6
DEFAULT_GRID = [
    {"horizon": 6,  "min_gain": 0.013, "regime_horizon": 24, "regime_tol": 0.004,
     "regime_up": 0.50, "buy": 0.60, "exit": 0.50},
    {"horizon": 6,  "min_gain": 0.011, "regime_horizon": 24, "regime_tol": 0.008,
     "regime_up": 0.50, "buy": 0.55, "exit": 0.50},
    {"horizon": 12, "min_gain": 0.013, "regime_horizon": 48, "regime_tol": 0.004,
     "regime_up": 0.50, "buy": 0.60, "exit": 0.50},
    {"horizon": 12, "min_gain": 0.011, "regime_horizon": 48, "regime_tol": 0.008,
     "regime_up": 0.55, "buy": 0.55, "exit": 0.45},
    {"horizon": 6,  "min_gain": 0.015, "regime_horizon": 24, "regime_tol": 0.006,
     "regime_up": 0.55, "buy": 0.60, "exit": 0.50},
]


def _hash(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


@dataclass
class TrainConfig:
    pairs: list = field(default_factory=lambda: ["BTC-USDC", "ETH-USDC", "SOL-USDC",
                                                 "DOGE-USDC", "XRP-USDC", "ADA-USDC"])
    granularity: str = "ONE_HOUR"
    days: int = 1460
    capital: float = 20.0
    taker_fee: float = 0.006
    slippage: float = 0.001
    position_fraction: float = 0.30
    cash_yield_apy: float = 0.045
    lead_bars: int = 6
    n_folds: int = 6
    grid: list = field(default_factory=list)
    db_path: str = "data/train_cache/data.db"
    art_dir: str = "data/train_cache/art"
    checkpoint_path: str = "state/training/checkpoint.json"
    deployed_path: str = "state/deployed_model.json"

    def __post_init__(self):
        if not self.grid:
            self.grid = list(DEFAULT_GRID)

    @property
    def universe_hash(self) -> str:
        return _hash([self.pairs, self.granularity, self.days, FEATURES_VERSION])

    @property
    def config_hash(self) -> str:
        return _hash([self.pairs, self.granularity, self.days, FEATURES_VERSION, self.grid])

    def marker(self) -> dict:
        return {"pairs": self.pairs, "granularity": self.granularity, "days": self.days}


class TrainingRun:
    def __init__(self, cfg: TrainConfig, budget_sec: float,
                 stop_margin_sec: float = 120.0):
        self.cfg = cfg
        self.deadline = time.monotonic() + budget_sec
        self.stop_margin = stop_margin_sec
        os.makedirs(os.path.dirname(cfg.db_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(cfg.checkpoint_path) or ".", exist_ok=True)
        self.store = Store(cfg.db_path)
        self.cp = load_checkpoint(cfg.checkpoint_path, cfg.config_hash) \
            or Checkpoint(cfg.config_hash, cfg.marker())
        self.cp.config_hash = cfg.config_hash
        self._done = False

    # -------------------------------------------------------------- budget
    def _remaining(self) -> float:
        return self.deadline - time.monotonic()

    def _out_of_time(self) -> bool:
        return self._remaining() < self.stop_margin

    def _save(self) -> None:
        self.cp.save(self.cfg.checkpoint_path, cache_dir=os.path.dirname(self.cfg.db_path))

    # ----------------------------------------------------------------- data
    def _data(self) -> None:
        bar_sec = GRANULARITY_SECONDS[self.cfg.granularity]
        now = int(time.time())
        now -= now % bar_sec
        for pair in self.cfg.pairs:
            if self._out_of_time():
                print(f"    [data] budget reached -> defer {pair}")
                continue
            marker = self.cp.data_done.get(pair)
            if marker is None:
                start = now - self.cfg.days * 86400
                print(f"    [data] full fetch {pair} ({self.cfg.days}d {self.cfg.granularity})")
            else:
                start = int(marker["latest_ts"]) + bar_sec   # incremental refresh
                print(f"    [data] incremental fetch {pair} since ts {start}")
            df = fetch_candles(pair, self.cfg.granularity,
                               datetime.fromtimestamp(start, timezone.utc),
                               datetime.fromtimestamp(now, timezone.utc))
            if df is None or df.empty:
                continue
            self.store.upsert_candles(pair, self.cfg.granularity, df)
            rows = self.store.candle_count(pair, self.cfg.granularity)
            latest = int(df.index[-1].timestamp())
            self.cp.mark_pair_fetched(pair, rows, latest)
            print(f"    [data] {pair}: {rows} rows, latest {pd.Timestamp(latest, unit='s', tz='UTC'):%Y-%m-%d %H:%M}")

    # ------------------------------------------------------------------ cv
    def _folds(self, n: int, purge: int):
        end_cv = int(n * (1.0 - HOLD_FRAC))
        win = max(1, (end_cv - purge) // self.cfg.n_folds)
        for f in range(self.cfg.n_folds):
            te_lo = purge + f * win
            te_hi = te_lo + win
            if te_hi > end_cv:
                break
            yield f, (0, te_lo - purge, te_lo, te_hi)

    def _cv(self) -> None:
        print("  phase: walk-forward CV")
        for ci, hyper in enumerate(self.cfg.grid):
            if self._out_of_time():
                print(f"    [cv] budget reached -> defer config {ci}")
                break
            for pair in self.cfg.pairs:
                df = self.store.load_candles(pair, self.cfg.granularity)
                if len(df) < 600:
                    print(f"    [cv] {pair}: too little data ({len(df)})")
                    continue
                for fold, (lo, tr_hi, te_lo, te_hi) in self._folds(len(df), purge=24):
                    if self.cp.cv_done_key(ci, pair, fold):
                        continue
                    if self._out_of_time():
                        print(f"    [cv] budget reached -> defer {pair} fold {fold}")
                        self._save()
                        return
                    train = df.iloc[lo:tr_hi]
                    test = df.iloc[te_lo:te_hi]
                    strat = MLTrendStrategy(hyper)
                    try:
                        strat.fit(train)
                        if strat.bundle is None:
                            res = {"n_trades": 0, "excess%": 0.0, "sharpe": 0.0,
                                   "max_dd%": 0.0, "win%": 0.0, "fees$": 0.0}
                        else:
                            r = run_backtest(test, strat, pair=pair,
                                             taker_fee=self.cfg.taker_fee,
                                             slippage=self.cfg.slippage,
                                             position_fraction=self.cfg.position_fraction,
                                             capital=self.cfg.capital,
                                             cash_yield_apy=self.cfg.cash_yield_apy)
                            res = {"n_trades": int(r.n_trades),
                                   "excess%": round(float(r.excess_return) * 100, 2),
                                   "sharpe": round(float(r.sharpe), 3),
                                   "max_dd%": round(float(r.max_drawdown) * 100, 2),
                                   "win%": round(float(r.win_rate) * 100, 1),
                                   "fees$": round(float(r.fee_take), 2)}
                    except Exception as e:  # noqa: BLE001
                        res = {"n_trades": 0, "excess%": 0.0, "error": str(e)[:80]}
                    self.cp.mark_cv_done(ci, pair, fold, res)
                    self._save()
                    print(f"    [cv] cfg{ci} {pair} fold{fold}: {res}")
        self._pick_best()

    def _aggregate(self, ci: int) -> dict | None:
        rows = [r["result"] for r in self.cp.cv_results if r["config_idx"] == ci]
        valid = [r for r in rows if r and r.get("n_trades", 0) > 0]
        if not valid:
            return None
        n_tr = sum(r["n_trades"] for r in valid)
        if n_tr < MIN_TRADES_GATE:
            return None
        per_pair = {}
        for r in self.cp.cv_results:
            if r["config_idx"] == ci and r["result"] and r["result"].get("n_trades", 0) > 0:
                per_pair[r["pair"]] = True
        if len(per_pair) < MIN_PAIRS:
            return None
        return {
            "config_idx": ci,
            "trades": n_tr,
            "pairs_covered": len(per_pair),
            "excess%": round(sum(r["excess%"] for r in valid) / len(valid), 3),
            "sharpe": round(sum(r["sharpe"] for r in valid) / len(valid), 3),
            "max_dd%": round(max(r["max_dd%"] for r in valid), 2),
            "win%": round(sum(r["win%"] for r in valid) / len(valid), 2),
            "fees$": round(sum(r["fees$"] for r in valid), 2),
            "units": len(valid),
        }

    def _pick_best(self) -> None:
        best = None
        best_score = -1e18
        for ci in range(len(self.cfg.grid)):
            agg = self._aggregate(ci)
            if not agg:
                continue
            # reject configs that don't clear transaction costs in aggregate
            if agg["excess%"] <= 0:
                continue
            if agg["excess%"] * self.cfg.capital < agg["fees$"]:
                continue
            if agg["excess%"] > best_score:
                best_score = agg["excess%"]
                agg["hyper"] = self.cfg.grid[ci]
                best = agg
        self.cp.best_config = best

    # --------------------------------------------------------- final + gate
    def _final(self) -> None:
        best = self.cp.best_config
        if best is None:
            print("  phase: final+gate -> no viable config (CV not done / all lost to fees)")
            return
        hyper = best["hyper"]
        print(f"  phase: final fit + holdout gate (best cfg{best['config_idx']} "
              f"excess {best['excess%']}%)")
        fee_metrics = {"taker_fee": self.cfg.taker_fee,
                       "slippage": self.cfg.slippage,
                       "position_fraction": self.cfg.position_fraction,
                       "capital": self.cfg.capital,
                       "cash_yield_apy": self.cfg.cash_yield_apy}
        new_holdout, incumbent_holdout = self._evaluate_holdout(hyper,
                                                                self.cp.deployed.get("hyper") if self.cp.deployed else None,
                                                                fee_metrics)
        self.cp.holdout_done = True
        self._promote(new_holdout, incumbent_holdout, hyper, fee_metrics)

    def _evaluate_holdout(self, hyper, incumbent_hyper, fee_metrics):
        total_rows = {p: len(self.store.load_candles(p, self.cfg.granularity))
                      for p in self.cfg.pairs}
        new = self._score_holdout(hyper, fee_metrics)
        inc = None
        if incumbent_hyper is not None:
            inc = self._score_holdout(incumbent_hyper, fee_metrics)
        return new, inc

    def _score_holdout(self, hyper, fee_metrics) -> dict:
        from bot.train.features import build_features
        lead = self.cfg.lead_bars
        per_pair = {}
        for pair in self.cfg.pairs:
            df = self.store.load_candles(pair, self.cfg.granularity)
            if len(df) < 600:
                continue
            cut = int(len(df) * (1.0 - HOLD_FRAC))
            train, test = df.iloc[:cut], df.iloc[cut:]
            strat = MLTrendStrategy(hyper)
            strat.fit(train)
            if strat.bundle is None:
                per_pair[pair] = {"n_trades": 0, "excess%": 0.0, "win%": 0.0,
                                  "sharpe": 0.0, "max_dd%": 0.0, "fees$": 0.0,
                                  "transition_auc_lead": 0.0, "regime_accuracy": 0.0}
                continue
            r = run_backtest(test, strat, pair=pair, **fee_metrics)
            # Transition prediction metric: compute regime + timing predictions
            # aligned to the test window and compare to actual forward regimes
            # ``lead`` bars ahead. This measures *forecasting* not just
            # identification.
            try:
                feats = build_features(test)
                cols = [c for c in feats.columns]
                reg_p = np.nan_to_num(strat.bundle.regime.predict(feats[cols]).to_numpy(dtype=float))
                # Actual regime lead bars ahead
                fwd_real = test["close"].shift(-lead) / test["close"] - 1.0
                fwd_real = fwd_real.iloc[:len(reg_p)]
                actual = np.zeros(len(fwd_real))
                tol = hyper.get("regime_tol", 0.004)
                actual[fwd_real > tol] = 2.0   # up
                actual[fwd_real < -tol] = 0.0   # down
                actual[abs(fwd_real) <= tol] = 1.0   # range
                # Transition prediction accuracy (up=2 vs down=0 vs range=1)
                pred_classes = np.where(reg_p >= 0.5, 2.0, 1.0)
                n_common = min(len(pred_classes), len(actual))
                if n_common > 0:
                    correct = np.mean(pred_classes[:n_common] == actual[:n_common])
                    transition_auc = round(float(correct), 4)
                else:
                    transition_auc = 0.0
                regime_accuracy = transition_auc
            except Exception:  # noqa: BLE001
                transition_auc = 0.0
                regime_accuracy = 0.0
            per_pair[pair] = {"n_trades": int(r.n_trades),
                              "excess%": round(float(r.excess_return) * 100, 2),
                              "win%": round(float(r.win_rate) * 100, 1),
                              "sharpe": round(float(r.sharpe), 3),
                              "max_dd%": round(float(r.max_drawdown) * 100, 2),
                              "fees$": round(float(r.fee_take), 2),
                              "transition_auc_lead": transition_auc,
                              "regime_accuracy": regime_accuracy,
                              "holdout_start": str(test.index[0]),
                              "holdout_end": str(test.index[-1]),
                              "holdout_rows": len(test)}
        valid = [v for v in per_pair.values() if v["n_trades"] > 0]
        n_tr = sum(v["n_trades"] for v in per_pair.values())
        transition_aucs = [v.get("transition_auc_lead", 0) for v in valid]
        return {
            "per_pair": per_pair,
            "total_trades": n_tr,
            "excess%": round(sum(v["excess%"] for v in valid) / len(valid), 3) if valid else 0.0,
            "win%": round(sum(v["win%"] for v in valid) / len(valid), 2) if valid else 0.0,
            "fees$": round(sum(v["fees$"] for v in per_pair.values()), 2),
            "n_pairs_traded": len(valid),
            "transition_auc_lead%": round(float(np.mean(transition_aucs) * 100), 2) if transition_aucs else 0.0,
        }

    def _promote(self, new, incumbent, hyper, fee_metrics) -> None:
        os.makedirs(os.path.dirname(self.cfg.deployed_path) or ".", exist_ok=True)
        incumbent = incumbent or {}
        inc_excess = incumbent.get("excess%", -1e18) if incumbent.get("excess%") is not None else -1e18
        meets_bars = (new["total_trades"] >= MIN_TRADES_GATE
                      and new["n_pairs_traded"] >= MIN_PAIRS
                      and new["excess%"] > 0.0)
        beats_incumbent = new["excess%"] > inc_excess
        if meets_bars and beats_incumbent:
            champion = {"hyper": hyper,
                        "universe_hash": self.cfg.universe_hash,
                        "config_hash": self.cfg.config_hash,
                        "pairs": self.cfg.pairs,
                        "granularity": self.cfg.granularity,
                        "days": self.cfg.days,
                        "features_version": FEATURES_VERSION,
                        "trained_at": utcnow(),
                        "pre_registered": True,
                        "deployment_source": "ml_regime_gate",
                        "metrics_holdout": new,
                        "fees": {"taker_fee": self.cfg.taker_fee,
                                 "slippage": self.cfg.slippage,
                                 "cash_yield_apy": self.cfg.cash_yield_apy}}
            self.cp.deployed = champion
            with open(self.cfg.deployed_path, "w", encoding="utf-8") as fh:
                json.dump(champion, fh, indent=1)
            promoted = True
        else:
            promoted = False
        inc_str = f"{inc_excess:+.2f}%" if incumbent.get("excess%") is not None else "n/a"
        print(f"    [gate] new{'+' if promoted else ''} excess {new['excess%']}% "
              f"({new['total_trades']} tr) vs incumbent {inc_str}"
              f"{' -> DEPLOYED' if promoted else ''}")

    # ----------------------------------------------------------------- run
    def run(self) -> int:
        self.cp.last_started = utcnow()
        print(f"[train] config hash {self.cfg.config_hash} | pairs {self.cfg.pairs} "
              f"| {self.cfg.granularity} | {self.cfg.days}d | {len(self.cfg.grid)} configs "
              f"| budget {max(0, int(self._remaining() - self.stop_margin))}s")
        self._data()
        self._cv()
        if self._remaining() > self.stop_margin:
            self._final()
        else:
            print("  phase: final+gate deferred to a later job (budget exhausted)")
        self.cp.last_finished = utcnow()
        self._save()
        if self.cp.best_config is not None and self.cp.deployed is not None:
            self._done = True
        # A budget-exhausted run is a normal, resumable stop -- NEVER a
        # failure. Returning 0 keeps the workflow's commit + self-chain
        # steps running so the next job continues where this one left off.
        print("[train] chunk complete; checkpoint saved. "
              + ("(resumable -- a continuation job will pick up any remaining work)"
                 if self._out_of_time() else "(pipeline complete)"))
        return 0

    @property
    def done(self) -> bool:
        return self._done

    @property
    def out_of_time(self) -> bool:
        return self._out_of_time()