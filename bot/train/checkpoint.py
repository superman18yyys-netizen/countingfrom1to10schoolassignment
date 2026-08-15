"""Resumable progress ledger for the training pipeline.

A single small JSON file (committed to the repo in ``state/training/``)
records exactly what has been completed so a subsequent time-budgeted
GitHub Actions job can continue where the previous one stopped. Heavy
artifacts (candles DB, feature parquet, model files) live in the Actions
cache under ``data/train_cache/`` and are addressed by an opaque
``config_hash`` stored here.

If the training configuration (pairs, granularity, days, feature version,
hyperparameters) changes, ``config_hash`` changes and the ledger is reset
so the pipeline rebuilds from scratch in a fresh cache directory.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

SCHEMA_VERSION = 1
DEFAULT_PATH = ("state/training/checkpoint.json")


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


class Checkpoint:
    def __init__(self, config_hash: str, config: dict):
        self.config_hash = config_hash
        self.config = config
        self.data_done: dict[str, dict] = {}        # pair -> {rows, latest_ts}
        self.features_built: list[str] = []
        self.cv_done: list[dict] = []               # [{config_idx, pair, fold}]
        self.cv_results: list[dict] = []            # aggregated per completed unit
        self.best_config: dict | None = None
        self.holdout_done: bool = False
        self.deployed: dict | None = None
        self.last_started = utcnow()
        self.last_finished = ""

    # ----------------------------------------------------------------- data
    def pair_fetched(self, pair: str) -> bool:
        return pair in self.data_done

    def mark_pair_fetched(self, pair: str, rows: int, latest_ts: int) -> None:
        self.data_done[pair] = {"rows": rows, "latest_ts": latest_ts}

    # -------------------------------------------------------------- features
    def features_done(self, pair: str) -> bool:
        return pair in self.features_built

    def mark_features_done(self, pair: str) -> None:
        if pair not in self.features_built:
            self.features_built.append(pair)

    # -------------------------------------------------------------------- cv
    def cv_key(self, config_idx: int, pair: str, fold: int) -> tuple:
        return (config_idx, pair, fold)

    def cv_done_key(self, config_idx: int, pair: str, fold: int) -> bool:
        return any(c["config_idx"] == config_idx and c["pair"] == pair
                   and c["fold"] == fold for c in self.cv_done)

    def mark_cv_done(self, config_idx: int, pair: str, fold: int,
                     result: dict | None = None) -> None:
        entry = {"config_idx": config_idx, "pair": pair, "fold": fold}
        self.cv_done.append(entry)
        if result is not None:
            entry["result"] = result
            self.cv_results.append({**entry, "result": result})

    # ------------------------------------------------------------- deployed
    def touched_at(self) -> str:
        return utcnow()

    def save(self, path: str = DEFAULT_PATH, cache_dir: str | None = None) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "config_hash": self.config_hash,
            "config": self.config,
            "cache_dir": cache_dir,
            "data_done": self.data_done,
            "features_built": self.features_built,
            "cv_done": self.cv_done,
            "cv_results": self.cv_results,
            "best_config": self.best_config,
            "holdout_done": self.holdout_done,
            "deployed": self.deployed,
            "last_started": self.last_started,
            "last_finished": utcnow(),
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1)


def load_checkpoint(path: str = DEFAULT_PATH,
                    config_hash: str | None = None) -> Checkpoint | None:
    """Load the ledger if it exists and matches the current config hash."""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if payload.get("schema_version") != SCHEMA_VERSION:
        return None
    if config_hash is not None and payload.get("config_hash") != config_hash:
        return None   # config changed -> caller rebuilds from scratch
    cp = Checkpoint(payload["config_hash"], payload["config"])
    cp.data_done = payload.get("data_done", {})
    cp.features_built = payload.get("features_built", [])
    cp.cv_done = payload.get("cv_done", [])
    cp.cv_results = payload.get("cv_results", [])
    cp.best_config = payload.get("best_config")
    cp.holdout_done = payload.get("holdout_done", False)
    cp.deployed = payload.get("deployed")
    cp.last_started = payload.get("last_started", "")
    cp.last_finished = payload.get("last_finished", "")
    return cp