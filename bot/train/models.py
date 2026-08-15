"""Two-stage trend model: a regime gate + a fee-aware timing model.

Stage 1 (regime) is a 3-class LightGBM that learns "is the market about
to be in a strong up / down / range state" from the causal feature
matrix. It decides *whether to be in the market at all* -- the regime
awareness is what lets the strategy sit out drawdowns.

Stage 2 (timing) is a binary LightGBM that predicts a *fee-aware* up/down
label (a move is "up" only if it clears round-trip fees). It decides
*where* within a favourable regime to take a long entry.

Both models are serialized as LightGBM text files plus a JSON meta blob
(a ModelBundle), so a trained bundle can be committed / cached and then
loaded by the deployable MLTrendStrategy.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:  # pragma: no cover
    lgb = None
    _HAS_LGB = False

from bot.train.features import FEATURE_COLS

REGIME_UP, REGIME_RANGE, REGIME_DOWN = 2, 1, 0

_LGB_BASE = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "verbose": -1,
    "n_jobs": 2,
}


def _dataset(X: pd.DataFrame, y: pd.Series, val_frac: float = 0.15):
    n = len(y)
    split = max(1, int(n * val_frac))
    xs = [c for c in FEATURE_COLS if c in X.columns]
    dtrain = lgb.Dataset(X[xs].iloc[:split], label=y.iloc[:split])
    dvalid = lgb.Dataset(X[xs].iloc[split:], label=y.iloc[split:], reference=dtrain)
    return dtrain, dvalid, xs


class RegimeClassifier:
    """3-class : up / range / down. Wrapped as a binary encoding of the
    tradable state (up) so predict_proba gives P(headed-up)."""

    def __init__(self, params=None):
        self.params = dict(_LGB_BASE)
        self.params.update(params or {})
        self.params["objective"] = "multiclass"
        self.params["num_class"] = 3
        self.params["metric"] = "multi_logloss"
        self._model = None
        self._cols = None
        self._best_iter = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        if not _HAS_LGB:
            raise RuntimeError("lightgbm is not installed (pip install lightgbm)")
        dtrain, dvalid, cols = _dataset(X, y)
        self._cols = cols
        model = lgb.train(
            self.params, dtrain, num_boost_round=300,
            valid_sets=[dvalid], callbacks=[lgb.early_stopping(30)],
        )
        self._model = model
        self._best_iter = model.best_iteration or None

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """P(class=up) for each row in [0,1]."""
        if self._model is None:
            return pd.Series(np.nan, index=X.index)
        proba = self._model.predict(X[self._cols], num_iteration=self._best_iter)
        return pd.Series(proba[:, REGIME_UP], index=X.index)

    def save(self, path: str) -> None:
        if self._model is not None:
            self._model.save_model(path)

    def load(self, path: str) -> None:
        self._model = lgb.Booster(model_file=path)
        self._best_iter = self._model.best_iteration or None
        self._cols = self._model.feature_name()


class TimingModel:
    """Fee-aware direction classifier (long/flat)."""

    def __init__(self, params=None):
        self.params = dict(_LGB_BASE)
        self.params.update(params or {})
        self._model = None
        self._cols = None
        self._best_iter = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        if not _HAS_LGB:
            raise RuntimeError("lightgbm is not installed (pip install lightgbm)")
        dtrain, dvalid, cols = _dataset(X, y)
        self._cols = cols
        model = lgb.train(
            self.params, dtrain, num_boost_round=400,
            valid_sets=[dvalid], callbacks=[lgb.early_stopping(30)],
        )
        self._model = model
        self._best_iter = model.best_iteration or None

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self._model is None:
            return pd.Series(np.nan, index=X.index)
        return pd.Series(self._model.predict(X[self._cols], num_iteration=self._best_iter),
                         index=X.index)

    def save(self, path: str) -> None:
        if self._model is not None:
            self._model.save_model(path)

    def load(self, path: str) -> None:
        self._model = lgb.Booster(model_file=path)
        self._best_iter = self._model.best_iteration or None
        self._cols = self._model.feature_name()


@dataclass
class ModelBundleMeta:
    horizon: int
    min_gain: float
    regime_horizon: int
    regime_tol: float
    buy_threshold: float
    exit_threshold: float
    features_version: str
    warmup_bars: int
    trained_at: str
    fit_pairs: list[str]
    fit_start: str
    fit_end: str


class ModelBundle:
    """Serializable pair of regime + timing models plus hyperparameters."""

    def __init__(self, regime: RegimeClassifier, timing: TimingModel,
                 meta: ModelBundleMeta):
        self.regime = regime
        self.timing = timing
        self.meta = meta

    @classmethod
    def empty(cls, hyper: dict) -> "ModelBundle":
        meta = ModelBundleMeta(**{f: hyper.get(f) for f in
                                  ModelBundleMeta.__dataclass_fields__},
                               features_version="?", trained_at="",
                               fit_pairs=[], fit_start="", fit_end="")
        return cls(RegimeClassifier(hyper.get("lgb_regime")),
                   TimingModel(hyper.get("lgb_timing")), meta)

    def save(self, directory: str) -> str:
        os.makedirs(directory, exist_ok=True)
        self.regime._model.save_model(os.path.join(directory, "regime.txt"))
        self.timing._model.save_model(os.path.join(directory, "timing.txt"))
        meta_path = os.path.join(directory, "meta.json")
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(self.meta.__dict__, fh, indent=1)
        return directory

    @classmethod
    def load(cls, directory: str) -> "ModelBundle":
        with open(os.path.join(directory, "meta.json"), "r", encoding="utf-8") as fh:
            meta = ModelBundleMeta(**json.load(fh))
        regime = RegimeClassifier()
        regime.load(os.path.join(directory, "regime.txt"))
        timing = TimingModel()
        timing.load(os.path.join(directory, "timing.txt"))
        return cls(regime, timing, meta)


def build_labels(feats: pd.DataFrame, horizon: int, min_gain: float,
                 regime_horizon: int, regime_tol: float,
                 close: pd.Series | None = None) -> pd.DataFrame:
    """Create causal (closed-candle) labels.

    * ``timing``  -- 1 if the forward return over ``horizon`` beats
      ``+min_gain``, 0 if it loses more than ``min_gain``, NaN in between
      (the fee-aware middle is dropped so the model must predict moves
      that actually clear costs).
    * ``regime``  -- 2/1/0 (up/range/down) from the forward return over
      ``regime_horizon`` bucketed by ``regime_tol``.
    """
    if close is None:
        if "close" in feats.columns:
            close = feats["close"]
        else:
            raise ValueError("build_labels needs a close series (caller must pass one)")
    fwd = close.copy()
    timing = pd.Series(np.nan, index=fwd.index, dtype=float)
    fwd_t = fwd.shift(-horizon) / fwd - 1.0
    timing[fwd_t > min_gain] = 1.0
    timing[fwd_t < -min_gain] = 0.0

    reg = pd.Series(REGIME_RANGE, index=fwd.index, dtype=int)
    fwd_r = fwd.shift(-regime_horizon) / fwd - 1.0
    reg[fwd_r > regime_tol] = REGIME_UP
    reg[fwd_r < -regime_tol] = REGIME_DOWN
    return pd.DataFrame({"timing": timing, "regime": reg}, index=fwd.index)