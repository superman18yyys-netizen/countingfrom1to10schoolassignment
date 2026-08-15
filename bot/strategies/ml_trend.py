"""Trainable, deployable two-stage trend strategy (long / flat).

Stage 1 (regime) decides *whether* to be long -- learned from the causal
feature matrix, it lets the strategy sit out flat/down regimes (the main
source of alpha in the research: "stay out of the drawdown"). Stage 2
(timing) picks *when* to enter within a favourable regime, using a
fee-aware up/down label.

Signal semantics (compatible with bot/backtest/engine):
    1  -> enter long on next bar's open
   -1  -> exit long on next bar's open
    0  -> no action

fit() trains both LightGBM models on a candle frame. compute_signals()
predicts on a (possibly forward) frame using those models. This lets a
single class drive walk-forward CV, the holdout gate, and live paper
trading, exactly like the existing MLStrategy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bot.strategies.base import Strategy
from bot.train.features import FEATURES_VERSION, build_features
from bot.train.models import (ModelBundle, ModelBundleMeta, RegimeClassifier,
                              TimingModel, build_labels)

_WARMUP = 200   # widest rolling window in the feature matrix

DEFAULT_HYPER = {
    "horizon": 6,
    "min_gain": 0.013,
    "regime_horizon": 24,
    "regime_tol": 0.004,
    "regime_up": 0.50,
    "buy": 0.60,
    "exit": 0.50,
}


class MLTrendStrategy(Strategy):
    name = "ml_trend"

    def __init__(self, params=None):
        super().__init__(params)
        self.p = {**DEFAULT_HYPER, **(params or {})}
        self.bundle: ModelBundle | None = None
        self._expiry: pd.Timestamp | None = None

    # ---------------------------------------------------------------- fit
    def fit(self, df: pd.DataFrame) -> None:
        if len(df) < 400:
            self.bundle = None
            return
        feats = self._features(df)
        labels = build_labels(feats, int(self.p["horizon"]),
                              float(self.p["min_gain"]),
                              int(self.p["regime_horizon"]),
                              float(self.p["regime_tol"]),
                              close=df["close"])
        cols = [c for c in feats.columns]
        mask_t = labels["timing"].notna()
        mask_r = labels["regime"].notna()

        regime = RegimeClassifier()
        timing = TimingModel()
        regime.fit(feats[mask_r][cols], labels["regime"][mask_r].astype(int))
        timing.fit(feats[mask_t][cols], labels["timing"][mask_t].astype(int))

        meta = ModelBundleMeta(
            horizon=int(self.p["horizon"]),
            min_gain=float(self.p["min_gain"]),
            regime_horizon=int(self.p["regime_horizon"]),
            regime_tol=float(self.p["regime_tol"]),
            buy_threshold=float(self.p["buy"]),
            exit_threshold=float(self.p["exit"]),
            features_version=FEATURES_VERSION,
            warmup_bars=max(_WARMUP, int(self.p["regime_horizon"])),
            trained_at=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC"),
            fit_pairs=[], fit_start=str(df.index[0]), fit_end=str(df.index[-1]),
        )
        self.bundle = ModelBundle(regime, timing, meta)

    # ----------------------------------------------------------- features
    def _features(self, df: pd.DataFrame) -> pd.DataFrame:
        # cross_price=None -> rel_strength stays a benign constant (0.0). A
        # future extension can feed an aligned reference pair series here.
        return build_features(df, cross_price=None)

    # ------------------------------------------------------------- predict
    def compute_signals(self, df: pd.DataFrame, live: bool = False) -> pd.Series:
        sig = pd.Series(0, index=df.index, dtype=int)
        if live:
            # Live/paper mode: keep a rolling-fit model, refresh on expiry.
            try:
                import lightgbm  # noqa: F401
            except ImportError:
                return sig
            bar_sec = max(60, int(round(
                (df.index[-1] - df.index[0]).total_seconds() / max(1, len(df) - 1)
            )))
            if self.bundle is None or self._expiry is None or df.index[-1] > self._expiry:
                train_bars = int(self.p.get("regime_horizon", 24) * 40)
                self.fit(df.iloc[-train_bars:])
                self._expiry = df.index[-1] + pd.Timedelta(
                    seconds=int(self.p.get("regime_horizon", 24)) * bar_sec)
            if self.bundle is None:
                return sig
        elif self.bundle is None:
            return sig
        feats = self._features(df)
        cols = [c for c in feats.columns]
        try:
            up = np.nan_to_num(self.bundle.regime.predict(feats[cols]).to_numpy(dtype=float))
            tp = np.nan_to_num(self.bundle.timing.predict(feats[cols]).to_numpy(dtype=float))
        except Exception:  # noqa: BLE001
            return sig
        buy_th = float(self.p["buy"])
        exit_th = float(self.p["exit"])
        regime_th = float(self.p["regime_up"])
        tradable = up >= regime_th
        sig = pd.Series(np.where(tradable & (tp >= buy_th), 1,
                                 np.where(~tradable | (tp <= exit_th), -1, 0)),
                        index=df.index, dtype=int)
        return sig

    def warmup_bars(self) -> int:
        return max(_WARMUP, int(self.p["regime_horizon"]))