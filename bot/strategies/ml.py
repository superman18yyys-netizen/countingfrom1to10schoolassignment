"""ML direction-classifier strategy (LightGBM gradient boosting).

Research basis (verified Aug 2026): gradient-boosted trees
(LightGBM/XGBoost) are the best-practice model family for crypto
direction prediction; realistic out-of-sample ROC-AUC is 0.57-0.61
(Sobreiro et al. 2026; Lyu 2022). Anything above ~0.65 is overfitting.
The label is fee-aware: UP only counts if the forward return exceeds the
round-trip cost (taker fee + slippage), and the ambiguous middle is
dropped, so the model is forced to predict moves that are actually
profitable.

Feature set (closed-candle-only, causal):
  * returns over [1, 2, 3, 6, 12, 24] bars
  * RSI(14)
  * Bollinger %B position (BB position is the single best 4h feature in Sobreiro 2026)
  * ATR% (ATR / close)
  * volume z-score (48 bars)
  * volume-to-volatility ratio (liquidity proxy, Islam et al. 2025)
  * EMA slope: (EMA12 - EMA48) / close
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:  # pragma: no cover
    lgb = None
    _HAS_LGB = False

from bot.indicators.ta import atr, bollinger, ema, rsi, volume_to_volatility, volume_zscore
from bot.strategies.base import Strategy


class MLOptimizerStrategy(Strategy):
    name = "ml"

    DEFAULTS = {
        "horizon": 6,          # bars of future return used for the label
        "min_gain": 0.013,     # ~ 0.6% taker + 0.1% slippage each side (round trip)
        "min_samples": 200,    # below this, fall back to plain directional labels
        "train_days": 60,
        "test_days": 7,
        "purge_bars": 12,      # gap between train and test windows (no leakage)
        "buy_threshold": 0.60,
        "exit_threshold": 0.50,
        "expiry_bars": 48,     # live model expiry before retraining
    }

    def __init__(self, params=None):
        super().__init__(params)
        self.p = {**self.DEFAULTS, **self.params}
        self._model = None
        self._model_expiry: pd.Timestamp | None = None

    # ------------------------------------------------------------- features
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        close, high, low = df["close"], df["high"], df["low"]
        volume = df["volume"]
        feats = {}
        for h in (1, 2, 3, 6, 12, 24):
            feats[f"ret_{h}"] = close.pct_change(h)
        feats["rsi"] = rsi(close, 14)
        bb = bollinger(close, 20, 2.0)
        feats["bb_position"] = bb["bb_position"]
        feats["atr_pct"] = atr(high, low, close, 14) / close
        feats["vol_z"] = volume_zscore(volume, 48)
        feats["vvr"] = volume_to_volatility(volume, feats["atr_pct"], 24)
        feats["ema_slope"] = (ema(close, 12) - ema(close, 48)) / close
        feats["close"] = close
        out = pd.DataFrame(feats, index=df.index)
        return out.replace([np.inf, -np.inf], np.nan)

    def build_label(self, df: pd.DataFrame) -> pd.Series:
        """Fee-aware UP/DOWN label; falls back to plain direction if the
        fee-aware version leaves too few usable samples (quiet regimes)."""
        h = int(self.p["horizon"])
        min_gain = float(self.p["min_gain"])
        fwd = df["close"].shift(-h) / df["close"] - 1.0
        label = pd.Series(np.nan, index=df.index, dtype=float)
        label[fwd > min_gain] = 1.0
        label[fwd < -min_gain] = 0.0
        if label.notna().sum() >= int(self.p["min_samples"]):
            return label
        label = pd.Series(np.nan, index=df.index, dtype=float)
        label[fwd > 0] = 1.0
        label[fwd <= 0] = 0.0
        return label

    def _dataset(self, df: pd.DataFrame):
        feats = self.build_features(df)
        label = self.build_label(df)
        mask = label.notna()
        X = feats[mask].drop(columns=["close"])
        y = label[mask].astype(int)
        return X, y

    # ------------------------------------------------------------- training
    def fit(self, df: pd.DataFrame) -> None:
        if not _HAS_LGB:
            raise RuntimeError("lightgbm is not installed (pip install lightgbm)")
        X, y = self._dataset(df)
        if len(y) < 200:
            return
        split = int(len(y) * 0.85)
        dtrain = lgb.Dataset(X.iloc[:split], label=y.iloc[:split])
        dvalid = lgb.Dataset(X.iloc[split:], label=y.iloc[split:], reference=dtrain)
        params = {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "verbose": -1,
            "n_jobs": 2,
        }
        self._model = lgb.train(
            params, dtrain, num_boost_round=400,
            valid_sets=[dvalid], callbacks=[lgb.early_stopping(30)],
        )
        self._model_expiry = df.index[-1]

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if self._model is None:
            return pd.Series(np.nan, index=df.index)
        feats = self.build_features(df).drop(columns=["close"])
        proba = self._model.predict(feats, num_iteration=self._model.best_iteration or None)
        return pd.Series(proba, index=df.index)

    # ------------------------------------------------------ walk-forward
    def _walkforward(self, df: pd.DataFrame) -> pd.Series:
        """Train on rolling windows and predict forward, with purge gaps.

        First window: train_days of history; afterwards retrain every
        test_days using a trailing train window (purging test_days of
        the oldest data, leaving a purge gap before the predict zone).
        """
        bar_sec = max(60, int(round(
            (df.index[-1] - df.index[0]).total_seconds() / max(1, len(df) - 1)
        )))
        train_bars = int(self.p["train_days"] * 86400 / bar_sec)
        test_bars = int(max(1, (self.p["test_days"] * 86400) / bar_sec))
        purge = int(self.p["purge_bars"])

        proba = pd.Series(np.nan, index=df.index, dtype=float)
        if len(df) < train_bars + test_bars + purge:
            return proba

        pos = train_bars
        while pos + purge < len(df):
            train_df = df.iloc[pos - train_bars:pos]
            self.fit(train_df)
            lo = pos + purge
            hi = min(pos + purge + test_bars, len(df))
            chunk = df.iloc[lo:hi]
            if self._model is not None and not chunk.empty:
                proba.iloc[lo:hi] = self.predict(chunk).values
            pos += test_bars

        self._model = None  # don't leak a partial-window model into paper mode
        self._model_expiry = None
        return proba

    # -------------------------------------------------------------- signals
    def compute_signals(self, df: pd.DataFrame, live: bool = False) -> pd.Series:
        buy_th = float(self.p["buy_threshold"])
        exit_th = float(self.p["exit_threshold"])
        sig = pd.Series(0, index=df.index, dtype=int)

        if live:
            # paper mode: keep one trailing-window model, refresh on expiry
            bar_sec = max(60, int(round(
                (df.index[-1] - df.index[0]).total_seconds() / max(1, len(df) - 1)
            )))
            if self._model is None or self._model_expiry is None \
                    or df.index[-1] > self._model_expiry:
                train_bars = int(self.p["train_days"] * 86400 / bar_sec)
                self.fit(df.iloc[-train_bars:])
                self._model_expiry = df.index[-1] + pd.Timedelta(
                    seconds=int(self.p["expiry_bars"]) * bar_sec)
            proba = self.predict(df)
        elif self._model is not None:
            # OOS mode: a model was fitted on train data; predict forward
            proba = self.predict(df)
        else:
            # full backtest: rolling walk-forward fit + predict
            proba = self._walkforward(df)

        sig[proba >= buy_th] = 1
        sig[proba <= exit_th] = -1
        sig[proba.isna()] = 0
        return sig

    def warmup_bars(self) -> int:
        return int(self.p["train_days"]) * 24