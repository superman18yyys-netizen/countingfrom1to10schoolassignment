"""Gen-5: direct improvements to the elite champion (mtf_trend)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from bot.indicators.ta import atr, ema, rsi
from bot.strategies.base import Strategy
from bot.strategies.sage import SageStrategy


class MTFChandelier(Strategy):
    """mtf_trend entry + wide ATR chandelier exit (tested: WORSE than
    the champion's slow cross-down exit — kept for reference)."""
    name = "mtf_chandelier"
    DEFAULTS = {"fast": 20, "slow": 50, "day_sma": 30, "day_band": 0.01,
                "atr_period": 14, "atr_mult": 4.5}

    def warmup_bars(self) -> int:
        return 240

    def compute_signals(self, df: pd.DataFrame, live: bool = False) -> pd.Series:
        p = {**self.DEFAULTS, **self.params}
        close = df["close"]
        f, s = ema(close, int(p["fast"])), ema(close, int(p["slow"]))
        cross_up = (f > s) & (f.shift(1) <= s.shift(1))
        daily = close.resample("1D").last().dropna()
        day_sma = daily.rolling(int(p["day_sma"]), min_periods=10).mean()
        day_ok = (daily > day_sma * (1 - float(p["day_band"]))).fillna(False)
        gate = day_ok.reindex(close.index, method="ffill").fillna(False)
        a = atr(df["high"], df["low"], close, int(p["atr_period"]))
        chan = df["high"].rolling(50, min_periods=10).max() \
            - float(p["atr_mult"]) * a
        buy = cross_up & gate
        sell = close < chan
        sig = pd.Series(0, index=df.index, dtype=int)
        sig[buy.fillna(False)] = 1
        sig[sell.fillna(False)] = -1
        return sig


class MTFPullback(Strategy):
    """mtf_trend with pullback-to-EMA entries (tested: WORSE — kept for
    reference)."""
    name = "mtf_pullback"
    DEFAULTS = {"fast": 20, "slow": 50, "day_sma": 30, "day_band": 0.01,
                "pullback_ema": 20, "rsi_lo": 40, "valid_bars": 30}

    def warmup_bars(self) -> int:
        return 240

    def compute_signals(self, df: pd.DataFrame, live: bool = False) -> pd.Series:
        p = {**self.DEFAULTS, **self.params}
        close = df["close"]
        f, s = ema(close, int(p["fast"])), ema(close, int(p["slow"]))
        cross_up = (f > s) & (f.shift(1) <= s.shift(1))
        since_cross = cross_up.cumsum()
        in_window = (since_cross == since_cross.shift(1)) & (since_cross > 0)
        age = in_window.groupby(in_window.cumsum()).cumcount()
        fresh = age <= int(p["valid_bars"])
        pb = close <= ema(close, int(p["pullback_ema"])) * 1.003
        cooled = rsi(close, 14) < float(p["rsi_lo"])
        daily = close.resample("1D").last().dropna()
        day_sma = daily.rolling(int(p["day_sma"]), min_periods=10).mean()
        day_ok = (daily > day_sma * (1 - float(p["day_band"]))).fillna(False)
        gate = day_ok.reindex(close.index, method="ffill").fillna(False)
        buy = in_window & fresh & pb & cooled & gate
        sell = (f < s) | ~gate
        sig = pd.Series(0, index=df.index, dtype=int)
        sig[buy.fillna(False)] = 1
        sig[sell.fillna(False)] = -1
        return sig


class ElitePair(Strategy):
    """mtf_trend + donchian_sage ensemble: buy only when BOTH the 4H
    trend-cross AND the evidence-confirmed breakout agree (highest
    conviction overlaps); chandelier exit. The blend of the two most
    consistent bots (9/10 folds positive each, worst folds -5.1%/-0.9%)
    — an overlap filter should retain only their agreement zone."""
    name = "elite_pair"
    DEFAULTS = {"day_sma": 30, "entry_period": 30, "min_score": 2.0,
                "fast": 20, "slow": 50, "atr_mult": 4.0}

    def warmup_bars(self) -> int:
        return 240

    def compute_signals(self, df: pd.DataFrame, live: bool = False) -> pd.Series:
        p = {**self.DEFAULTS, **self.params}
        close = df["close"]
        f, s = ema(close, int(p["fast"])), ema(close, int(p["slow"]))
        cross_up = (f > s) & (f.shift(1) <= s.shift(1))
        daily = close.resample("1D").last().dropna()
        day_sma = daily.rolling(int(p["day_sma"]), min_periods=10).mean()
        day_ok = (daily > day_sma).fillna(False)
        gate = day_ok.reindex(close.index, method="ffill").fillna(False)
        hi = df["high"].rolling(int(p["entry_period"])).max().shift(1)
        panel = SageStrategy({}).score_series(df)
        brk = (close > hi) & (panel >= float(p["min_score"]))
        a = atr(df["high"], df["low"], close, 14)
        chan = df["high"].rolling(50, min_periods=10).max() \
            - float(p["atr_mult"]) * a
        buy = cross_up & gate & brk
        sell = close < chan
        sig = pd.Series(0, index=df.index, dtype=int)
        sig[buy.fillna(False)] = 1
        sig[sell.fillna(False)] = -1
        return sig
