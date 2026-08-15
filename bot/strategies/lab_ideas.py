"""New bot archetypes for the R&D lab.

Each class is a hypothesis. The lab (run_lab.py) trains the small
tunable surface on the TRAIN span and one-shot validates on the
untouched span via live-path replay, then ranks vs the Sage baseline.

Ideas and their hypotheses:
  VolAwakening   — vol expansions out of a drought start the big moves;
                   buy the FIRST expansion bar with trend confirmation.
  SageRS         — Sage's panel + cross-sectional relative strength:
                   only trade the pair when it is stronger than BTC.
  DonchianSage   — breakout entry only when Sage's evidence panel
                   agrees (kills false breakouts, Costa 2026).
  RangeSniper    — pure mean reversion, RANGE regime only, extreme
                   entries (BB lower + RSI), patient fee-clearing exits.
  Seasonal       — time-of-week seasonality gate on top of trend
                   (crypto's documented weekend/hour effects).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bot.indicators.ta import atr, bollinger, ema, rsi, sma
from bot.strategies.base import Strategy
from bot.strategies.sage import SageStrategy


class VolAwakening(Strategy):
    """Buy the first volatility expansion out of a drought when the
    trend agrees — the 'market wakes up' moment."""
    name = "vol_awakening"
    DEFAULTS = {"drought_pctl": 0.25, "wake_mult": 1.6, "trend_ema": 100,
                "confirm": 2, "rsi_cap": 65}
    # warmup handled via trend_ema

    def warmup_bars(self) -> int:
        return 250

    def compute_signals(self, df: pd.DataFrame, live: bool = False) -> pd.Series:
        p = {**self.DEFAULTS, **self.params}
        close = df["close"]
        a = atr(df["high"], df["low"], close, 14) / close
        base = a.rolling(540, min_periods=120).median()
        drought = a < base  # subdued vs its own norm
        # awakening: was in drought, now ATR jumps wake_mult x the base
        wake = drought.shift(1) & (a >= base * float(p["wake_mult"]))
        trend_ok = close > ema(close, int(p["trend_ema"]))
        not_extended = rsi(close, 14) < float(p["rsi_cap"])
        buy = wake & trend_ok & not_extended
        # exit when the move exhausts (RSI high) or trend breaks
        sell = (rsi(close, 14) > 75) | (close < ema(close, int(p["trend_ema"])))
        # require `confirm` consecutive wake bars
        conf = int(p["confirm"])
        buy_conf = buy.rolling(conf).sum() >= conf if conf > 1 else buy
        sig = pd.Series(0, index=df.index, dtype=int)
        sig[buy_conf.fillna(False)] = 1
        sig[sell] = -1
        return sig


class SageRS(SageStrategy):
    """Sage + relative strength vs the market: only take long signals
    when THIS pair is outperforming BTC over the near window. Needs
    cross data injected via df.attrs['cross'] (set by the lab/live
    runner when available; without it, degrades to plain Sage)."""
    name = "sage_rs"
    DEFAULTS = {**SageStrategy.DEFAULTS, "rs_bars": 120, "rs_edge": 0.0}

    def warmup_bars(self) -> int:
        return super().warmup_bars() + 10

    def compute_signals(self, df: pd.DataFrame, live: bool = False) -> pd.Series:
        sig = super().compute_signals(df, live=live)
        # cross close prices must be stored as a PLAIN numpy array:
        # pandas Series/DataFrame in df.attrs breaks pd.concat's
        # internal attrs comparison.
        cross = df.attrs.get("cross_close")
        if cross is None:
            return sig
        cross = np.asarray(cross, dtype=float)
        p = {**self.DEFAULTS, **self.params}
        n = int(p["rs_bars"])
        if len(df) < n or len(cross) < n:
            return sig
        rs_pair = float(df["close"].iloc[-1]) / float(df["close"].iloc[-n]) - 1.0
        rs_btc = float(cross[-1]) / float(cross[-n]) - 1.0
        if rs_pair - rs_btc <= float(p["rs_edge"]):
            # weak vs market: entries blocked, exits always pass
            return sig.where(sig == -1, 0).astype(int)
        return sig


class DonchianSage(Strategy):
    """Breakout + evidence agreement: 20-bar high breakout only fires
    when Sage's panel score already supports it."""
    name = "donchian_sage"
    DEFAULTS = {"entry_period": 20, "exit_period": 10,
                "min_score": 1.5}
    _panel = None

    def warmup_bars(self) -> int:
        return 220

    def compute_signals(self, df: pd.DataFrame, live: bool = False) -> pd.Series:
        p = {**self.DEFAULTS, **self.params}
        close = df["close"]
        hi = df["high"].rolling(int(p["entry_period"])).max().shift(1)
        lo = df["low"].rolling(int(p["exit_period"])).min().shift(1)
        panel = SageStrategy({}).score_series(df)
        buy = (close > hi) & (panel >= float(p["min_score"]))
        sell = close < lo
        sig = pd.Series(0, index=df.index, dtype=int)
        sig[buy.fillna(False)] = 1
        sig[sell.fillna(False)] = -1
        return sig


class RangeSniper(Strategy):
    """Mean reversion, RANGE regime only, extreme entries, exits that
    clear fees. Few, patient, high-win-rate trades."""
    name = "range_sniper"
    DEFAULTS = {"rsi_period": 14, "rsi_lo": 25, "bb_period": 40,
                "bb_std": 2.0, "exit_rsi": 55, "trend_sma": 150,
                "band_pct": 0.04}

    def warmup_bars(self) -> int:
        return 200

    def compute_signals(self, df: pd.DataFrame, live: bool = False) -> pd.Series:
        p = {**self.DEFAULTS, **self.params}
        close = df["close"]
        sma_t = sma(close, int(p["trend_sma"]))
        dist = (close - sma_t) / sma_t
        in_range = dist.abs() <= float(p["band_pct"])
        r = rsi(close, int(p["rsi_period"]))
        bb = bollinger(close, int(p["bb_period"]), float(p["bb_std"]))
        buy = in_range & (r < float(p["rsi_lo"])) & (close < bb["bb_lower"])
        sell = r > float(p["exit_rsi"])
        sig = pd.Series(0, index=df.index, dtype=int)
        sig[buy.fillna(False)] = 1
        sig[sell] = -1
        return sig


class SeasonalTrend(Strategy):
    """Trend following gated by time-of-week: only take entries in the
    historically favorable part of the week (documented crypto
    weekday/hour effects); exits always active."""
    name = "seasonal_trend"
    DEFAULTS = {"trend_ema": 150, "fast": 20, "slow": 50,
                "good_days": (1, 2, 3), "good_hours": (6, 7, 8, 9, 10, 11)}

    def warmup_bars(self) -> int:
        return 200

    def compute_signals(self, df: pd.DataFrame, live: bool = False) -> pd.Series:
        p = {**self.DEFAULTS, **self.params}
        close = df["close"]
        f, s = ema(close, int(p["fast"])), ema(close, int(p["slow"]))
        t = ema(close, int(p["trend_ema"]))
        cross_up = (f > s) & (f.shift(1) <= s.shift(1))
        cross_dn = (f < s) & (f.shift(1) >= s.shift(1))
        idx = df.index
        day_ok = pd.Series(np.isin(idx.dayofweek, p["good_days"]), index=idx)
        hour_ok = pd.Series(np.isin(idx.hour, p["good_hours"]), index=idx)
        buy = cross_up & (close > t) & day_ok & hour_ok
        sell = cross_dn | (close < t)
        sig = pd.Series(0, index=df.index, dtype=int)
        sig[buy.fillna(False)] = 1
        sig[sell.fillna(False)] = -1
        return sig
