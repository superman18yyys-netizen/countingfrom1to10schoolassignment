"""R&D generation 2: ideas informed by gen-1 verdicts.

Gen-1 lessons: cash-discipline earns bear excess but dies in bull legs
(fold 2, -69%); only evidence-confirmed entries that actually TRADE
won OOS (donchian_sage). Fees demand few, big, well-timed trades.

Gen-2 hypotheses:
  TrendPullback  — the highest-expectancy classic: buy pullbacks INSIDE
                   confirmed uptrends (not new highs): better entries,
                   tighter risk, rides the same edge as trend-following
                   without paying for breakouts.
  Committee      — Sage panel AND donchian_sage agreement: variance
                   reduction via independent-signal consensus.
  MTFTrend       — 4H entries only when the 1D trend (resampled from
                   the same window) agrees: multi-timeframe confluence.
  VolTrailExit   — trend entry, ATR-scaled chandelier exit: profits
                   scale with realized volatility instead of fixed bars.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bot.indicators.ta import atr, ema, rsi, sma
from bot.strategies.base import Strategy
from bot.strategies.sage import SageStrategy


class TrendPullback(Strategy):
    """Buy the dip inside a confirmed uptrend; exit when the trend
    breaks or the bounce completes."""
    name = "trend_pullback"
    DEFAULTS = {"trend_sma": 150, "pullback_ema": 20, "rsi_lo": 35,
                "exit_rsi": 60, "slope_bars": 50}

    def warmup_bars(self) -> int:
        return 210

    def compute_signals(self, df: pd.DataFrame, live: bool = False) -> pd.Series:
        p = {**self.DEFAULTS, **self.params}
        close = df["close"]
        sma_t = sma(close, int(p["trend_sma"]))
        slope = close.rolling(int(p["slope_bars"])).apply(
            lambda v: np.polyfit(np.arange(len(v)), v, 1)[0] / v.mean()
            if v.mean() > 0 else 0.0, raw=True).fillna(0.0)
        uptrend = (close > sma_t) & (slope > 0)
        # pullback: touched the fast EMA (or below) with stretched RSI
        pb = close <= ema(close, int(p["pullback_ema"])) * 1.002
        stretched = rsi(close, 14) < float(p["rsi_lo"])
        buy = uptrend & pb & stretched
        # exit: bounce completes (RSI high) or trend breaks
        sell = (rsi(close, 14) > float(p["exit_rsi"])) | (close < sma_t)
        sig = pd.Series(0, index=df.index, dtype=int)
        sig[buy.fillna(False)] = 1
        sig[sell.fillna(False)] = -1
        return sig


class Committee(Strategy):
    """Sage evidence panel AND DonchianSage breakout must agree before
    any entry; either can exit."""
    name = "committee"
    DEFAULTS = {"sage_buy": 3.0, "entry_period": 30, "exit_period": 10,
                "min_score": 2.0}

    def warmup_bars(self) -> int:
        return 220

    def compute_signals(self, df: pd.DataFrame, live: bool = False) -> pd.Series:
        p = {**self.DEFAULTS, **self.params}
        close = df["close"]
        sage = SageStrategy({"buy_score": float(p["sage_buy"])}) \
            .compute_signals(df)
        hi = df["high"].rolling(int(p["entry_period"])).max().shift(1)
        lo = df["low"].rolling(int(p["exit_period"])).min().shift(1)
        panel = SageStrategy({}).score_series(df)
        brk = (close > hi) & (panel >= float(p["min_score"]))
        buy = (sage == 1) & brk
        sell = (sage == -1) | (close < lo)
        sig = pd.Series(0, index=df.index, dtype=int)
        sig[buy.fillna(False)] = 1
        sig[sell.fillna(False)] = -1
        return sig


class MTFTrend(Strategy):
    """4H trend entry gated by the 1D trend (resampled from the same
    window — no extra data dependency)."""
    name = "mtf_trend"
    DEFAULTS = {"fast": 20, "slow": 50, "day_sma": 30, "day_band": 0.01}

    def warmup_bars(self) -> int:
        return 240

    def compute_signals(self, df: pd.DataFrame, live: bool = False) -> pd.Series:
        p = {**self.DEFAULTS, **self.params}
        close = df["close"]
        f, s = ema(close, int(p["fast"])), ema(close, int(p["slow"]))
        cross_up = (f > s) & (f.shift(1) <= s.shift(1))
        cross_dn = (f < s) & (f.shift(1) >= s.shift(1))
        # resample to daily using only data up to each bar (expanding
        # means are causal; the last partial day uses closes so far)
        daily = close.resample("1D").last().dropna()
        day_sma = daily.rolling(int(p["day_sma"]), min_periods=10).mean()
        day_ok = (daily > day_sma * (1 - float(p["day_band"]))).fillna(False)
        # map daily gate back onto hourly index (forward-fill, causal)
        gate = day_ok.reindex(close.index, method="ffill").fillna(False)
        buy = cross_up & gate
        sell = cross_dn | ~gate
        sig = pd.Series(0, index=df.index, dtype=int)
        sig[buy.fillna(False)] = 1
        sig[sell.fillna(False)] = -1
        return sig


class VolTrailExit(Strategy):
    """Trend entry, chandelier exit at 3x ATR below the running high:
    winners run as far as volatility allows, losers cut fast."""
    name = "vol_trail_exit"
    DEFAULTS = {"trend_sma": 150, "atr_period": 14, "atr_mult": 3.0,
                "rsi_cap": 65}

    def warmup_bars(self) -> int:
        return 200

    def compute_signals(self, df: pd.DataFrame, live: bool = False) -> pd.Series:
        p = {**self.DEFAULTS, **self.params}
        close = df["close"]
        sma_t = sma(close, int(p["trend_sma"]))
        a = atr(df["high"], df["low"], close, int(p["atr_period"]))
        # chandelier: rolling max of high minus atr_mult * ATR
        chan = df["high"].rolling(50, min_periods=10).max() \
            - float(p["atr_mult"]) * a
        trend_up = close > sma_t
        not_extended = rsi(close, 14) < float(p["rsi_cap"])
        # enter when trend is up and price pulls back near the chandelier
        near_stop = close <= chan * 1.01
        buy = trend_up & not_extended & near_stop
        sell = close < chan
        sig = pd.Series(0, index=df.index, dtype=int)
        sig[buy.fillna(False)] = 1
        sig[sell.fillna(False)] = -1
        return sig


class SwingRider(Strategy):
    """Born from the miss analysis (Aug 2026): 84% of the >=8% swings
    the bots missed were rallies that began INSIDE downtrends (signal-
    blind), and on the swings they did catch they exited at ~+2% while
    +19.8% more was available (fee-band profit taking).

    Fixes both directly:
      ENTRY  — momentum ignition: price up >= surge_pct over surge_bars
               (a rally is a rally, whatever the regime; the chassis
               stop + sizing bound the risk)
      EXIT   — chandelier trail: 50-bar high minus atr_mult*ATR. Winners
               ride the swing; losers cut immediately.
    """
    name = "swing_rider"
    DEFAULTS = {"surge_pct": 0.05, "surge_bars": 12,
                "atr_period": 14, "atr_mult": 2.5, "cooldown": 6,
                "vol_ok_pctile": 0.30}

    def warmup_bars(self) -> int:
        return 200

    def compute_signals(self, df: pd.DataFrame, live: bool = False) -> pd.Series:
        import numpy as _np
        p = {**self.DEFAULTS, **self.params}
        close = df["close"]
        surge = close / close.shift(int(p["surge_bars"])) - 1.0
        a = atr(df["high"], df["low"], close, int(p["atr_period"]))
        a_pct = a / close
        a_pctile = a_pct.rolling(540, min_periods=120).apply(
            lambda v: (v <= v[-1]).mean(), raw=True).fillna(0.5)
        ignite = (surge >= float(p["surge_pct"])) & \
                 (a_pctile >= float(p["vol_ok_pctile"]))
        # take only the FIRST ignition bar of a cluster (cooldown)
        buy = ignite & ~ignite.shift(1, fill_value=False)
        buy = buy & ~buy.rolling(int(p["cooldown"])).sum().shift(1).fillna(0).gt(0)
        chan = df["high"].rolling(50, min_periods=10).max() \
            - float(p["atr_mult"]) * a
        sell = close < chan
        sig = pd.Series(0, index=df.index, dtype=int)
        sig[buy.fillna(False)] = 1
        sig[sell.fillna(False)] = -1
        return sig
