"""Version-2 winners + the consensus combiner.

Improvements driven by the lab's own findings (what actually worked):

* DeepRecoveryV2 -- deep_recovery was the best OOS candidate; v2 adds:
    - volume confirmation (recovery must have real participation)
    - ATR-adaptive dip threshold (in high vol, demand deeper discounts;
      in low vol, act on shallower dips) so the bot self-scales
    - "momentum death" exit (recovery failed -> leave) alongside the
      target/stop exits
* AdaptiveGrid -- guarded_grid was the second positive-OOS bot; v2 makes
    the grid step ATR-scaled so every round trip clears fees by
    construction, instead of a fixed 1.5% step.
* Consensus -- only trades when MULTIPLE independent winning signals
    agree: a real dip + confirmed recovery + volume participation +
    not-in-freefall. Each component filters the others' failure modes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bot.indicators.ta import atr, ema, rolling_median, sma, volume_zscore
from bot.strategies.base import Strategy


class DeepRecoveryV2(Strategy):
    name = "deep_recovery_v2"

    DEFAULTS = {
        "drawdown_window": 48,
        "base_drawdown_pct": 0.12,   # scaled by ATR each bar (adaptive)
        "atr_ref_pct": 0.015,        # ATR% at which the base threshold applies
        "recover_ema": 9,
        "momentum_bars": 6,
        "volume_z_min": 0.5,         # recovery must come with volume participation
        "target_pct": 0.12,
        "stop_pct": 0.06,
    }

    def __init__(self, params=None):
        super().__init__(params)
        self.p = {**self.DEFAULTS, **(params or {})}

    def warmup_bars(self) -> int:
        return int(self.p["drawdown_window"]) + int(self.p["recover_ema"]) + 5

    def compute_signals(self, df: pd.DataFrame, live: bool = False) -> pd.Series:
        p = self.p
        win = int(p["drawdown_window"])
        close, high, low = df["close"], df["high"], df["low"]

        rolling_high = close.rolling(win, min_periods=1).max()
        trailing_low = low.rolling(win, min_periods=1).min().shift(1)
        drawdown = close / rolling_high - 1.0

        # Adaptive dip threshold: scale with realized volatility.
        atr_pct = (atr(high, low, close, 14) / close.replace(0.0, np.nan)).fillna(0.0)
        scale = (atr_pct / float(p["atr_ref_pct"])).clip(0.5, 2.5)
        dip_thresh = -float(p["base_drawdown_pct"]) * scale
        in_dip = drawdown <= dip_thresh

        fast_ema = ema(close, int(p["recover_ema"]))
        momentum = close.pct_change(int(p["momentum_bars"]))
        vol_z = volume_zscore(df["volume"], 48)
        recovering = (close >= fast_ema) & (momentum > 0)
        vol_ok = vol_z >= float(p["volume_z_min"])

        target = (trailing_low > 0) & (close >= trailing_low * (1.0 + float(p["target_pct"])))
        stop = (trailing_low > 0) & (close <= trailing_low * (1.0 - float(p["stop_pct"])))
        momentum_dead = (close < fast_ema) & (momentum < 0)   # recovery failed

        sig = pd.Series(0, index=df.index, dtype=int)
        sig[in_dip & recovering & vol_ok] = 1
        sig[target | stop | momentum_dead] = -1
        return sig


class AdaptiveGrid(Strategy):
    name = "adaptive_grid"

    DEFAULTS = {
        "reference_bars": 96,
        "base_step": 0.012,          # floor step; ATR-scaled upward
        "atr_mult": 2.0,             # step = max(base, atr_mult * ATR%)
        "fee_clear_mult": 1.5,       # step must be >= 1.5x the round trip
        "round_trip_cost": 0.014,
        "trend_sma": 200,
        "range_band": 0.05,          # only grid within +/-5% of the SMA
    }

    def __init__(self, params=None):
        super().__init__(params)
        self.p = {**self.DEFAULTS, **(params or {})}

    def warmup_bars(self) -> int:
        return int(self.p["reference_bars"]) + 5

    def compute_signals(self, df: pd.DataFrame, live: bool = False) -> pd.Series:
        p = self.p
        close, high, low = df["close"], df["high"], df["low"]
        ref = rolling_median(close, int(p["reference_bars"]))
        atr_pct = (atr(high, low, close, 14) / close.replace(0.0, np.nan)).fillna(0.0)
        # Step scales with volatility and always clears fees by construction.
        step = np.maximum.reduce([
            np.full(len(df), float(p["base_step"])),
            (float(p["atr_mult"]) * atr_pct).to_numpy(),
            np.full(len(df), float(p["fee_clear_mult"]) * float(p["round_trip_cost"])),
        ])
        step = pd.Series(step, index=df.index)

        sma_long = sma(close, int(p["trend_sma"]))
        in_range = ((close - sma_long) / sma_long.replace(0.0, np.nan)).abs() \
            <= float(p["range_band"])

        buy = in_range & (close <= ref * (1.0 - step))
        sell = (close >= ref * (1.0 + step)) | ~in_range.fillna(False)

        sig = pd.Series(0, index=df.index, dtype=int)
        sig[buy] = 1
        sig[sell] = -1
        return sig


class Consensus(Strategy):
    name = "consensus"

    DEFAULTS = {
        "drawdown_window": 96,
        "drawdown_pct": 0.10,
        "recover_ema": 12,
        "momentum_bars": 12,
        "volume_z_min": 0.3,
        "trend_sma": 200,
        "min_votes": 3,              # of 4 components
        "target_pct": 0.15,
        "stop_pct": 0.07,
    }

    def __init__(self, params=None):
        super().__init__(params)
        self.p = {**self.DEFAULTS, **(params or {})}

    def warmup_bars(self) -> int:
        return max(int(self.p["drawdown_window"]), int(self.p["trend_sma"])) + 5

    def compute_signals(self, df: pd.DataFrame, live: bool = False) -> pd.Series:
        p = self.p
        close, low = df["close"], df["low"]
        win = int(p["drawdown_window"])
        rolling_high = close.rolling(win, min_periods=1).max()
        trailing_low = low.rolling(win, min_periods=1).min().shift(1)
        drawdown = close / rolling_high - 1.0

        # Four independent votes; each filters the others' failure modes.
        v_dip = drawdown <= -float(p["drawdown_pct"])
        fast = ema(close, int(p["recover_ema"]))
        mom = close.pct_change(int(p["momentum_bars"]))
        v_recovery = (close >= fast) & (mom > 0)
        v_volume = volume_zscore(df["volume"], 48) >= float(p["volume_z_min"])
        sma_long = sma(close, int(p["trend_sma"]))
        v_not_freefall = (close > sma_long * 0.9).fillna(False) | (drawdown.diff() > 0)

        votes = (v_dip.astype(int) + v_recovery.astype(int)
                 + v_volume.astype(int) + v_not_freefall.astype(int))
        # The dip setup and the confirmed recovery are MANDATORY; the vote
        # threshold then demands broad agreement on top of them.
        extra_needed = max(0, int(p["min_votes"]) - 2)
        extras = (v_volume.astype(int) + v_not_freefall.astype(int))
        enter = v_dip & v_recovery & (extras >= extra_needed)

        target = (trailing_low > 0) & (close >= trailing_low * (1.0 + float(p["target_pct"])))
        stop = (trailing_low > 0) & (close <= trailing_low * (1.0 - float(p["stop_pct"])))
        failed = (close < fast) & (mom < 0)

        sig = pd.Series(0, index=df.index, dtype=int)
        sig[enter] = 1
        sig[target | stop | failed] = -1
        return sig


__all__ = ["DeepRecoveryV2", "AdaptiveGrid", "Consensus"]
