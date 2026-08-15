"""Aggressive deep-dip recovery (variant of deep_value).

The idea this tests: the plan's "real coins with real crash->recover
asymmetry" -- but this is the tuned-to-fire version. deep_value was so
conservative it sat out most of the year (only SOL/ADA crashed enough to
trigger it). This bot keeps the same *mechanism* (buy a confirmed dip,
sell on recovery) but with looser thresholds so it actually engages more
often:

  * Lower drawdown bar (-12% vs -30%): more coins qualify more often.
  * Faster recovery EMA + shorter drawdown window: catches sharper, more
    frequent dips.
  * Tighter stop: high-knife-risk trades get cut fast.

This directly tests whether a *more active* dip-selection clears fees
net -- the whole open question from the deep_value backtest. If this
fires a lot and still clears fees on the holdout, that is a real finding;
if it bleeds fees on many small whipsaws, that is also a real finding
(and tells us the conservative version was right).

Signal contract: 1 = buy at next open, -1 = sell at next open, 0 = hold.
"""
from __future__ import annotations

import pandas as pd

from bot.indicators.ta import ema
from bot.strategies.base import Strategy

DEFAULTS = {
    "drawdown_window": 48,    # ~2 days @1h
    "drawdown_pct": 0.12,     # enter after -12% from the high
    "recover_ema": 9,         # fast recovery confirmation
    "momentum_bars": 6,
    "target_pct": 0.12,
    "stop_pct": 0.06,
}


class DeepRecoveryStrategy(Strategy):
    name = "deep_recovery"

    def __init__(self, params=None):
        super().__init__(params)
        self.p = {**DEFAULTS, **(params or {})}

    def warmup_bars(self) -> int:
        return int(self.p["drawdown_window"]) + int(self.p["recover_ema"]) + 5

    def compute_signals(self, df: pd.DataFrame, live: bool = False) -> pd.Series:
        p = self.p
        win = int(p["drawdown_window"])
        close = df["close"]
        low = df["low"]

        rolling_high = close.rolling(win, min_periods=1).max()
        trailing_low = low.rolling(win, min_periods=1).min().shift(1)
        drawdown = close / rolling_high - 1.0

        fast_ema = ema(close, int(p["recover_ema"]))
        momentum = close.pct_change(int(p["momentum_bars"]))
        in_dip = drawdown <= -float(p["drawdown_pct"])
        recovering = (close >= fast_ema) & (momentum > 0)

        target = (trailing_low > 0) & (close >= trailing_low * (1.0 + float(p["target_pct"])))
        stop = (trailing_low > 0) & (close <= trailing_low * (1.0 - float(p["stop_pct"])))

        sig = pd.Series(0, index=df.index, dtype=int)
        sig[in_dip & recovering] = 1
        sig[target | stop] = -1
        return sig


__all__ = ["DeepRecoveryStrategy"]
