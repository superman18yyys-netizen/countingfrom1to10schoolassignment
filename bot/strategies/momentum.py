"""Time-series momentum / trend-following strategy.

Research basis (verified Aug 2026): time-series momentum was the most
evidence-backed direction strategy on crypto 2020-2025 (Gbadebo 2026,
31.96% annualized TS momentum; Han/Kang/Ryu 2024 - only net of realistic
costs). Strategy:

* Long when EMA(fast) crosses above EMA(slow) AND close is above the
  long trend EMA (regime filter).
* Exit when EMA(fast) crosses below EMA(slow).
"""
from __future__ import annotations

import pandas as pd

from bot.indicators.ta import ema
from bot.strategies.base import Strategy


class MomentumStrategy(Strategy):
    name = "momentum"

    DEFAULTS = {"ema_fast": 12, "ema_slow": 26, "trend_ema": 200}

    def warmup_bars(self) -> int:
        return int(self.params.get("trend_ema", self.DEFAULTS["trend_ema"]))

    def compute_signals(self, df: pd.DataFrame, live: bool = False) -> pd.Series:
        p = {**self.DEFAULTS, **self.params}
        close = df["close"]
        fast = ema(close, int(p["ema_fast"]))
        slow = ema(close, int(p["ema_slow"]))
        trend = ema(close, int(p["trend_ema"]))

        cross_up = (fast > slow) & (fast.shift(1) <= slow.shift(1))
        cross_down = (fast < slow) & (fast.shift(1) >= slow.shift(1))
        regime_ok = close > trend

        sig = pd.Series(0, index=df.index, dtype=int)
        sig[cross_up & regime_ok] = 1
        sig[cross_down] = -1
        return sig