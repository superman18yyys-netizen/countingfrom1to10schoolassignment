"""Mean-reversion strategy on USDC pairs (RSI + Bollinger fade).

Research basis (verified Aug 2026): Yan/Huang/Wu 2026 found stablecoin-
quoted pairs (USDC/USDT/DAI) exhibit inefficiency and anti-persistence
(mean-reverting dynamics), unlike BTC/ETH. This is the structural
tailwind for a "buy low, sell high" range strategy on USDC pairs:

* Buy  when RSI < oversold AND close <= lower Bollinger band.
* Exit when RSI recovers above exit_rsi (the fade has completed) or
  the price reaches the upper band.
"""
from __future__ import annotations

import pandas as pd

from bot.indicators.ta import bollinger, rsi
from bot.strategies.base import Strategy


class MeanReversionStrategy(Strategy):
    name = "mean_reversion"

    DEFAULTS = {
        "rsi_period": 14, "bb_period": 20, "bb_std": 2.0,
        "oversold": 30, "overbought": 70, "exit_rsi": 55,
    }

    def warmup_bars(self) -> int:
        return max(int(self.DEFAULTS["bb_period"]), int(self.DEFAULTS["rsi_period"]))

    def compute_signals(self, df: pd.DataFrame, live: bool = False) -> pd.Series:
        p = {**self.DEFAULTS, **self.params}
        close = df["close"]
        rsi_s = rsi(close, int(p["rsi_period"]))
        bb = bollinger(close, int(p["bb_period"]), float(p["bb_std"]))

        buy = (rsi_s < float(p["oversold"])) & (close <= bb["bb_lower"])
        sell = (rsi_s > float(p["exit_rsi"])) | \
               ((rsi_s > float(p["overbought"])) & (close >= bb["bb_upper"]))

        sig = pd.Series(0, index=df.index, dtype=int)
        sig[buy] = 1
        sig[sell] = -1
        return sig