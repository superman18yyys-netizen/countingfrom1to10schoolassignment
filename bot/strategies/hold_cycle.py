"""Long-horizon trend-cycle holder: few, big, slow trades.

The idea this tests (from the plan's "fewer, bigger, slower trades so
fees compound less"): most of crypto's real profit historically comes
from HOLDING through cycles, not from trading around them. So this bot
is deliberately boring:

  * A slow, persistent trend filter (price vs a long SMA over a *long*
    window on a long timeframe). It only flips position on a major,
    durable regime change -- not on noise.
  * It enters with a LARGE position fraction (one big trade, not many
    small ones) so fees are paid once per cycle, not once per scalp.
  * It earns the risk-free yield on idle cash while flat, and holds
    through intraday noise once long.

Mechanism: time-series momentum on a slow horizon is the most
evidence-backed direction signal in crypto (2020-2025). The key change
vs a normal momentum bot is the *decision cadence*: this bot is
practically buy-and-hold-with-a-filter. Few trades = few fees = fees
can't silently eat it. It will look unimpressive on a daily basis and
that is the point.

Signal contract: 1 = go long at next open, -1 = go flat at next open,
0 = do nothing. Long/flat only; cash earns cash_yield_apy.
"""
from __future__ import annotations

import pandas as pd

from bot.indicators.ta import sma
from bot.strategies.base import Strategy

DEFAULTS = {
    "trend_sma": 1000,      # very long filter window (bars)
    "enter_frac": 0.60,     # deploy 60% of equity on entry (one big trade)
    "exit_sma": 500,        # shorter exit filter (confirms trend break)
}


class HoldCycleStrategy(Strategy):
    name = "hold_cycle"

    def __init__(self, params=None):
        super().__init__(params)
        self.p = {**DEFAULTS, **(params or {})}

    def warmup_bars(self) -> int:
        return int(self.p["trend_sma"]) + 5

    def compute_signals(self, df: pd.DataFrame, live: bool = False) -> pd.Series:
        p = self.p
        close = df["close"]
        # Durable long condition: price above both a very long and a
        # shorter exit SMA -- a real regime, not a one-bar wobble.
        # Smooth both to avoid flapping at the boundary.
        long_s = sma(close, int(p["trend_sma"]))
        exit_s = sma(close, int(p["exit_sma"]))
        enter = (close > long_s) & (close > exit_s)

        # One-shot transitions (0/1/-1): buy on the FIRST bar the durable
        # trend turns positive; sell on the FIRST bar it turns negative.
        state = enter.astype(int)
        state_change = state - state.shift(1).fillna(0)   # +1 enter, -1 exit
        sig = state_change.replace({1: 1, -1: -1, 0: 0})
        return sig.copy()


__all__ = ["HoldCycleStrategy"]
