"""Deep-value drawdown reversion strategy (liquidity majors only).

This is the strategy family you described: scan liquid coins, buy after
a sustained drawdown (e.g. SOL -50% over a few days), and sell when they
recover. Plain "buy every big drop" is hindsight bias and buys falling
knives, so this is a disciplined, pre-registered form of it:

  1. Only act AFTER a confirmed capitulation: price has fallen a defined
     amount below its rolling high (drawdown_pct). We do not chase the
     first bar of a crash -- we wait for the heavy down-leg to register.
  2. Wait for a recovery-confirmation: after the low, momentum turns
     positive (price back above a fast EMA AND short-term return > 0).
     This filters out coins still in free-fall (where the fast EMA is
     overhead and momentum still points down).
  3. Hold long with a target/stop measured from the recent LOW, so a
     coin that keeps breaking down is cut (a new deeper low = stop),
     while a recovering coin is taken to a realised profit (price well
     above the low).

Signal semantics are fully state-free (relative to price + the trailing
low, never to the position's own entry), so the same ``compute_signals``
drives the backtest engine, the paper engine, and live execution. There
is no separate execute() path -- long/flat only, managed by the
0/1/-1 signal series (1 buys next open, -1 sells next open).

Design honesty:
  * Long horizon by construction -- the drawdown window is ~days, so this
    is a multi-day/week hold, not an intraday scalp, letting the ~1.4%
    round-trip fee amortize across the intended recovery move.
  * When no setup is present the strategy is flat and just earns the
    risk-free yield on cash (cash_yield_apy).
  * Pre-registered parameters (no tuning on the holdout).
"""
from __future__ import annotations

import pandas as pd

from bot.indicators.ta import ema
from bot.strategies.base import Strategy

DEFAULTS = {
    "drawdown_window": 96,    # bars for the rolling high/low (~4 days @1h)
    "drawdown_pct": 0.30,     # enter only after this drawdown from the high (-30%)
    "recover_ema": 24,        # fast EMA for recovery confirmation
    "momentum_bars": 24,      # window for the short-term momentum check
    "recover_prct": 0.30,     # profit target: sell after price is +30% above the low
    "stop_pct": 0.15,         # stop: sell if a new low forms that is -15% below the low
}


class DeepValueStrategy(Strategy):
    name = "deep_value"

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

        # Recovery confirmation: price back above a fast EMA and short-term
        # momentum positive => not a free-fall knife.
        fast_ema = ema(close, int(p["recover_ema"]))
        momentum = close.pct_change(int(p["momentum_bars"]))
        in_drawdown = drawdown <= -float(p["drawdown_pct"])
        recovering = (close >= fast_ema) & (momentum > 0)

        # Exits measured from the trailing low (state-free):
        target_reached = (trailing_low > 0) & (close >= trailing_low * (1.0 + float(p["recover_prct"])))
        stop_hit = (trailing_low > 0) & (close <= trailing_low * (1.0 - float(p["stop_pct"])))

        sig = pd.Series(0, index=df.index, dtype=int)
        sig[in_drawdown & recovering] = 1
        sig[target_reached | stop_hit] = -1
        # A -1 that follows no open position is harmless (base execution
        # only sells a held position); it simply keeps us flat.
        return sig


__all__ = ["DeepValueStrategy"]
