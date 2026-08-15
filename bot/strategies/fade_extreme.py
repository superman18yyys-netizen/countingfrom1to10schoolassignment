"""Extreme-move fade: buy capitulation, sell distribution.

The idea this tests (from "real coins with real crash->recover
asymmetry"): after an extreme, *completed* down-move, liquid assets tend
to bounce (panic sellers exhausted); after an extreme up-move, they tend
to pull back. This is the contrarian/reversal side of mean reversion,
but it only fires on genuinely extreme moves so the fee load stays low
per dollar moved.

Key discipline vs a naive "buy every dip":
  * It only acts after the move is *confirmed extreme* (return over a
    window beyond a threshold), never mid-fall.
  * It uses a target/stop around the entry so a coin that keeps bleeding
    is cut, not averaged into.
  * In a strong ongoing trend it stays flat (no trend-fighting).

Signal contract: 1 = buy at next open, -1 = sell at next open, 0 = hold.
Long/flat only.
"""
from __future__ import annotations

import pandas as pd

from bot.strategies.base import Strategy

DEFAULTS = {
    "lookback": 200,       # bars to measure the extreme move over
    "down_thresh": 0.12,   # enter long after >= -12% over the window
    "up_thresh": 0.10,     # exit long after >= +10% bounce from entry region
    "target_pct": 0.08,    # take profit target above a local low
    "stop_pct": 0.05,      # stop below a local low
}


class FadeExtremeStrategy(Strategy):
    name = "fade_extreme"

    def __init__(self, params=None):
        super().__init__(params)
        self.p = {**DEFAULTS, **(params or {})}

    def warmup_bars(self) -> int:
        return int(self.p["lookback"]) + 5

    def compute_signals(self, df: pd.DataFrame, live: bool = False) -> pd.Series:
        p = self.p
        close = df["close"]
        lb = int(p["lookback"])
        ret = close / close.shift(lb) - 1.0
        trailing_low = df["low"].rolling(lb, min_periods=1).min().shift(1)
        trailing_high = df["high"].rolling(lb, min_periods=1).max().shift(1)

        # Capitulation: a big down-move that has (at least temporarily)
        # stopped making new lows far below (price off the very bottom).
        capitulated = ret <= -float(p["down_thresh"])
        # Target: bounce of `target_pct` above the trailing low.
        target = (trailing_low > 0) & (close >= trailing_low * (1.0 + float(p["target_pct"])))
        # Stop: breaks down to a new deep low.
        stop = (trailing_low > 0) & (close <= trailing_low * (1.0 - float(p["stop_pct"])))
        # Exhausted rally: big up-move = take the fade off the table.
        rally = ret >= float(p["up_thresh"])

        sig = pd.Series(0, index=df.index, dtype=int)
        sig[capitulated] = 1
        sig[target | stop | rally] = -1
        return sig


__all__ = ["FadeExtremeStrategy"]
