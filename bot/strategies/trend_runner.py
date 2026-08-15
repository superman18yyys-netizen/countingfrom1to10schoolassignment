"""TrendRunner: long/flat trend-following with a trailing, volatility-scaled
exit — the shape most likely to clear fees and beat buy & hold.

Why this bot is designed the way it is (grounded in the lab's findings):

  * FEES DOMINATE (~1.2% round trip). Every prior postmortem lost on Coinbase
    fees. TrendRunner therefore (a) only enters when realized volatility
    (ATR%) clears a fee hurdle — if the move isn't big enough for a winner to
    survive the round trip, it stays in cash — and (b) trades rarely: long/flat
    means one entry, one exit per cycle, no mean-reversion churn.
  * THE PAYOFF SHAPE that beats buy & hold is asymmetric trend capture, not
    high win rate. A chandelier-style trailing stop (rolling-high minus k*ATR)
    lets winners run and cuts losers early. Holds ride big up-moves, losers get
    stopped out quickly — which is exactly how momentum systems beat a 40%
    drawdown in the 1y backtest while mean-reversion merely dodged it.
  * REGIME FILTER: only go long in a durable uptrend (price above a long SMA).
    It never fights the trend, and never "averages" into a falling knife.

Signal contract matches the engine: 1 = enter on next open, -1 = exit on next
open, 0 = hold. Stateless and deterministic: every signal derives from closed
candles via rolling windows, so the same backtest engine that runs the other
zoo bots races it honestly with no look-ahead.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from bot.indicators.ta import atr, sma
from bot.strategies.base import Strategy

_TUNED_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "state", "runner_params.json")


def _load_tuned(path: str | None = None) -> dict:
    """Auto-tuned trend_runner params (written nightly by the auto-tuner).

    Accepts either a flat params dict or the tuner's envelope format
    {"params": {...}, "score": ...}. Returns {} if absent/corrupt so the
    bot always works with defaults.
    """
    try:
        with open(path or _TUNED_PATH, "r", encoding="utf-8") as fh:
            d = json.load(fh)
    except Exception:  # noqa: BLE001
        return {}
    if isinstance(d, dict) and isinstance(d.get("params"), dict):
        d = d["params"]
    return d if isinstance(d, dict) else {}


class TrendRunner(Strategy):
    name = "trend_runner"

    DEFAULTS = {
        "trend_sma": 100,        # regime filter: only long above this SMA
        "atr_period": 14,        # ATR window
        "atr_mult": 3.0,         # trailing stop width = atr_mult * ATR
        "trail_bars": 96,        # rolling-high window for the chandelier reference (4d @1h)
        "atr_hurdle_pct": 0.005, # min ATR% to enter (moves that clear the ~1.4% round trip)
    }

    def __init__(self, params=None, tuned_path: str | None = None):
        super().__init__(params)
        # Priority: explicit params (swarm mutation / tests) > auto-tuned
        # overrides (nightly tuner) > research defaults.
        tuned = _load_tuned(tuned_path) or {}
        merged = {**self.DEFAULTS, **tuned}
        merged.update(self.params or {})
        self.p = merged

    def warmup_bars(self) -> int:
        return max(int(self.p["trail_bars"]), int(self.p["trend_sma"])) \
            + int(self.p["atr_period"]) + 5

    def compute_signals(self, df: pd.DataFrame, live: bool = False) -> pd.Series:
        p = self.p
        close, high, low = df["close"], df["high"], df["low"]

        rolling_high = high.rolling(int(p["trail_bars"]), min_periods=1).max().shift(1)
        atr_series = atr(high, low, close, int(p["atr_period"]))
        atr_pct = (atr_series / close.replace(0.0, np.nan)).fillna(0.0)
        trail_stop = rolling_high - float(p["atr_mult"]) * atr_series

        sma_trend = sma(close, int(p["trend_sma"]))
        regime_ok = close > sma_trend
        vol_ok = atr_pct >= float(p["atr_hurdle_pct"])

        # Enter ONCE going long: a clean cross above the trend SMA (regime
        # shift), gated by fee-clearing volatility. A one-bar crossing pulse
        # avoids both look-ahead and re-firing every bar of an uptrend; the
        # account's single-position guard ignores duplicates anyway.
        enter = (close > sma_trend) & (close.shift(1) <= sma_trend.shift(1))
        enter &= vol_ok
        # Exit: close below the chandelier trailing stop (let winners run,
        # cut losers), or a hard regime break below the trend SMA.
        exit_ = (close < trail_stop).fillna(False) \
            | (~regime_ok & (close < sma_trend.shift(1))).fillna(False)

        sig = pd.Series(0, index=df.index, dtype=int)
        sig[enter.fillna(False)] = 1
        sig[exit_.fillna(False)] = -1
        return sig


__all__ = ["TrendRunner"]
