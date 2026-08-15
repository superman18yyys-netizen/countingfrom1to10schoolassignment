"""Guarded strategy wrapper: apply the lab's winning logic to any base bot.

The live + backtest findings so far point at three transferable rules:

  1. FEE DISCIPLINE: don't trade unless the expected move clearly clears
     the ~1.4% round trip. Proxy: ATR% (volatility) must exceed a hurdle,
     so we only act when moves are big enough for a win to survive fees.
  2. REGIME FILTER: only trade WITH the durable trend (momentum-style
     bases) or only inside a range (mean-reversion bases) -- never fight
     the regime. Proxied by price vs a long SMA + a short term trend.
  3. CONFIRMATION: a raw signal only becomes an action if it persists
     (signal must be true N consecutive bars), killing whipsaw churn.

The wrapper takes any base strategy class and re-emits its 1/-1 signals
after the gates. Same signal contract, same backtest engine, so guarded
and unguarded versions race honestly side by side in the zoo.

Subclasses (registered): GuardedMomentum, GuardedMACD, GuardedDonchian,
GuardedRSI2, GuardedStochastic, GuardedGrid -- all with the same shared
guard parameters, tuned per family via mode.
"""
from __future__ import annotations

import json
import os

import pandas as pd

from bot.indicators.ta import atr, sma
from bot.strategies.base import Strategy
from bot.strategies.community import (DonchianBreakout, GridTrader, MACDCross,
                                      RSI2, StochasticReversion)
from bot.strategies.momentum import MomentumStrategy

GUARD_DEFAULTS = {
    "atr_hurdle_pct": 0.008,   # only trade when ATR% >= 0.8% (move clears fees)
    "trend_sma": 200,          # regime filter window
    "confirm_bars": 2,         # signal must persist this many bars
    "mode": "trend",           # "trend": buy only above SMA; "range": only near SMA
}

_OVERRIDES_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "state", "guard_params.json")


def load_guard_overrides(path: str | None = None) -> dict:
    """Load auto-tuned guard parameters (written nightly by the auto-tuner).

    Returns {} if absent/corrupt so bots always work with defaults.
    """
    try:
        with open(path or _OVERRIDES_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        return {}


class GuardedStrategy(Strategy):
    """Gate a base strategy's signals through the shared guard rules.

    Parameters resolve in priority order: explicit params > auto-tuned
    overrides for this guard mode (state/guard_params.json, refreshed
    nightly) > research defaults. This is how the system auto-applies
    winning logic over time without code changes.
    """

    BASE = None  # subclass sets this to the base strategy class
    name = "guarded"
    MODE = "trend"

    def __init__(self, params=None):
        super().__init__(params)
        tuned = (load_guard_overrides() or {}).get(self.MODE) or {}
        self.p = {**GUARD_DEFAULTS, **tuned, **self.params, "mode": self.MODE}
        self._base = self.BASE(self.params.get("base", {}) if self.params else {})

    def warmup_bars(self) -> int:
        return max(self._base.warmup_bars(), int(self.p["trend_sma"]) + 5,
                   int(self.p["confirm_bars"]) + 2)

    def _gates(self, df: pd.DataFrame) -> pd.Series:
        """True where the guard rules ALL pass (tradable bar)."""
        p = self.p
        close, high, low = df["close"], df["high"], df["low"]
        atr_pct = atr(high, low, close, 14) / close.replace(0.0, float("nan"))
        vol_ok = atr_pct >= float(p["atr_hurdle_pct"])
        sma_long = sma(close, int(p["trend_sma"]))
        dist = (close - sma_long) / sma_long.replace(0.0, float("nan"))
        if p["mode"] == "trend":
            regime_ok = dist > 0.0                      # with the durable trend
        else:                                           # range mode
            regime_ok = dist.abs() <= 0.05              # not in a strong trend
        return (vol_ok & regime_ok).fillna(False)

    def compute_signals(self, df: pd.DataFrame, live: bool = False) -> pd.Series:
        base_sig = self._base.compute_signals(df, live=live)
        gate = self._gates(df)
        confirm = int(self.p["confirm_bars"])
        # Signal must persist `confirm` consecutive bars before it acts.
        confirmed = pd.Series(False, index=df.index)
        if confirm <= 1:
            confirmed = base_sig != 0
        else:
            for k in range(confirm):
                shifted = base_sig.shift(k).eq(base_sig)
                if k == 0:
                    confirmed = shifted
                else:
                    confirmed &= shifted
        sig = pd.Series(0, index=df.index, dtype=int)
        sig[(confirmed & (base_sig != 0)) & gate] = base_sig[(confirmed & (base_sig != 0)) & gate]
        # Keep a -1 exit live even when the regime filter fails (you can
        # always exit); the gate only filters NEW entries.
        sig[(base_sig == -1) & ~gate] = -1
        return sig


class GuardedMomentum(GuardedStrategy):
    BASE = MomentumStrategy
    name = "guarded_momentum"
    MODE = "trend"


class GuardedMACD(GuardedStrategy):
    BASE = MACDCross
    name = "guarded_macd"
    MODE = "trend"


class GuardedDonchian(GuardedStrategy):
    BASE = DonchianBreakout
    name = "guarded_donchian"
    MODE = "trend"


class GuardedRSI2(GuardedStrategy):
    BASE = RSI2
    name = "guarded_rsi2"
    MODE = "range"


class GuardedStochastic(GuardedStrategy):
    BASE = StochasticReversion
    name = "guarded_stochastic"
    MODE = "range"


class GuardedGrid(GuardedStrategy):
    BASE = GridTrader
    name = "guarded_grid"
    MODE = "range"


__all__ = ["GuardedStrategy", "GuardedMomentum", "GuardedMACD", "GuardedDonchian",
           "GuardedRSI2", "GuardedStochastic", "GuardedGrid"]
