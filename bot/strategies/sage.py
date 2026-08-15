"""Sage: the expert decision-maker.

Design contract (the "trader who has seen everything up to now"):
at every bar it receives the FULL window of past candles ending at
the current bar — exactly the live ``execute`` contract — and makes
one of three decisions: BUY, HOLD, or SELL. No future data, no
vectorized shortcuts at decision time.

Decision = weighted evidence panel. Each witness votes from the past
window; Sage buys when enough independent witnesses agree, sells when
agreement collapses:

  trend     price above EMA200 AND EMA12>EMA26 (durable uptrend)
  dip       RSI14 below threshold (stretched to the downside)
  panic     RSI14 in capitulation zone (extra weight)
  value     price in bottom quartile of its 1-year range
  band      close below lower Bollinger band (statistically cheap)
  momentum  sign of the 24-bar return (flow direction)

Weights are FIXED from evidence (research + 2y data analysis); only
the two decision thresholds are ever tuned, inside a small capped
grid, walk-forward validated (Bailey/Lopez de Prado overfit guard).

The chassis still wraps Sage: regime allowlist, vol-drought gate,
fee EV gate, disaster/time stops, vol-target sizing. Sage is the
brain; the chassis is the discipline.
"""
from __future__ import annotations

import pandas as pd

from bot.indicators.ta import bollinger, ema, rsi
from bot.strategies.base import Strategy


class SageStrategy(Strategy):
    name = "sage"

    DEFAULTS = {
        "trend_ema": 200,
        "fast": 12, "slow": 26,
        "rsi_period": 14,
        "dip_rsi": 32, "panic_rsi": 24,
        "bb_period": 20, "bb_std": 2.0,
        "mom_bars": 24,
        "buy_score": 2.0,     # evidence needed to buy (tunable, capped grid)
        "sell_score": 0.0,    # evidence at/below which Sage sells (tunable)
    }

    # fixed witness weights (NOT tuned — by design)
    W = {"trend": 1.0, "dip": 1.5, "panic": 0.5, "value": 0.5,
         "band": 1.0, "momentum": 0.5}

    def warmup_bars(self) -> int:
        return int(self.DEFAULTS["trend_ema"]) + 10

    def score_series(self, df: pd.DataFrame) -> pd.Series:
        """Evidence score per bar (vectorized for TRAINING only; the
        live decision path calls this on the closed window each bar —
        identical values, computed causally)."""
        p = {**self.DEFAULTS, **self.params}
        close = df["close"]

        ema_t = ema(close, int(p["trend_ema"]))
        ema_f = ema(close, int(p["fast"]))
        ema_s = ema(close, int(p["slow"]))
        trend = ((close > ema_t) & (ema_f > ema_s)).astype(float)

        r14 = rsi(close, int(p["rsi_period"]))
        dip = (r14 < float(p["dip_rsi"])).astype(float)
        panic = (r14 < float(p["panic_rsi"])).astype(float)

        roll_max = close.rolling(2160, min_periods=100).max()
        roll_min = close.rolling(2160, min_periods=100).min()
        rng = (roll_max - roll_min)
        rank = ((close - roll_min) / rng.replace(0.0, pd.NA)).clip(0, 1)
        value = (rank < 0.25).astype(float)

        bb = bollinger(close, int(p["bb_period"]), float(p["bb_std"]))
        band = (close < bb["bb_lower"]).astype(float)

        momo = (close / close.shift(int(p["mom_bars"])) - 1.0).apply(
            lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0))

        w = self.W
        return (w["trend"] * trend + w["dip"] * dip + w["panic"] * panic
                + w["value"] * value + w["band"] * band
                + w["momentum"] * momo)

    def compute_signals(self, df: pd.DataFrame, live: bool = False) -> pd.Series:
        p = {**self.DEFAULTS, **self.params}
        score = self.score_series(df)
        sig = pd.Series(0, index=df.index, dtype=int)
        sig[score >= float(p["buy_score"])] = 1
        sig[score <= float(p["sell_score"])] = -1
        return sig
