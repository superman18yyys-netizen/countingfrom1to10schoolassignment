"""MarketContext: the "look at the year first" layer (chassis layer 1).

Vectorized, causal per-bar market context for one pair: where price
sits in its trailing range, trend state, volatility regime, drawdown.
Feeds the chassis regime gate and conviction sizing. Degrades
gracefully on short windows (CI cold-start) — nothing crashes,
conviction simply scales down with context_confidence.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from bot.indicators.ta import atr, sma

# Frozen chassis constants (see chassis design doc — NOT tunable).
CONTEXT_FULL_BARS = 8760        # 1y of 1h bars = full confidence
SMA_WINDOW = 200
SLOPE_BARS = 50
CRASH_DD = 0.25                 # drawdown from 1y high beyond this...
CRASH_ATR_PCTL = 0.90           # ...AND ATR in top decile => CRASH
TREND_BAND = 0.03               # +/-3% vs SMA200 defines UP/DOWN

# Regime codes (ints keep the vectorized path cheap; names for humans)
RANGE, UP, DOWN, CRASH = 0, 1, -1, -2
REGIME_NAMES = {RANGE: "RANGE", UP: "UP", DOWN: "DOWN", CRASH: "CRASH"}


def build_context(df: pd.DataFrame) -> pd.DataFrame:
    """Per-bar context for one pair. Causal: every column at bar i uses
    data up to and including bar i only."""
    close, high, low = df["close"], df["high"], df["low"]
    n = len(df)
    out = pd.DataFrame(index=df.index)

    atr14 = atr(high, low, close, 14)
    out["atr_pct"] = (atr14 / close.replace(0.0, np.nan)).fillna(0.0)

    sma_long = sma(close, SMA_WINDOW)
    out["sma_dist"] = ((close - sma_long) / sma_long.replace(0.0, np.nan)
                       ).fillna(0.0)

    # normalized linear-regression slope over SLOPE_BARS (fit on the
    # window ending at each bar — causal rolling apply)
    out["slope"] = close.rolling(SLOPE_BARS).apply(
        _norm_slope, raw=True).fillna(0.0)

    roll_max = close.rolling(CONTEXT_FULL_BARS, min_periods=30).max()
    out["dd_from_high"] = ((close - roll_max) / roll_max.replace(
        0.0, np.nan)).fillna(0.0)

    out["atr_pctile"] = out["atr_pct"].rolling(90, min_periods=20).apply(
        _pctile_rank, raw=True).fillna(0.5)

    # Long-window volatility regime (90 days): the empirical Aug 2026
    # analysis showed that when ATR sits in its lowest quartile vs the
    # trailing 90 days, only ~38% of 16-bar moves clear the 1.4% toll
    # (vs ~57% at normal vol) — the fee gate's EV assumption breaks in
    # droughts. The chassis blocks entries during droughts.
    out["atr_pctile_long"] = out["atr_pct"].rolling(
        2160, min_periods=500).apply(_pctile_rank, raw=True).fillna(0.5)

    ret_pos = (close.diff() > 0).astype(float)
    out["trend_frac_pos"] = ret_pos.rolling(SLOPE_BARS).mean().fillna(0.5)

    roll_min = close.rolling(CONTEXT_FULL_BARS, min_periods=30).min()
    rng = (roll_max - roll_min).replace(0.0, np.nan)
    out["pct_rank_1y"] = ((close - roll_min) / rng).clip(0.0, 1.0).fillna(0.5)

    out["context_confidence"] = min(1.0, n / CONTEXT_FULL_BARS)
    return out


def _norm_slope(values: np.ndarray) -> float:
    """Slope of a least-squares line fit, normalized by mean price."""
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    slope = np.polyfit(x, values, 1)[0]
    mean = values.mean()
    return slope / mean if mean > 0 else 0.0


def _pctile_rank(values: np.ndarray) -> float:
    """Percentile (0..1) of the LAST value within the window."""
    last = values[-1]
    return float((values <= last).mean())


def classify_regime(ctx: pd.DataFrame) -> pd.Series:
    """Deterministic regime per bar: CRASH > UP/DOWN > RANGE."""
    crash = (ctx["dd_from_high"] < -CRASH_DD) & \
            (ctx["atr_pctile"] > CRASH_ATR_PCTL)
    up = (ctx["sma_dist"] > TREND_BAND) & (ctx["slope"] > 0)
    down = (ctx["sma_dist"] < -TREND_BAND) & (ctx["slope"] < 0)
    reg = pd.Series(RANGE, index=ctx.index, dtype=int)
    reg[up] = UP
    reg[down] = DOWN
    reg[crash] = CRASH       # checked last = highest priority
    return reg


def regime_row(ctx: pd.DataFrame, i: int = -1) -> Dict[str, float]:
    """Context snapshot at bar i for scalar decisions."""
    row = ctx.iloc[i]
    return {k: (0.0 if pd.isna(v) else float(v)) for k, v in row.items()}
