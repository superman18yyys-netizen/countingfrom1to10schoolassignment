"""Multivariate causal feature matrix for the ML trend model.

Every feature is computed on *closed candles only* and lags/pools only
past data, so nothing peeks at the present or future bar. This is what
lets the model "understand trends" rather than just fade the noise:
trend strength, momentum, volatility/regime, drawdown-from-high,
cross-sectional relative strength and calendar effects are all present
alongside the mean-reversion basics.

Reuses bot/indicators.ta for the heavy lifting and stays pandas-only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bot.indicators.ta import (atr, bollinger, donchian, ema, linreg_slope_pct,
                               macd, rsi, sma, stochastic, volume_to_volatility,
                               volume_zscore)

FEATURES_VERSION = "v1"   # bump when the feature set changes -> invalidates cache


# Union of all feature columns, kept as a documented constant.
FEATURE_COLS = [
    "ret_1", "ret_2", "ret_3", "ret_6", "ret_12", "ret_24", "ret_72",
    "rsi14", "bb_position", "atr_pct",
    "vol_z", "vvr", "ema_slope", "sma_dist",
    "roll_sharpe", "trend_frac_pos", "dd_from_high",
    "macd_hist", "stoch_k", "dc_position", "linreg_slope",
    "rel_strength", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]


def rolling_sharpe(close: pd.Series, window: int = 72) -> pd.Series:
    r = close.pct_change()
    mean = r.rolling(window).mean()
    std = r.rolling(window).std(ddof=0).replace(0.0, np.nan)
    return (mean / std).fillna(0.0)


def _calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    hour = index.hour
    dow = index.dayofweek
    out = pd.DataFrame(index=index)
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    return out


def build_features(df: pd.DataFrame,
                   cross_price: pd.Series | None = None) -> pd.DataFrame:
    """Build the full feature matrix for one pair's candles.

    ``cross_price`` is an optional reference series (e.g. BTC-USDC when
    building ETH-USDC) used for a cross-sectional relative-strength
    feature (alpha vs the market basket).
    """
    close, high, low = df["close"], df["high"], df["low"]
    vol, idx = df["volume"], df.index
    feats: dict[str, pd.Series] = {}

    for h in (1, 2, 3, 6, 12, 24, 72):
        feats[f"ret_{h}"] = close.pct_change(h)
    feats["rsi14"] = rsi(close, 14)
    feats["bb_position"] = bollinger(close, 20, 2.0)["bb_position"]
    atr_pct = atr(high, low, close, 14) / close
    feats["atr_pct"] = atr_pct
    feats["vol_z"] = volume_zscore(vol, 48)
    feats["vvr"] = volume_to_volatility(vol, atr_pct, 24)
    feats["ema_slope"] = (ema(close, 12) - ema(close, 48)) / close
    feats["sma_dist"] = (close - sma(close, 200)) / sma(close, 200)
    feats["roll_sharpe"] = rolling_sharpe(close, 72)
    feats["trend_frac_pos"] = (close.pct_change() > 0).rolling(72).mean()
    feats["dd_from_high"] = close / close.cummax() - 1.0
    feats["macd_hist"] = macd(close)["macd_hist"]
    feats["stoch_k"] = stochastic(high, low, close)["stoch_k"]
    dc = donchian(high, low, 20)
    feats["dc_position"] = ((close - dc["dc_lower"])
                            / (dc["dc_upper"] - dc["dc_lower"]).replace(0.0, np.nan))
    feats["linreg_slope"] = linreg_slope_pct(close, 48)

    if cross_price is not None:
        aligned = cross_price.reindex(idx).ffill()
        feats["rel_strength"] = (close.pct_change(72)
                                 - aligned.pct_change(72))
        feats["rel_strength"] = feats["rel_strength"].replace(0.0, np.nan)
    else:
        feats["rel_strength"] = 0.0

    cal = _calendar_features(idx)
    out = pd.DataFrame(feats, index=idx).join(cal)
    # calendar features never contain NaN; backfill the rolling head so the
    # warmup window is only as long as the widest price feature.
    out = out.replace([np.inf, -np.inf], np.nan)
    return out