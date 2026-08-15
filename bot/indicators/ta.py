"""Technical indicators (pandas/numpy only, no heavy dependencies)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI computed on closed candles only."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.fillna(50.0)


def bollinger(close: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    mid = sma(close, period)
    std = close.rolling(period).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    position = (close - lower) / (upper - lower).replace(0.0, np.nan)
    return pd.DataFrame({"bb_mid": mid, "bb_upper": upper, "bb_lower": lower,
                         "bb_position": position.fillna(0.5)})


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def volume_zscore(volume: pd.Series, window: int = 48) -> pd.Series:
    mean = volume.rolling(window).mean()
    std = volume.rolling(window).std(ddof=0).replace(0.0, np.nan)
    return ((volume - mean) / std).fillna(0.0)


def volume_to_volatility(volume: pd.Series, atr_pct: pd.Series, window: int = 24) -> pd.Series:
    """Volume-to-volatility ratio: rolling mean volume normalised by ATR%."""
    return volume.rolling(window).mean() / atr_pct.replace(0.0, np.nan)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
               k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    ll = low.rolling(k_period).min()
    hh = high.rolling(k_period).max()
    k = 100.0 * (close - ll) / (hh - ll).replace(0.0, np.nan)
    k = k.fillna(50.0)
    d = k.rolling(d_period).mean()
    return pd.DataFrame({"stoch_k": k, "stoch_d": d})


def donchian(high: pd.Series, low: pd.Series, period: int = 20) -> pd.DataFrame:
    return pd.DataFrame({
        "dc_upper": high.rolling(period).max(),
        "dc_lower": low.rolling(period).min(),
        "dc_mid": (high.rolling(period).max() + low.rolling(period).min()) / 2.0,
    })


def rolling_median(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).median()


def linreg_slope_pct(close: pd.Series, window: int = 48) -> pd.Series:
    """Rolling OLS slope of close, expressed as % price change per `window` bars."""
    y = close.values.astype(float)
    n = window
    if len(y) < n:
        return pd.Series(np.nan, index=close.index)
    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    denom = ((x - x_mean) ** 2).sum()
    slopes = np.full(len(y), np.nan)
    # vectorised weighted rolling sums; sum(w)=0 so slope = sum(w*y)/sum(w^2)
    w = x - x_mean
    yw = np.convolve(y, w[::-1], mode="valid")
    slopes[n - 1:] = yw / denom
    out = pd.Series(slopes, index=close.index)
    return out / close * n * 100.0