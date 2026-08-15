"""Community-classic trading models (the "zoo").

Each class is a faithful, canonical implementation of a strategy that is
widely published and used by retail/quant communities. Parameters are
fixed at their community-standard values on purpose: the zoo experiment
measures how these published solutions perform AS-IS on live USDC data,
untuned.

Trend-following:  MACDCross, GoldenCross, DonchianBreakout
Mean-reversion:   RSI2 (Connors), StochasticReversion, GridTrader
Volatility:       BBandsBreakout (squeeze)
Accumulation:     DCABot (averaging-down with profit target + stop)
"""
from __future__ import annotations

import pandas as pd

from bot.indicators.ta import (bollinger, donchian, ema, macd, rolling_median,
                               rsi, sma, stochastic)
from bot.strategies.base import Strategy


class MACDCross(Strategy):
    """MACD(12,26,9) line/signal crossover — the most published momentum
    oscillator strategy in retail trading."""
    name = "macd_cross"

    def warmup_bars(self) -> int:
        return 40

    def compute_signals(self, df: pd.DataFrame, live: bool = False) -> pd.Series:
        m = macd(df["close"], 12, 26, 9)
        cross_up = (m["macd"] > m["macd_signal"]) & \
                   (m["macd"].shift(1) <= m["macd_signal"].shift(1))
        cross_down = (m["macd"] < m["macd_signal"]) & \
                     (m["macd"].shift(1) >= m["macd_signal"].shift(1))
        sig = pd.Series(0, index=df.index, dtype=int)
        sig[cross_up] = 1
        sig[cross_down] = -1
        return sig


class GoldenCross(Strategy):
    """SMA fast/slow golden cross — the classic long-horizon trend filter."""
    name = "golden_cross"
    DEFAULTS = {"fast": 50, "slow": 200}

    def warmup_bars(self) -> int:
        slow = int(self.params.get("slow", self.DEFAULTS["slow"]))
        return slow + 10

    def compute_signals(self, df: pd.DataFrame, live: bool = False) -> pd.Series:
        p = {**self.DEFAULTS, **self.params}
        close = df["close"]
        fast, slow = sma(close, int(p["fast"])), sma(close, int(p["slow"]))
        cross_up = (fast > slow) & (fast.shift(1) <= slow.shift(1))
        cross_down = (fast < slow) & (fast.shift(1) >= slow.shift(1))
        sig = pd.Series(0, index=df.index, dtype=int)
        sig[cross_up] = 1
        sig[cross_down] = -1
        return sig


class DonchianBreakout(Strategy):
    """Turtle Trading breakout: buy the 20-bar high breakout, exit on the
    10-bar low. One of the oldest published trend systems (1980s)."""
    name = "donchian_breakout"
    DEFAULTS = {"entry_period": 20, "exit_period": 10}

    def warmup_bars(self) -> int:
        return 25

    def compute_signals(self, df: pd.DataFrame, live: bool = False) -> pd.Series:
        p = {**self.DEFAULTS, **self.params}
        upper = donchian(df["high"], df["low"], int(p["entry_period"]))["dc_upper"]
        lower = donchian(df["high"], df["low"], int(p["exit_period"]))["dc_lower"]
        close = df["close"]
        buy = close > upper.shift(1)
        sell = close < lower.shift(1)
        sig = pd.Series(0, index=df.index, dtype=int)
        sig[buy] = 1
        sig[sell] = -1
        return sig


class RSI2(Strategy):
    """Larry Connors' RSI(2) short-term mean reversion: buy extreme
    oversold (RSI2 < 10) above the long trend, exit when RSI2 > 70.
    One of the most backtested published strategies."""
    name = "rsi2"
    DEFAULTS = {"entry_rsi": 10, "exit_rsi": 70, "trend_sma": 200}

    def warmup_bars(self) -> int:
        return int(self.DEFAULTS["trend_sma"]) + 5

    def compute_signals(self, df: pd.DataFrame, live: bool = False) -> pd.Series:
        p = {**self.DEFAULTS, **self.params}
        close = df["close"]
        r2 = rsi(close, 2)
        trend_ok = close > sma(close, int(p["trend_sma"]))
        buy = (r2 < float(p["entry_rsi"])) & trend_ok
        sell = r2 > float(p["exit_rsi"])
        sig = pd.Series(0, index=df.index, dtype=int)
        sig[buy] = 1
        sig[sell] = -1
        return sig


class StochasticReversion(Strategy):
    """Stochastic %K/%D: buy the %K-up cross inside oversold (<20), exit
    on the %K-down cross inside overbought (>80)."""
    name = "stochastic_reversion"
    DEFAULTS = {"k_period": 14, "d_period": 3, "oversold": 20, "overbought": 80}

    def warmup_bars(self) -> int:
        return 20

    def compute_signals(self, df: pd.DataFrame, live: bool = False) -> pd.Series:
        p = {**self.DEFAULTS, **self.params}
        st = stochastic(df["high"], df["low"], df["close"],
                        int(p["k_period"]), int(p["d_period"]))
        k, d = st["stoch_k"], st["stoch_d"]
        cross_up = (k > d) & (k.shift(1) <= d.shift(1))
        cross_down = (k < d) & (k.shift(1) >= d.shift(1))
        buy = cross_up & (k < float(p["oversold"]))
        sell = cross_down & (k > float(p["overbought"]))
        sig = pd.Series(0, index=df.index, dtype=int)
        sig[buy] = 1
        sig[sell] = -1
        return sig


class BBandsBreakout(Strategy):
    """Bollinger squeeze breakout: after volatility compresses (band width
    in the bottom 20th percentile of the last 100 bars), buy the break
    above the upper band; exit when price falls back below the mid band."""
    name = "bbands_breakout"
    DEFAULTS = {"bb_period": 20, "bb_std": 2.0, "squeeze_lookback": 100,
                "squeeze_pct": 0.2}

    def warmup_bars(self) -> int:
        return 130

    def compute_signals(self, df: pd.DataFrame, live: bool = False) -> pd.Series:
        p = {**self.DEFAULTS, **self.params}
        close = df["close"]
        bb = bollinger(close, int(p["bb_period"]), float(p["bb_std"]))
        width = (bb["bb_upper"] - bb["bb_lower"]) / bb["bb_mid"]
        lb = int(p["squeeze_lookback"])
        squeeze = width <= width.rolling(lb).quantile(float(p["squeeze_pct"]))
        buy = squeeze.shift(1).fillna(False) & (close > bb["bb_upper"])
        sell = close < bb["bb_mid"]
        sig = pd.Series(0, index=df.index, dtype=int)
        sig[buy] = 1
        sig[sell] = -1
        return sig


class GridTrader(Strategy):
    """Percentage grid around a rolling 24h median: buy one grid step
    below the reference, sell one step above. The canonical community
    range-trading bot (research says ~zero expectation after fees — the
    zoo tests that claim on live data)."""
    name = "grid_trader"
    DEFAULTS = {"reference_bars": 96, "grid_step": 0.015}

    def warmup_bars(self) -> int:
        return int(self.DEFAULTS["reference_bars"]) + 5

    def compute_signals(self, df: pd.DataFrame, live: bool = False) -> pd.Series:
        p = {**self.DEFAULTS, **self.params}
        close = df["close"]
        ref = rolling_median(close, int(p["reference_bars"]))
        step = float(p["grid_step"])
        buy = close <= ref * (1.0 - step)
        sell = close >= ref * (1.0 + step)
        sig = pd.Series(0, index=df.index, dtype=int)
        sig[buy] = 1
        sig[sell] = -1
        return sig


class DCABot(Strategy):
    """Dollar-cost averaging with an exit plan (stateless, restart-safe):

    * no position      -> buy the first tranche
    * price >= avg entry x (1 + profit_target) -> sell all (take profit)
    * price <= avg entry x (1 - stop_loss)     -> sell all (stop)
    * price dips dip_step below avg entry and allocation cap not reached
                         -> average down (add a tranche)

    The account must have ``allow_averaging=True``.
    """
    name = "dca_bot"
    DEFAULTS = {"profit_target": 0.02, "stop_loss": 0.08, "dip_step": 0.01,
                "max_alloc_frac": 0.85}

    def compute_signals(self, df: pd.DataFrame, live: bool = False) -> pd.Series:
        return pd.Series(0, index=df.index, dtype=int)  # order logic lives in execute()

    def execute(self, account, pair: str, df: pd.DataFrame,
                price: float, ts: int) -> dict | None:
        p = {**self.DEFAULTS, **self.params}
        pos = account.positions.get(pair)
        if pos is None:
            pos = account.open_position(pair, price, ts)
            if pos is not None:
                return {"action": "buy", "qty": pos.qty,
                        "fee": pos.entry_fee, "price": price}
            return None
        avg = pos.entry_cost / pos.qty if pos.qty else price
        if price >= avg * (1.0 + float(p["profit_target"])) or \
                price <= avg * (1.0 - float(p["stop_loss"])):
            closed = account.close_position(pair, price, ts)
            if closed is not None:
                return {"action": "sell", "qty": closed["qty"],
                        "fee": closed["exit_fee"], "price": price,
                        "pnl": closed["pnl"], "pnl_pct": closed["pnl_pct"]}
            return None
        if price <= avg * (1.0 - float(p["dip_step"])) and \
                pos.entry_cost < float(p["max_alloc_frac"]) * account.capital:
            added = account.open_position(pair, price, ts)
            if added is not None:
                return {"action": "buy", "qty": added.qty,
                        "fee": added.entry_fee, "price": price}
        return None
