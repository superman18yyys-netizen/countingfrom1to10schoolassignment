"""Event-driven backtester with honest execution modeling.

Rules enforced by construction:
  * Signals are computed on candle ``i`` (closed data only) and executed
    at the OPEN of candle ``i+1`` -- no look-ahead.
  * Every fill pays the taker fee (default 0.6%, Coinbase Advanced Trade)
    plus pessimistic slippage (default 0.1%).
  * Position sizing: fixed fraction of current equity per trade.
  * Buy & hold baseline is always computed for the same period.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from bot.strategies.base import Strategy


@dataclass
class Trade:
    pair: str
    entry_ts: pd.Timestamp
    entry_price: float
    exit_ts: pd.Timestamp
    exit_price: float
    qty: float
    entry_fee: float
    exit_fee: float
    pnl: float
    pnl_pct: float
    hold_bars: int


@dataclass
class BacktestResult:
    strategy: str
    pair: str
    start: pd.Timestamp
    end: pd.Timestamp
    equity_curve: pd.Series
    trades: List[Trade] = field(default_factory=list)
    total_return: float = 0.0
    buy_hold_return: float = 0.0
    excess_return: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    n_trades: int = 0
    avg_trade_pct: float = 0.0
    fee_take: float = 0.0

    def summary_row(self) -> dict:
        return {
            "strategy": self.strategy,
            "pair": self.pair,
            "total_return%": round(self.total_return * 100, 2),
            "buy_hold%": round(self.buy_hold_return * 100, 2),
            "excess%": round(self.excess_return * 100, 2),
            "sharpe": round(self.sharpe, 3),
            "max_dd%": round(self.max_drawdown * 100, 2),
            "win_rate%": round(self.win_rate * 100, 1),
            "n_trades": self.n_trades,
            "avg_trade%": round(self.avg_trade_pct * 100, 3),
            "fees_paid$": round(self.fee_take, 2),
        }


def _annualization_factor(bar_sec: int) -> float:
    return 365 * 86400 / bar_sec


def run_backtest(df: pd.DataFrame, strategy: Strategy, pair: Optional[str] = None,
                 taker_fee: float = 0.006, slippage: float = 0.001,
                 position_fraction: float = 0.25, max_positions: int = 3,
                 capital: float = 10_000.0, cash_yield_apy: float = 0.0) -> BacktestResult:
    """Run one strategy over one pair's candle history.

    ``cash_yield_apy`` — annualized yield earned on *idle* cash (USDC).
    This models the cost of holding cash instead of being invested, and
    the risk-free hurdle a strategy must beat. Compounded each bar on the
    cash balance. Default 0.0 keeps legacy backtests unchanged; the rest
    of the pipeline sets it to RISK_FREE_APY for the honest gate.
    """
    pair = pair or df.attrs.get("pair", "?")
    close = df["close"]
    open_ = df["open"]
    signals = strategy.compute_signals(df)
    # Optional per-entry position fractions published by the chassis
    # (decided at the signal bar, executed at the next bar's open).
    entry_fracs = getattr(strategy, "_entry_fractions", None)
    n = len(df)
    warmup = max(1, strategy.warmup_bars())

    cash = capital
    pos_qty = 0.0
    pos_entry_cost = 0.0          # fill price * qty (slippage included)
    pos_entry_fee = 0.0
    pos_entry_ts: Optional[pd.Timestamp] = None
    trades: List[Trade] = []
    equity = np.full(n, np.nan)
    equity[:max(0, warmup - 0)] = capital
    fee_take = 0.0

    bar_sec = max(60, int(round((df.index[-1] - df.index[0]).total_seconds() / max(1, n - 1))))
    yield_factor = 1.0 + bar_sec * cash_yield_apy / 31536000.0

    for i in range(warmup, n - 1):  # last signal can't execute (no next bar)
        if pd.isna(signals.iloc[i]):
            continue
        sig = int(signals.iloc[i])
        cash *= yield_factor   # idle cash earns the risk-free yield this bar
        if pos_qty > 0:
            if sig == -1:  # exit at next open
                fill = open_.iloc[i + 1] * (1.0 - slippage)
                fee = fill * pos_qty * taker_fee
                proceeds = fill * pos_qty - fee
                cash += proceeds
                fee_take += fee
                pnl = proceeds - (pos_entry_cost + pos_entry_fee)
                pnl_pct = pnl / (pos_entry_cost + pos_entry_fee) if pos_entry_cost else 0.0
                trades.append(Trade(
                    pair=pair, entry_ts=pos_entry_ts, entry_price=pos_entry_cost / pos_qty,
                    exit_ts=df.index[i + 1], exit_price=fill, qty=pos_qty,
                    entry_fee=pos_entry_fee, exit_fee=fee,
                    pnl=pnl, pnl_pct=pnl_pct,
                    hold_bars=int((df.index[i + 1] - pos_entry_ts).total_seconds() // bar_sec),
                ))
                pos_qty = 0.0
            # else: hold (signal 0/1 while in position = no action)
        else:
            if sig == 1 and cash > 0:
                pf = position_fraction
                if entry_fracs is not None:
                    f = entry_fracs.iloc[i]
                    if not pd.isna(f):
                        pf = float(f)
                size = cash * pf
                fill = open_.iloc[i + 1] * (1.0 + slippage)
                qty = size / fill
                cost = qty * fill
                fee = cost * taker_fee
                cash -= cost + fee
                fee_take += fee
                pos_qty = qty
                pos_entry_cost = cost
                pos_entry_fee = fee
                pos_entry_ts = df.index[i + 1]
        if pos_qty > 0:
            mark = close.iloc[i + 1] * (1.0 - taker_fee)
            equity[i + 1] = cash + pos_qty * mark
        else:
            equity[i + 1] = cash

    # close any open position at the last close (mark-to-market for reporting)
    equity = pd.Series(equity, index=df.index).ffill().fillna(capital)
    eq = equity.values
    total_return = eq[-1] / capital - 1.0

    bh = capital / open_.iloc[0] * close.iloc[-1] / capital - 1.0

    rets = np.diff(eq) / eq[:-1]
    sharpe = (rets.mean() / rets.std(ddof=0)) * np.sqrt(_annualization_factor(bar_sec)) if rets.std(ddof=0) > 0 else 0.0

    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    max_dd = float(-dd.min()) if len(dd) else 0.0

    closed = [t for t in trades if t.exit_ts is not None]
    n_tr = len(closed)
    win_rate = sum(1 for t in closed if t.pnl > 0) / n_tr if n_tr else 0.0
    avg_pct = np.mean([t.pnl_pct for t in closed]) if n_tr else 0.0

    result = BacktestResult(
        strategy=strategy.name, pair=pair,
        start=df.index[0], end=df.index[-1],
        equity_curve=equity, trades=closed,
        total_return=total_return, buy_hold_return=bh, excess_return=total_return - bh,
        sharpe=sharpe, max_drawdown=max_dd, win_rate=win_rate,
        n_trades=n_tr, avg_trade_pct=avg_pct, fee_take=fee_take,
    )
    return result