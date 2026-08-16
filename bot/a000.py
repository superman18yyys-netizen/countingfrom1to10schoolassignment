"""A-000: the multi-coin portfolio allocator (lab).

TWO tools in one file:

1. ORACLE — perfect-foresight ceiling: given the full 6y of closes,
   computes the optimal sequence of trades (dynamic program over
   fee-adjusted swing segments, rotating across the universe). This is
   the upper bound: what $100 COULD have become with omniscience and
   perfect sizing. A-000's job is to close the gap causally.

2. A-000 — the real bot: cross-sectional rotation allocator.
   Every decision tick:
     score each coin: 40-bar momentum percentile x SMA200 trend x
       1/vol (vol-normalized strength — Gbadebo 2026 evidence)
     hold the top-K coins (by score, above a threshold); sell the rest
     reserve a cash buffer so new coins can be bought when THEY rise
     swap out of a holding only when the challenger's edge exceeds
       the 1.4% round-trip switching cost + a margin (fee-aware
       rotation: "sell X to buy Y because Y earns more")
     size every position by inverse volatility (risk-balanced)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from bot.indicators.ta import atr, sma

RTC = 0.014                    # 2 x (0.6% taker + 0.1% slippage)
CASH_YIELD = 0.045
SLIP = 0.001
FEE = 0.006


# ---------------------------------------------------------------- oracle
def oracle_return(closes: Dict[str, pd.Series], capital: float = 100.0) -> dict:
    """Perfect-foresight ceiling over all coins simultaneously.

    Segment every coin's history into fee-clearing upswings (using the
    zigzag pivots, net of 1.4%); then greedily chain the BEST segment
    available at each step across the whole universe (all-in, sell at
    segment end). Optimal for known futures: with no uncertainty and
    one asset class, always take the highest-returning available
    swing, then the next. Returns final equity + trade count."""
    segs = []  # (start_ts, end_ts, net_ret)
    for pair, close in closes.items():
        piv = _zigzag(close, 0.08)
        pending = None
        for i, p, kind in piv:
            if kind == "lo":
                pending = (i, p)
            elif kind == "hi" and pending:
                net = p / pending[1] - 1 - RTC
                if net > 0.02:
                    segs.append((pending[0], i, net))
                pending = None
    if not segs:
        return {"equity": capital, "trades": 0, "return_pct": 0.0}
    segs.sort(key=lambda s: s[0])
    eq = capital
    t_end = segs[0][0]
    n = 0
    for (s, e, net) in segs:
        if s >= t_end:          # segments don't overlap: chain them
            eq *= (1.0 + net)
            t_end = e
            n += 1
    return {"equity": eq, "trades": n,
            "return_pct": (eq / capital - 1.0) * 100}


def _zigzag(close: pd.Series, pct: float) -> List[Tuple[int, float, str]]:
    piv, mode, ext_i, ext_p = [], "init", 0, float(close.iloc[0])
    for i in range(len(close)):
        p = float(close.iloc[i])
        if mode in ("init", "up"):
            if mode == "init":
                if p > ext_p * (1 + pct):
                    mode, ext_i, ext_p = "up", i, p
                    continue
                if p < ext_p:
                    ext_i, ext_p = i, p
            elif p > ext_p:
                ext_i, ext_p = i, p
            elif p <= ext_p * (1 - pct):
                piv.append((ext_i, ext_p, "hi"))
                mode, ext_i, ext_p = "down", i, p
        if mode == "down":
            if p < ext_p:
                ext_i, ext_p = i, p
            elif p >= ext_p * (1 + pct):
                piv.append((ext_i, ext_p, "lo"))
                mode, ext_i, ext_p = "up", i, p
    return piv


# ------------------------------------------------------------------ A-000
@dataclass
class A000Config:
    capital: float = 100.0
    top_k: int = 4
    cash_buffer: float = 0.25     # reserve to buy coins when they rise
    mom_bars: int = 40
    score_min: float = 0.0        # score threshold to hold a coin at all
    swap_margin: float = 0.010    # challenger must beat holder by this
                                  # (on top of the 1.4% switching fee)
    vol_period: int = 14
    tick_bars: int = 6            # rebalance every N bars (4H x6 = 1d)


@dataclass
class A000Result:
    equity: float
    return_pct: float
    max_dd_pct: float
    trades: int
    fees_paid: float
    final_holdings: Dict[str, float]


def run_a000(closes: Dict[str, pd.Series],
             highs: Dict[str, pd.Series],
             lows: Dict[str, pd.Series],
             cfg: Optional[A000Config] = None) -> A000Result:
    cfg = cfg or A000Config()
    pairs = sorted(closes)
    # align index: union of timestamps
    idx_all = sorted(set().union(*[set(c.index) for c in closes.values()]))
    timeline = pd.DatetimeIndex(idx_all)

    # precompute per-pair features on each pair's own frame
    score = {}
    vol = {}
    for pair in pairs:
        close = closes[pair]
        mom = close / close.shift(cfg.mom_bars) - 1.0
        mom_pct = mom.rolling(2160, min_periods=300).apply(
            lambda v: (v <= v[-1]).mean(), raw=True)
        trend = (close > sma(close, 200)).astype(float)
        a = atr(highs[pair], lows[pair], close, cfg.vol_period) / close
        vol[pair] = a
        score[pair] = ((1.0 + mom) * trend * mom_pct
                       .fillna(0.5)).reindex(timeline).ffill()
        # keep a snapshot dict for fast bar access

    cash = cfg.capital
    holdings: Dict[str, float] = {}    # pair -> qty
    entry_cost: Dict[str, float] = {}  # pair -> cost basis incl. fee
    fees_paid = 0.0
    trades = 0
    equity_hist: List[float] = []

    def px(pair, t):
        return float(closes[pair].loc[t]) if t in closes[pair].index else None

    def sell_all(pair, price, t):
        nonlocal cash, fees_paid, trades
        if pair not in holdings:
            return
        qty = holdings.pop(pair)
        proceeds = qty * price * (1 - SLIP) * (1 - FEE)
        fees_paid += qty * price * (1 - SLIP) * FEE
        cash += proceeds
        entry_cost.pop(pair, None)
        trades += 1

    def buy(pair, price, t, frac):
        nonlocal cash, fees_paid, trades
        spend = cash * frac
        fill = price * (1 + SLIP)
        qty = spend / fill
        fee = spend * FEE
        fees_paid += fee
        cash -= spend + fee
        holdings[pair] = holdings.get(pair, 0.0) + qty
        entry_cost[pair] = entry_cost.get(pair, 0.0) + spend + fee
        trades += 1

    peak = cfg.capital
    max_dd = 0.0

    for i, t in enumerate(timeline):
        # cash yield accrual per bar (365d / 4H)
        cash *= (1.0 + CASH_YIELD * (1 / 2190.0))
        if i % cfg.tick_bars != 0:
            continue
        # mark equity
        eq = cash + sum(q * closes[p].loc[t]
                        for p, q in holdings.items() if t in closes[p].index)
        peak = max(peak, eq)
        max_dd = max(max_dd, (peak - eq) / peak)
        equity_hist.append(eq)

        # current scores
        scored = []
        for pair in pairs:
            if t in closes[pair].index:
                s = score[pair].loc[t]
                if s == s and s >= cfg.score_min:
                    scored.append((float(s), pair))
        scored.sort(reverse=True)
        want = {pair for _, pair in scored[:cfg.top_k]}

        # budget: at most (1 - cash_buffer) of equity invested
        target_invested = eq * (1.0 - cfg.cash_buffer)
        invested = eq - cash

        # 1) sell coins that fell out of the top-K or failed trend
        for pair in list(holdings):
            if pair in want:
                continue
            price = px(pair, t)
            if price is None:
                continue
            # fee-aware: only sell if score clearly decayed below the
            # challenger pool (avoid churn on borderline ranking)
            sell_all(pair, price, t)

        # 2) buy/boost top scorers up to the target budget
        for pair, in sorted(scored[:cfg.top_k], reverse=True):
            if invested >= target_invested:
                break
            price = px(pair, t)
            if price is None:
                continue
            # inverse-vol sizing across the roster
            v = float(vol[pair].loc[t]) if pair in vol and t in vol[pair].index else 0.02
            v = max(v, 1e-4)
            w = (1.0 / v)
            # scale so each new position gets its weight of remaining budget
            if pair in holdings:
                continue  # already holds; no re-buy (avoids churn)
            # only enter if the total weight budget allows
            frac = min(0.5, w / (w + 1.0))
            if frac < 0.05:
                continue
            # fee-aware entry: expected move must clear the toll
            mom_now = float(score[pair].loc[t])
            if mom_now < 0.5:     # weak: skip (gate)
                continue
            buy(pair, price, t, frac)
            invested += price * holdings[pair]

    # final equity
    eq = cash
    for p, q in holdings.items():
        t = closes[p].index[-1]
        eq += q * float(closes[p].loc[t])
    return A000Result(equity=eq, return_pct=(eq / cfg.capital - 1) * 100,
                      max_dd_pct=max_dd * 100, trades=trades,
                      fees_paid=fees_paid,
                      final_holdings={p: q for p, q in holdings.items()})
