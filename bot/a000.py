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

from bot.indicators.ta import atr, rsi, sma

RTC = 0.014                    # 2 x (0.6% taker + 0.1% slippage)
CASH_YIELD = 0.045
SLIP = 0.001
FEE = 0.006

# feature memoization: features are pure functions of the data; the
# walk-forward loop re-runs run_a000 with the same data and only
# different trade windows — recomputing 200 coins x 8 rolling features
# per trial would take hours. Keyed on the identity of the input dicts.
_FEAT_CACHE: dict = {}


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
    mom_bars: int = 540           # 90-day momentum lookback (4H bars)
    sell_slack: int = 2           # holding survives while in top K+slack
    swap_margin: float = 0.02     # challenger must beat holder by this
                                  # raw momentum (on top of 1.4% fees)
    vol_period: int = 14
    tick_bars: int = 42           # rebalance weekly (42 x 4H)
    crash_filter: bool = False    # SMA200 crash filter. Default OFF:
                                  # tested 1.0/0.95/0.9/0.85 bands — all
                                  # whipsaw more value than they save
                                  # (results non-monotonic, +13..+150%
                                  # vs +189% without)
    trade_from: int = 0           # timeline position where trading may
    trade_to: int = 10**9         # begin/end (walk-forward spans;
                                  # features always use full history)
    core_frac: float = 0.65       # core-satellite: fraction of the
                                  # budget always held in BTC+ETH
                                  # (0 = pure rotation). Core sells only
                                  # when its own momentum turns negative.
    core_pairs: tuple = ("BTC-USDC", "ETH-USDC")
    oracle_filter: bool = True    # entry filter mined from the optimal
                                  # path (reports/oracle-rules.json):
                                  # mom_40 >= 0, atr_pctile >= 0.40,
                                  # rsi14 in [30, 65], dd > -0.75,
                                  # micro-pullback not surge-chase


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

    # feature memoization: keyed on data identity + mom_bars (the only
    # feature that depends on cfg besides vol_period)
    ckey = (tuple(id(closes[p]) for p in pairs), cfg.mom_bars)
    if ckey in _FEAT_CACHE:
        (mom, vol, sma200, mom40, rsi14, atr_pctl, surge12,
         dd1y, timeline_c) = _FEAT_CACHE[ckey]
    else:
        # precompute per-pair risk-adjusted momentum on each frame
        mom, vol = {}, {}
        for pair in pairs:
            close = closes[pair]
            m = close / close.shift(cfg.mom_bars) - 1.0
            r = close.pct_change()
            v = r.rolling(cfg.mom_bars).std(ddof=0)
            mom[pair] = (m / v.replace(0.0, np.nan)) \
                .reindex(timeline).ffill()
            a = atr(highs[pair], lows[pair], close, cfg.vol_period) / close
            vol[pair] = a.reindex(timeline).ffill()
        sma200 = {p: sma(closes[p], 200).reindex(timeline).ffill()
                  for p in pairs}
        mom40, rsi14, atr_pctl, surge12, dd1y = {}, {}, {}, {}, {}
        for pair in pairs:
            close = closes[pair]
            mom40[pair] = (close / close.shift(40) - 1.0) \
                .reindex(timeline).ffill()
            rsi14[pair] = rsi(close, 14).reindex(timeline).ffill()
            a = atr(highs[pair], lows[pair], close, 14) / close
            atr_pctl[pair] = a.rolling(540, min_periods=100).apply(
                lambda v: (v <= v[-1]).mean(), raw=True) \
                .reindex(timeline).ffill()
            surge12[pair] = (close / close.shift(12) - 1.0) \
                .reindex(timeline).ffill()
            dd1y[pair] = (close / close.rolling(2160, min_periods=100).max()
                          - 1.0).reindex(timeline).ffill()
        _FEAT_CACHE[ckey] = (mom, vol, sma200, mom40, rsi14,
                             atr_pctl, surge12, dd1y, timeline)

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
        # mark equity
        eq = cash + sum(q * closes[p].loc[t]
                        for p, q in holdings.items() if t in closes[p].index)
        peak = max(peak, eq)
        max_dd = max(max_dd, (peak - eq) / peak)
        equity_hist.append(eq)

        if i % cfg.tick_bars != 0:
            continue

        trading = cfg.trade_from <= i <= cfg.trade_to
        if not trading:
            continue

        # -------- crash filter: long-trend breaks close positions -----
        if cfg.crash_filter:
            for pair in list(holdings):
                if t not in closes[pair].index:
                    continue
                price = float(closes[pair].loc[t])
                if price < float(sma200[pair].loc[t]):
                    sell_all(pair, price, t)

        # current scores (risk-adjusted momentum); only positive
        # absolute momentum is buyable (absolute filter), plus the
        # oracle-mined entry rules when enabled
        scored = []
        for pair in pairs:
            if t in closes[pair].index:
                s = mom[pair].loc[t]
                if s == s:
                    scored.append((float(s), pair))
        scored.sort(reverse=True)
        buyable = []
        for s, pair in scored:
            if s <= 0:
                continue
            if cfg.oracle_filter and pair not in cfg.core_pairs:
                m40 = mom40[pair].loc[t]
                r14 = rsi14[pair].loc[t]
                ap = atr_pctl[pair].loc[t]
                dd = dd1y[pair].loc[t]
                ok = (m40 == m40 and m40 >= 0.0
                      and r14 == r14 and 30.0 <= r14 <= 65.0
                      and ap == ap and ap >= 0.40
                      and dd == dd and dd > -0.75)
                if not ok:
                    continue
            buyable.append((s, pair))

        # risk-parity target weights: core (BTC/ETH anchor) + satellite
        # (top-K alts by momentum, excluding core pairs)
        budget = eq * (1.0 - cfg.cash_buffer)
        target_val: Dict[str, float] = {}
        if cfg.core_frac > 0:
            core_ok = [p for p in cfg.core_pairs
                       if p in closes and t in closes[p].index
                       and (lambda s: s == s and s > 0)(mom[p].loc[t])]
            if core_ok:
                each = budget * cfg.core_frac / len(core_ok)
                for p in core_ok:
                    target_val[p] = each
        sat_budget = budget * (1.0 - cfg.core_frac)
        sat = [(s, p) for s, p in buyable[:cfg.top_k]
               if p not in cfg.core_pairs]
        vols = {}
        for _, pair in sat:
            v = float(vol[pair].loc[t]) if (pair in vol
                                            and t in vol[pair].index) else 0.02
            vols[pair] = max(v, 1e-4)
        inv = {p: 1.0 / v for p, v in vols.items()}
        inv_sum = sum(inv.values()) or 1.0
        for p in vols:
            target_val[p] = target_val.get(p, 0.0) + \
                sat_budget * (inv[p] / inv_sum)

        # 1) SELL with hysteresis: keep while the holding sits inside
        #    the top (K+slack) of the FULL ranking; otherwise only keep
        #    if its momentum is still positive and within swap_margin
        #    of the Kth-buyable. Core pairs are exempt while their
        #    momentum is positive (the index anchor).
        kth = buyable[cfg.top_k - 1][0] if len(buyable) >= cfg.top_k \
            else -1e9
        for pair in list(holdings):
            price = px(pair, t)
            if price is None:
                continue
            if pair in cfg.core_pairs and cfg.core_frac > 0:
                s_own = next((s for s, p in scored if p == pair), -1e9)
                if s_own > 0:
                    continue                       # core anchor: hold
            rank = next((i for i, (_, p) in enumerate(scored)
                         if p == pair), len(scored))
            s_own = next((s for s, p in scored if p == pair), -1e9)
            if rank < cfg.top_k + cfg.sell_slack:
                continue                       # still elite: hold
            if s_own > 0 and s_own >= kth - cfg.swap_margin:
                continue                       # positive & close: hold
            sell_all(pair, price, t)

        # 2) BUY/REBALANCE toward risk-parity targets, cash permitting
        order = ([(0.0, p) for p in cfg.core_pairs] +
                 buyable[:cfg.top_k])
        for _, pair in order:
            if cash < eq * 0.01:     # reserve floor
                break
            price = px(pair, t)
            if price is None:
                continue
            cur_val = (holdings.get(pair, 0.0)
                       * price * (1 - SLIP) * (1 - FEE))
            want_val = target_val.get(pair, 0.0)
            gap = want_val - cur_val
            if gap <= 0:
                continue
            # don't trade tiny gaps: the round-trip cost would eat it
            if gap < eq * cfg.swap_margin * 2:
                continue
            frac = min(gap / eq, cash / eq)
            if frac < 0.02:
                continue
            buy(pair, price, t, frac)
            invested = eq - cash

    # final equity
    eq = cash
    for p, q in holdings.items():
        t = closes[p].index[-1]
        eq += q * float(closes[p].loc[t])
    return A000Result(equity=eq, return_pct=(eq / cfg.capital - 1) * 100,
                      max_dd_pct=max_dd * 100, trades=trades,
                      fees_paid=fees_paid,
                      final_holdings={p: q for p, q in holdings.items()})
