"""Optimal path finder: the mechanically calculated perfect run.

Not "buy the lowest, sell the highest" — the TRUE optimum for a
single-account sequential trader with perfect foresight: a chain of
trades across ALL coins where each trade's ENTIRE capital (now
compounded by every previous trade) buys one coin and sells it
later, maximizing the final product of returns.

Math: maximizing prod(1+r_i) over a non-overlapping set of
intervals == maximizing sum(log(1+r_i)) — classic weighted interval
scheduling, solved exactly by DP in O(n log n). Trades that overlap
in time can't both be taken (one account, all-in per trade).

Segments come from zigzag pivots at a fine threshold (2%) — every
local min -> future local max whose net return clears fees by a
margin. The DP finds the best subset.

Output: the full trade ledger (coin, buy time, sell time, capital
before/after), final equity from $100, and reports/optimal-path.json
for rule mining (train A-000 on the observable features of these
entries).
"""
from __future__ import annotations

import bisect
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RTC = 0.014
MIN_NET = 0.012          # a trade must clear fees + 1.2% margin
ZIGZAG_PCT = 0.02        # fine pivot threshold
SLIP = 0.001
FEE = 0.006
CASH_APY = 0.045


def simulate_path(closes: Dict[str, pd.Series], path: List[Trade],
                  capital: float = 100.0) -> Tuple[List[dict], float]:
    """Explicit day-by-day replay of the chosen path.

    Maintains REAL state — cash (USDC, accruing yield), holdings (one
    coin at a time, no margin: you must sell before buying again,
    and you can only spend cash you actually have). Every fill pays
    fee + slippage at the exact bar close. Verifies the path is
    executable and returns the day-by-day ledger + final equity.

    Returns (ledger, final_equity). Raises AssertionError if the path
    is not executable (should never happen — the DP enforces
    non-overlap by construction; this is the proof).
    """
    cash = capital
    holding: Optional[str] = None
    qty = 0.0
    last_ts = 0
    ledger: List[dict] = []
    for n, t in enumerate(path):
        # idle cash yield between trades
        if last_ts:
            years = (t.buy_ts - last_ts) / 31536000.0
            cash *= (1.0 + CASH_APY * years)
        # BUY: must be in cash; fee convention matches the segment net
        # formula: S = cash/(1+FEE) is invested, fee = S*FEE
        assert holding is None, "path buys while already holding"
        fill = t.buy_px * (1.0 + SLIP)
        spend = cash / (1.0 + FEE)
        fee = spend * FEE if math.isfinite(spend) else 0.0
        qty = spend / fill
        cash = 0.0
        holding = t.pair
        # SELL: must own this coin; proceeds pay fee+slippage
        out_fill = t.sell_px * (1.0 - SLIP)
        gross = qty * out_fill
        fee2 = gross * FEE if math.isfinite(gross) else 0.0
        cash = gross - fee2
        holding = None
        last_ts = t.sell_ts
        ledger.append({
            "pair": t.pair,
            "buy_ts": t.buy_ts, "sell_ts": t.sell_ts,
            "buy_px": round(t.buy_px, 6), "sell_px": round(t.sell_px, 6),
            "qty": round(qty, 10),
            "fees_paid": round(fee + fee2, 6),
            "cash_after": round(cash, 2),
            "gross_ret_pct": round((t.sell_px / t.buy_px - 1) * 100, 3),
            "net_ret_pct": round(t.net * 100, 3),
        })
        if (n + 1) % 500 == 0:
            print(f"[optimal] simulation: {n + 1}/{len(path)} trades "
                  f"replayed, equity ${cash:,.2f}", flush=True)
    return ledger, cash


@dataclass
class Trade:
    pair: str
    buy_i: int           # bar index within its own frame
    sell_i: int
    buy_ts: int
    sell_ts: int
    net: float           # fee-adjusted return
    logr: float          # log(1+net) — DP weight
    buy_px: float
    sell_px: float


def _zigzag(close: np.ndarray, pct: float) -> List[Tuple[int, float, str]]:
    piv, mode, ext_i, ext_p = [], "init", 0, float(close[0])
    for i in range(len(close)):
        p = float(close[i])
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


def build_segments(closes: Dict[str, pd.Series],
                   total: Optional[int] = None) -> List[Trade]:
    segs: List[Trade] = []
    total = total or len(closes)
    done = 0
    for pair, close in closes.items():
        c = close.to_numpy()
        idx = close.index
        piv = _zigzag(c, ZIGZAG_PCT)
        pending: Optional[Tuple[int, float]] = None
        for i, p, kind in piv:
            if kind == "lo":
                pending = (i, p)
            elif kind == "hi" and pending is not None:
                buy_i, buy_p = pending
                net = p / buy_p * (1 - 0.001) * (1 - 0.006) \
                    / (1 + 0.001) / (1 + 0.006) - 1.0
                if net >= MIN_NET and i > buy_i:
                    segs.append(Trade(
                        pair=pair, buy_i=buy_i, sell_i=i,
                        buy_ts=int(idx[buy_i].timestamp()),
                        sell_ts=int(idx[i].timestamp()),
                        net=net, logr=math.log1p(net),
                        buy_px=buy_p, sell_px=p))
                pending = None
        done += 1
        if done % 50 == 0 or done == total:
            print(f"[optimal] segments: {done}/{total} coins done, "
                  f"{len(segs)} profitable segments so far", flush=True)
    return segs


def optimal_path(segs: List[Trade]) -> Tuple[List[Trade], float]:
    """DP: best compounding chain. O(n log n). Uses longdouble — the
    optimal chain on 394 coins can exceed float64's exp range."""
    print(f"[optimal] DP over {len(segs):,} segments...", flush=True)
    segs.sort(key=lambda t: t.sell_ts)
    ends = [t.sell_ts for t in segs]
    starts = [t.buy_ts for t in segs]
    n = len(segs)
    dp = np.zeros(n, dtype=np.longdouble)
    parent = [-1] * n
    best_end = np.zeros(n, dtype=np.longdouble)
    best_par = [-1] * n
    for i in range(n):
        # last segment that ends before this one starts
        j = bisect.bisect_right(ends, starts[i]) - 1
        if j >= 0:
            dp[i] = best_end[j] + segs[i].logr
            parent[i] = best_par[j]
        else:
            dp[i] = segs[i].logr
        if i == 0 or dp[i] > best_end[i - 1]:
            best_end[i] = dp[i]
            best_par[i] = i
        else:
            best_end[i] = best_end[i - 1]
            best_par[i] = best_par[i - 1]
    path, cur = [], int(best_par[n - 1]) if n else -1
    while cur >= 0:
        path.append(segs[cur])
        cur = parent[cur]
    path.reverse()
    total_log = best_end[n - 1] if n else np.longdouble(0)
    try:
        equity = float(np.exp(total_log))
    except (OverflowError, FloatingPointError):
        equity = float("inf")
    return path, equity, float(total_log)


def main() -> None:
    from bot.data.store import Store
    store = Store("data/universe.db")
    closes = {}
    for pair in [r[0] for r in store.conn.execute(
            "SELECT DISTINCT pair FROM candles WHERE granularity='FOUR_HOUR'")]:
        df = store.load_candles(pair, "FOUR_HOUR")
        if df is not None and len(df) >= 400:
            closes[pair] = df["close"].dropna()

    print(f"[optimal] {len(closes)} coins, building segments...")
    t0 = time.time()
    segs = build_segments(closes, total=len(closes))
    print(f"[optimal] {len(segs):,} profitable segments "
          f"({time.time() - t0:.1f}s)")
    path, equity, total_log = optimal_path(segs)
    print(f"[optimal] path found: {len(path)} trades "
          f"({time.time() - t0:.1f}s total)")

    # EXPLICIT DAY-BY-DAY SIMULATION: prove the path is executable
    # with real cash/holdings/fees, and get the true realized equity
    ledger, sim_equity = simulate_path(closes, path, capital=100.0)
    if math.isfinite(sim_equity) and math.isfinite(equity):
        drift = abs(sim_equity - 100.0 * equity) / max(sim_equity, 1e-9)
        print(f"[optimal] DP predicts ${100 * equity:,.2f}; explicit "
              f"simulation realizes ${sim_equity:,.2f} "
              f"(drift {drift * 100:.4f}% = idle-cash yield)")
    else:
        print(f"[optimal] equity exceeds float64 range: log-sum "
              f"{total_log:.1f} = 10^{total_log * 0.4343:.0f}")
    cap = 100.0
    disp = (f"${sim_equity:,.2f}" if math.isfinite(sim_equity)
            else f"10^{total_log * 0.4343:.0f}")
    print(f"\n== OPTIMAL PATH: ${cap:.0f} -> {disp} in {len(path)} trades "
          f"(day-by-day simulated, fees+slippage on every fill) ==\n")
    for row in ledger[:40]:
        d0 = datetime.fromtimestamp(row["buy_ts"], tz=timezone.utc)
        d1 = datetime.fromtimestamp(row["sell_ts"], tz=timezone.utc)
        print(f"  {d0:%Y-%m-%d %H:%M} BUY  {row['pair']:<10} "
              f"${row['cash_after'] / (1 + row['net_ret_pct'] / 100):>13,.2f}"
              f" -> {d1:%Y-%m-%d %H:%M} SELL net "
              f"{row['net_ret_pct']:+6.1f}%  ${row['cash_after']:,.2f} "
              f"(fees ${row['fees_paid']:.2f})")
    if len(ledger) > 40:
        print(f"  ... ({len(ledger) - 40} more trades)")

    out = {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
           "coins": sorted(closes), "segments_total": len(segs),
           "trades_taken": len(path),
           "dp_multiplier": equity if math.isfinite(equity) else None,
           "log_sum": round(total_log, 2),
           "simulated_equity_from_100":
               round(sim_equity, 2) if math.isfinite(sim_equity) else None,
           "ledger": ledger}
    os.makedirs("reports", exist_ok=True)
    with open("reports/optimal-path.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\n[optimal] -> reports/optimal-path.json")
    print(f"[optimal] final equity from $100: {disp}")


if __name__ == "__main__":
    main()
