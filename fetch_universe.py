#!/usr/bin/env python3
"""Fetch the expanded USDC universe (6y 4H) for the A-000 lab."""
from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

from bot.data.fetcher import (BASE_URL, GRANULARITY_SECONDS,
                              _parse_candles, list_products)
from bot.data.store import Store

UNIVERSE = [
    "BTC-USDC", "ETH-USDC", "SOL-USDC", "DOGE-USDC", "XRP-USDC",
    "ADA-USDC", "LTC-USDC", "LINK-USDC", "AVAX-USDC", "DOT-USDC",
    "UNI-USDC", "AAVE-USDC", "MATIC-USDC", "SHIB-USDC", "OP-USDC",
    "ARB-USDC", "NEAR-USDC", "APT-USDC", "SUI-USDC", "FET-USDC",
]


def fetch_all_backward(pair: str, gran: str, end: datetime,
                       days: int) -> pd.DataFrame:
    """Page backward from `end` in 50-day steps until history runs out.
    (Forward paging breaks on the FIRST empty page, which loses every
    pair listed after the requested start date.)"""
    import requests as _requests
    step = timedelta(seconds=300 * GRANULARITY_SECONDS[gran])
    sess = _requests.Session()
    frames = []
    cur_end = end
    stop = end - timedelta(days=days)
    while cur_end > stop:
        cur_start = max(cur_end - step, stop)
        params = {"start": int(cur_start.timestamp()),
                  "end": int(cur_end.timestamp()),
                  "granularity": gran}
        resp = sess.get(
            f"{BASE_URL}/market/products/{pair}/candles",
            params=params, timeout=30)
        if resp.status_code == 429:
            time.sleep(2.0)
            continue
        resp.raise_for_status()
        page = _parse_candles(resp.json())
        if page.empty:
            break                      # reached the listing date
        frames.append(page)
        cur_end = cur_start
        time.sleep(0.12)
    if not frames:
                return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    out = pd.concat(frames)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out.astype(float)


def main() -> None:
    days = 2300
    gran = "FOUR_HOUR"
    store = Store("data/universe.db")
    end = datetime.now(timezone.utc)

    available = list_products("USDC")
    ids = {p.get("product_id") or p.get("id") for p in available}
    missing = [u for u in UNIVERSE if u not in ids]
    if missing:
        print(f"[universe] not listed on Coinbase: {missing}")

    for pair in UNIVERSE:
        if pair not in ids:
            continue
        got = False
        for attempt in range(4):
            try:
                df = fetch_all_backward(pair, gran, end, days)
                if not df.empty:
                    store.upsert_candles(pair, gran, df)
                    span_days = (df.index[-1] - df.index[0]).total_seconds() / 86400
                    print(f"[universe] {pair}: {len(df)} bars "
                          f"({span_days:.0f}d)", flush=True)
                else:
                    print(f"[universe] {pair}: no history available",
                          flush=True)
                got = True
                break
            except Exception as exc:  # noqa: BLE001
                print(f"[universe] {pair} attempt {attempt+1} failed: {exc}",
                      flush=True)
                time.sleep(10 * (attempt + 1))
        if not got:
            print(f"[universe] {pair}: SKIPPED (all attempts failed)",
                  flush=True)


if __name__ == "__main__":
    main()
