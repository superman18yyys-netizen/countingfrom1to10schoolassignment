#!/usr/bin/env python3
"""Fetch the FULL USDC universe (6y 4H) for the A-000 lab.

Every *-USDC product Coinbase lists — not a curated subset. Backward
paging from today handles any listing date automatically. Pairs with
less than MIN_BARS of history are discarded.
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

from bot.data.fetcher import (BASE_URL, GRANULARITY_SECONDS,
                              _parse_candles, list_products)
from bot.data.store import Store

MIN_BARS = 400                # ~2.5 months of 4H — enough to matter
DAYS = 2300
GRAN = "FOUR_HOUR"


def fetch_all_backward(pair: str, end: datetime, days: int) -> pd.DataFrame:
    """Page backward from `end` in ~58-day steps until history runs out."""
    import requests as _requests
    step = timedelta(seconds=300 * GRANULARITY_SECONDS[GRAN])
    sess = _requests.Session()
    frames = []
    cur_end = end
    stop = end - timedelta(days=days)
    while cur_end > stop:
        cur_start = max(cur_end - step, stop)
        params = {"start": int(cur_start.timestamp()),
                  "end": int(cur_end.timestamp()),
                  "granularity": GRAN}
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
        time.sleep(0.08)
    if not frames:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    out = pd.concat(frames)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out.astype(float)


def main() -> None:
    store = Store("data/universe.db")
    end = datetime.now(timezone.utc)

    products = list_products("USDC")
    pairs = sorted({p.get("product_id") or p.get("id") for p in products}
                   - {None})
    print(f"[universe] Coinbase lists {len(pairs)} USDC pairs; "
          f"fetching all with {DAYS}d of history...")

    kept = 0
    for pair in pairs:
        got = False
        for attempt in range(3):
            try:
                df = fetch_all_backward(pair, end, DAYS)
                if len(df) >= MIN_BARS:
                    store.upsert_candles(pair, GRAN, df)
                    kept += 1
                    span = (df.index[-1] - df.index[0]).total_seconds() / 86400
                    print(f"[universe] {pair}: {len(df)} bars "
                          f"({span:.0f}d)", flush=True)
                got = True
                break
            except Exception as exc:  # noqa: BLE001
                print(f"[universe] {pair} attempt {attempt+1} failed: {exc}",
                      flush=True)
                time.sleep(8 * (attempt + 1))
        if not got:
            print(f"[universe] {pair}: SKIPPED", flush=True)
    print(f"[universe] done: {kept} pairs with >= {MIN_BARS} bars kept")


if __name__ == "__main__":
    main()
