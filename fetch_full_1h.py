#!/usr/bin/env python3
"""Fetch the ENTIRE 1H history for the true top-20 USDC pairs.

Chunked backward-compatible paging: four 900-day windows cover
3600 days (~10y); chunks older than a coin's listing date return
empty harmlessly, so each coin gets everything Coinbase serves.
Runs on CI (reliable IPs); the artifact is merged locally where
Athena trains (local-only).
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta, timezone

sys.path.insert(0, ".")

from bot.data.fetcher import fetch_candles
from bot.data.store import Store

CHUNKS = [(3600, 2700), (2700, 1800), (1800, 900), (900, 0)]
TOP_N = 20


def main() -> None:
    store = Store("data/athena_full.db")
    u4 = Store("data/universe.db")
    vols = {}
    for pair in [r[0] for r in u4.conn.execute(
            "SELECT DISTINCT pair FROM candles "
            "WHERE granularity='FOUR_HOUR'")]:
        df = u4.load_candles(pair, "FOUR_HOUR")
        if df is not None and len(df) >= 1000:
            v = (df["volume"] * df["close"]).tail(1000).median()
            if v == v:
                vols[pair] = float(v)
    pairs = sorted(vols, key=lambda p: -vols[p])[:TOP_N]
    print(f"[fetch-full] top-{TOP_N} by USDC volume: "
          f"{', '.join(p[:p.index('-')] for p in pairs)}", flush=True)
    now = datetime.now(timezone.utc)
    got = 0
    for pair in pairs:
        total = 0
        for days_back, days_to in CHUNKS:
            for attempt in range(5):
                try:
                    df = fetch_candles(
                        pair, "ONE_HOUR",
                        now - timedelta(days=days_back),
                        now - timedelta(days=days_to))
                    if len(df):
                        store.upsert_candles(pair, "ONE_HOUR", df)
                        total += len(df)
                    break
                except Exception as exc:  # noqa: BLE001
                    print(f"[fetch-full] {pair} retry {attempt + 1}: "
                          f"{exc}", flush=True)
                    time.sleep(10 * (attempt + 1))
        if total >= 500:
            got += 1
            span = total / 24 / 365
            print(f"[fetch-full] {pair}: {total} bars ({span:.1f}y)",
                  flush=True)
    print(f"[fetch-full] done: {got}/{len(pairs)}", flush=True)


if __name__ == "__main__":
    main()
