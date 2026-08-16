#!/usr/bin/env python3
"""Fetch 1H candles for the top-30 liquid USDC pairs (chunked
2300-day lookback) into data/athena_1h.db. Runs on CI (runner IPs
are not rate-limited); the artifact is downloaded locally for the
Athena training (which itself stays local-only)."""
from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta, timezone

sys.path.insert(0, ".")

from bot.data.fetcher import fetch_candles
from bot.data.store import Store

CHUNKS = [(700, 0), (1500, 700), (2300, 1500)]
TOP_N = 30


def main() -> None:
    store = Store("data/athena_1h.db")
    u4 = Store("data/universe.db")
    vols = {}
    for pair in [r[0] for r in u4.conn.execute(
            "SELECT DISTINCT pair FROM candles "
            "WHERE granularity='FOUR_HOUR'")]:
        df = u4.load_candles(pair, "FOUR_HOUR")
        if df is not None and len(df) >= 1000:
            v = df["volume"].tail(1000).median()
            if v == v:
                vols[pair] = float(v)
    pairs = sorted(vols, key=lambda p: -vols[p])[:TOP_N]
    print(f"[fetch-1h] {len(pairs)} liquid pairs", flush=True)
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
                    print(f"[fetch-1h] {pair} retry {attempt+1}: {exc}",
                          flush=True)
                    time.sleep(10 * (attempt + 1))
        if total >= 500:
            got += 1
            print(f"[fetch-1h] {pair}: {total} bars", flush=True)
    print(f"[fetch-1h] done: {got}/{len(pairs)}", flush=True)


if __name__ == "__main__":
    main()
