#!/usr/bin/env python3
"""Fetch 15m candles for the top-15 liquid USDC pairs (chunked ~500d)
into data/athena_15m.db. CI runs this; the artifact is downloaded
locally for the 15m-resolution nose."""
from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta, timezone

sys.path.insert(0, ".")

from bot.data.fetcher import fetch_candles
from bot.data.store import Store

CHUNKS = [(450, 0), (900, 450)]
TOP_N = 15


def main() -> None:
    store = Store("data/athena_15m.db")
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
    print(f"[fetch-15m] top-{TOP_N} pairs", flush=True)
    now = datetime.now(timezone.utc)
    got = 0
    for pair in pairs:
        total = 0
        for days_back, days_to in CHUNKS:
            for attempt in range(5):
                try:
                    df = fetch_candles(
                        pair, "FIFTEEN_MINUTE",
                        now - timedelta(days=days_back),
                        now - timedelta(days=days_to))
                    if len(df):
                        store.upsert_candles(pair, "FIFTEEN_MINUTE", df)
                        total += len(df)
                    break
                except Exception as exc:  # noqa: BLE001
                    print(f"[fetch-15m] {pair} retry {attempt+1}: {exc}",
                          flush=True)
                    time.sleep(10 * (attempt + 1))
        if total >= 5000:
            got += 1
            print(f"[fetch-15m] {pair}: {total} bars", flush=True)
    print(f"[fetch-15m] done: {got}/{len(pairs)}", flush=True)


if __name__ == "__main__":
    main()
