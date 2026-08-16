#!/usr/bin/env python3
"""Fetch the expanded USDC universe (6y 4H) for the A-000 lab."""
from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta, timezone

sys.path.insert(0, ".")

from bot.data.fetcher import fetch_candles, list_products
from bot.data.store import Store

UNIVERSE = [
    "BTC-USDC", "ETH-USDC", "SOL-USDC", "DOGE-USDC", "XRP-USDC",
    "ADA-USDC", "LTC-USDC", "LINK-USDC", "AVAX-USDC", "DOT-USDC",
    "UNI-USDC", "AAVE-USDC", "MATIC-USDC", "SHIB-USDC", "OP-USDC",
    "ARB-USDC", "NEAR-USDC", "APT-USDC", "SUI-USDC", "FET-USDC",
]


def main() -> None:
    days = 2300
    gran = "FOUR_HOUR"
    store = Store("data/universe.db")
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

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
                df = fetch_candles(pair, gran, start, end)
                if not df.empty:
                    store.upsert_candles(pair, gran, df)
                    span_days = (df.index[-1] - df.index[0]).total_seconds() / 86400
                    print(f"[universe] {pair}: {len(df)} bars "
                          f"({span_days:.0f}d)", flush=True)
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
