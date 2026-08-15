"""Historical OHLCV fetcher for Coinbase Advanced Trade public REST API.

All endpoints are public market data (no API keys required):
  GET https://api.coinbase.com/api/v3/brokerage/market/products/{id}/candles

The API returns at most 350 candles per request, so long ranges are
paginated forward with a small overlap to avoid gaps.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import requests

from bot.config import GRANULARITY_SECONDS

BASE_URL = "https://api.coinbase.com/api/v3/brokerage"
_MAX_CANDLES = 350
_SLEEP_BETWEEN_PAGES = 0.25  # stay well under the ~10 req/s public limit


def _parse_candles(payload: dict) -> pd.DataFrame:
    candles = payload.get("candles") or []
    rows = []
    for c in candles:
        rows.append({
            "start": pd.to_datetime(int(c["start"]), unit="s", utc=True),
            "open": float(c["open"]),
            "high": float(c["high"]),
            "low": float(c["low"]),
            "close": float(c["close"]),
            "volume": float(c.get("volume", 0.0)),
        })
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(rows).set_index("start")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.astype(float)


def fetch_candles(product_id: str, granularity: str,
                  start: datetime, end: datetime,
                  session: Optional[requests.Session] = None) -> pd.DataFrame:
    """Fetch candles in [start, end) for one product/granularity, paginated."""
    if end <= start:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    sess = session or requests.Session()
    step = timedelta(seconds=_MAX_CANDLES * GRANULARITY_SECONDS[granularity])
    frames: list[pd.DataFrame] = []
    cur_start = start
    while cur_start < end:
        cur_end = min(cur_start + step, end)
        params = {
            "start": int(cur_start.timestamp()),
            "end": int(cur_end.timestamp()),
            "granularity": granularity,
        }
        resp = sess.get(f"{BASE_URL}/market/products/{product_id}/candles",
                        params=params, timeout=30)
        if resp.status_code == 429:
            time.sleep(2.0)
            continue
        resp.raise_for_status()
        page = _parse_candles(resp.json())
        if page.empty:
            break
        frames.append(page)
        last_ts = page.index[-1].to_pydatetime()
        if last_ts + timedelta(seconds=GRANULARITY_SECONDS[granularity]) >= cur_end:
            cur_start = cur_end
        else:
            # partial page: we have reached the end of available history
            break
        time.sleep(_SLEEP_BETWEEN_PAGES)
    if not frames:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    out = pd.concat(frames)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out.astype(float)


def fetch_history(product_id: str, granularity: str, days: int,
                  session: Optional[requests.Session] = None) -> pd.DataFrame:
    """Fetch the last ``days`` of finished candles (rounded to candle alignment)."""
    now = datetime.now(timezone.utc)
    end = now - timedelta(seconds=GRANULARITY_SECONDS[granularity])  # only closed candles
    seconds = GRANULARITY_SECONDS[granularity]
    end = end.replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=days)
    return fetch_candles(product_id, granularity, start, end, session)


def list_products(quote: str = "", session: Optional[requests.Session] = None) -> list[dict]:
    """List public products; optionally filter by quote currency (e.g. 'USDC')."""
    sess = session or requests.Session()
    resp = sess.get(f"{BASE_URL}/market/products", timeout=30)
    resp.raise_for_status()
    products = resp.json().get("products", [])
    if quote:
        products = [p for p in products if p.get("quote_currency_id") == quote]
    return products


def fetch_recent_trades(product_id: str, limit: int = 100,
                        session: Optional[requests.Session] = None) -> list[dict]:
    """Fetch the most recent trades (public, no auth) for order-flow analysis.

    Each trade: {trade_id, product_id, price, size, time, side (BUY/SELL),
    exchange}. ``side`` is the taker side (who crossed the spread), which is
    the standard way to read aggressive buy vs sell flow.
    """
    sess = session or requests.Session()
    resp = sess.get(f"{BASE_URL}/market/products/{product_id}/ticker",
                    params={"limit": int(limit)}, timeout=15)
    resp.raise_for_status()
    trades = resp.json().get("trades") or []
    for t in trades:
        try:
            t["price"] = float(t.get("price", 0.0))
            t["size"] = float(t.get("size", 0.0))
        except (TypeError, ValueError):
            t["price"] = 0.0
            t["size"] = 0.0
    return trades


def fetch_order_book(product_id: str, limit: int = 10,
                     session: Optional[requests.Session] = None) -> dict:
    """Fetch the live public order book for one product (no auth)."""
    sess = session or requests.Session()
    resp = sess.get(f"{BASE_URL}/market/product_book",
                    params={"product_id": product_id, "limit": int(limit)}, timeout=15)
    resp.raise_for_status()
    return resp.json()