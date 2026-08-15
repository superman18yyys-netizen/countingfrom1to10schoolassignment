"""Live data feeds for the paper-trading engine.

Two feed implementations back the same interface:

* ``RestPollingFeed`` (default) -- polls the public REST candles endpoint
  every ``poll_seconds``. Simple and reliable; works directly for ``-USDC``
  products.

* ``WebSocketFeed`` -- subscribes to the public ``candles`` channel at
  ``wss://advanced-trade-ws.coinbase.com`` (no auth). Note: per Coinbase
  docs, ``-USDC`` products are only streamed on the authenticated ``user``
  channel; the public channels mirror the equivalent ``-USD`` product.
  Since USDC is pegged 1:1 to USD these prices are equivalent for paper
  trading purposes.
"""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

import pandas as pd
import requests

from bot.data.fetcher import fetch_candles
from bot.config import GRANULARITY_SECONDS

WS_URL = "wss://advanced-trade-ws.coinbase.com"

CandleCallback = Callable[[str, pd.DataFrame], None]


class RestPollingFeed:
    """Poll public REST candles for a set of pairs on an interval."""

    def __init__(self, pairs: list[str], granularity: str, poll_seconds: int = 30,
                 session: Optional[requests.Session] = None):
        self.pairs = pairs
        self.granularity = granularity
        self.poll_seconds = poll_seconds
        self.session = session or requests.Session()

    def latest_candles(self, pair: str) -> pd.DataFrame:
        """Return recently closed candles (a few bars beyond the last 2)."""
        seconds = GRANULARITY_SECONDS[self.granularity]
        end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        # fetch a bit past the last aligned boundary so fresh candles arrive
        start = end - timedelta(seconds=seconds * 4)
        return fetch_candles(pair, self.granularity, start, end, self.session)

    async def run(self, on_candle: CandleCallback) -> None:
        while True:
            for pair in self.pairs:
                try:
                    df = self.latest_candles(pair)
                    if not df.empty:
                        on_candle(pair, df)
                except Exception as exc:  # noqa: BLE001 - keep the loop alive
                    print(f"[feed] {pair} poll failed: {exc}")
            await asyncio.sleep(self.poll_seconds)


class WebSocketFeed:
    """WebSocket candles feed (public channel) with reconnect + heartbeats."""

    def __init__(self, pairs: list[str]):
        self.pairs = pairs
        self._latest: dict[str, pd.DataFrame] = {}

    def latest_candles(self, pair: str) -> pd.DataFrame:
        return self._latest.get(pair, pd.DataFrame())

    async def _consume(self, ws) -> None:
        async for raw in ws:
            msg = json.loads(raw)
            if msg.get("type") == "candles_update":
                for event in msg.get("events", []):
                    for candle in event.get("candles", []):
                        pid = candle.get("product_id")
                        if pid not in self._latest:
                            continue
                        frame = pd.DataFrame([{
                            "start": pd.to_datetime(int(candle["start"]), unit="s", utc=True),
                            "open": float(candle["open"]),
                            "high": float(candle["high"]),
                            "low": float(candle["low"]),
                            "close": float(candle["close"]),
                            "volume": float(candle.get("volume", 0.0)),
                        }]).set_index("start").astype(float)
                        merged = pd.concat([self._latest[pid], frame])
                        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
                        self._latest[pid] = merged.tail(5000)
            elif msg.get("type") == "error":
                print(f"[feed] ws error: {msg}")

    async def run(self, on_candle: CandleCallback) -> None:
        while True:
            try:
                async with __import__("websockets").connect(
                        WS_URL, ping_interval=20, ping_timeout=20, max_size=2**22) as ws:
                    subscribe = {
                        "type": "subscribe",
                        "product_ids": self.pairs,
                        "channel": "candles",
                    }
                    await ws.send(json.dumps(subscribe))
                    heartbeat = {
                        "type": "subscribe",
                        "product_ids": self.pairs,
                        "channel": "heartbeats",
                    }
                    await ws.send(json.dumps(heartbeat))
                    await self._consume(ws)
            except Exception as exc:  # noqa: BLE001
                print(f"[feed] ws disconnected ({exc}); reconnecting in 5s")
                await asyncio.sleep(5)