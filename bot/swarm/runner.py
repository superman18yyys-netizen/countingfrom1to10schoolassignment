"""Swarm runner: gap-fill replay + live candle loop for N agents.

One code path handles both catch-up and live trading:

* ``sync()`` fetches recent candles for every pair and replays every
  closed candle newer than ``population.last_ts`` through all agents.
  This makes the system self-healing: if a GitHub Actions run is late,
  skipped, or crashed, the next run simply replays the missed candles
  (signals are deterministic given the candles, so no trades are lost).

* The live loop then polls for newly closed candles until the window
  ends, executing pretend fills (fees + slippage) via each agent's
  PaperAccount -- the "virtual buy/sell tool".
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import pandas as pd

from bot.config import GRANULARITY_SECONDS
from bot.data.fetcher import fetch_candles
from bot.data.store import Store
from bot.swarm.population import Population

WINDOW_BARS = 1200          # candles kept per pair (warmup for indicators)
FETCH_BACK_BARS = 1200      # default lookback when no state timestamp yet


class SwarmRunner:
    def __init__(self, pairs: List[str], granularity: str, population: Population,
                 db_path: str = "data/swarm.db", poll_seconds: int = 60,
                 verbose: bool = True, on_save=None):
        self.pairs = pairs
        self.granularity = granularity
        self.bar_sec = GRANULARITY_SECONDS[granularity]
        self.population = population
        self.store = Store(db_path)
        self.poll_seconds = poll_seconds
        self.verbose = verbose
        self.on_save = on_save      # optional callback fired after each save
        self._processed: Dict[str, int] = {}   # pair -> last processed candle ts

    # ---------------------------------------------------------------- sync
    def _fetch_recent(self, pair: str, start_ts: int, end_ts: int) -> pd.DataFrame:
        start = datetime.fromtimestamp(start_ts, tz=timezone.utc)
        end = datetime.fromtimestamp(end_ts, tz=timezone.utc)
        last_exc: Optional[Exception] = None
        for attempt in range(4):
            try:
                df = fetch_candles(pair, self.granularity, start, end)
                if not df.empty:
                    self.store.upsert_candles(pair, self.granularity, df)
                return df
            except Exception as exc:  # noqa: BLE001 - retry transient network errors
                last_exc = exc
                wait = 2 ** attempt
                self._log(f"fetch {pair} attempt {attempt + 1} failed ({exc}); "
                          f"retrying in {wait}s")
                time.sleep(wait)
        raise RuntimeError(f"fetch {pair} failed after retries: {last_exc}")

    def _aligned_now(self) -> int:
        """Current time truncated to the candle boundary (start of the
        currently-forming candle). Only candles before this are closed."""
        now = int(datetime.now(timezone.utc).timestamp())
        return now - (now % self.bar_sec)

    def _sync_start_ts(self, aligned_now: int) -> int:
        """Fetch start: always far enough back to fill a full indicator
        window even on a fresh machine (CI runners are ephemeral)."""
        window_start = aligned_now - WINDOW_BARS * self.bar_sec
        if self.population.last_ts > 0:
            return min(self.population.last_ts - self.bar_sec, window_start)
        return aligned_now - FETCH_BACK_BARS * self.bar_sec

    def sync(self) -> int:
        """Fetch data and replay all unprocessed closed candles.
        Returns the number of candles processed."""
        aligned_now = self._aligned_now()
        start_ts = self._sync_start_ts(aligned_now)
        processed = 0
        pending: List[tuple[int, str]] = []
        for pair in self.pairs:
            df = self._fetch_recent(pair, start_ts, aligned_now)
            if df.empty:
                continue
            last_done = max(self.population.last_ts,
                            self._processed.get(pair, 0))
            fresh = [int(ts.timestamp()) for ts in df.index
                     if int(ts.timestamp()) > last_done and int(ts.timestamp()) < aligned_now]
            pending.extend((ts, pair) for ts in fresh)
        for ts, pair in sorted(pending):
            self._step_candle(pair, ts)
            processed += 1
        if pending:
            self.population.last_ts = max(ts for ts, _ in pending)
            self._log(f"processed {processed} candle(s) up to "
                      f"{datetime.fromtimestamp(self.population.last_ts, tz=timezone.utc):%Y-%m-%d %H:%M} UTC")
        return processed

    # ---------------------------------------------------------------- step
    def _window(self, pair: str, ts: int) -> pd.DataFrame:
        start_ts = ts - WINDOW_BARS * self.bar_sec
        df = self.store.load_candles(pair, self.granularity, start=start_ts, end=ts)
        return df

    def _step_candle(self, pair: str, ts: int) -> None:
        """Run every agent's strategy on the window ending at `ts`; each
        strategy places its own orders via the virtual account (fees +
        slippage applied inside the account)."""
        df = self._window(pair, ts)
        if len(df) < 30:
            return
        price = float(df["close"].iloc[-1])
        for agent in self.population.agents:
            try:
                agent.account.accrue_yield(ts)   # idle USDC earns the risk-free rate
                # Strategies that need a live/streaming data feed (order
                # flow, ML, LLM) get live=True so they hit the REST feed /
                # model rather than a static backtest model.
                live = agent.genome.strategy in ("ml_trend", "order_flow", "llm_trader")
                result = agent.strategy.execute(agent.account, pair, df, price, ts,
                                                live=live)
            except Exception as exc:  # noqa: BLE001 - one bad bot must not kill the swarm
                self._log(f"[{agent.genome.id}] execute error: {exc}")
                continue
            if result is None:
                continue
            reason = getattr(agent.strategy, "last_reason", None) and \
                agent.strategy.last_reason()
            if not self.verbose and not reason:
                continue
            if reason:
                self._log(f"[{agent.genome.id}] {reason}")
            elif result["action"] == "buy":
                self._log(f"[{agent.genome.id}] BUY  {pair} @ {price:.6g} "
                          f"(qty {result['qty']:.5g}, fee ${result['fee']:.3f})")
            elif result["action"] == "sell":
                self._log(f"[{agent.genome.id}] SELL {pair} @ {price:.6g} "
                          f"pnl ${result['pnl']:+.3f} ({result['pnl_pct'] * 100:+.2f}%)")
        self._processed[pair] = ts

    def _log(self, msg: str) -> None:
        stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        print(f"[swarm {stamp}] {msg}", flush=True)

    # ----------------------------------------------------------------- run
    def run(self, hours: float, save_every_loops: int = 1,
            state_path: Optional[str] = None) -> None:
        deadline = datetime.now(timezone.utc) + timedelta(hours=hours)
        self.sync()
        loop = 0
        while datetime.now(timezone.utc) < deadline:
            time.sleep(self.poll_seconds)
            try:
                self.sync()
            except Exception as exc:  # noqa: BLE001 - network hiccups must not kill the run
                self._log(f"sync error (will retry): {exc}")
            loop += 1
            # Save local state each loop so the workflow's git commit step
            # always has fresh state/zoo.json to commit, even if a long
            # window is interrupted or a later loop stalls.
            if state_path and (loop == 1 or loop % save_every_loops == 0):
                self._mark_and_save(state_path)
        self._mark_and_save(state_path)

    def _mark_and_save(self, state_path: Optional[str]) -> None:
        prices = self.latest_prices()
        self.population.mark_equity(prices)
        if state_path:
            self.population.save(state_path)
        if self.on_save is not None:
            try:
                self.on_save()
            except Exception as exc:  # noqa: BLE001 - board render must never kill the run
                self._log(f"on_save callback failed: {exc}")

    def latest_prices(self) -> Dict[str, float]:
        prices: Dict[str, float] = {}
        for pair in self.pairs:
            df = self.store.load_candles(pair, self.granularity)
            if not df.empty:
                prices[pair] = float(df["close"].iloc[-1])
        return prices
