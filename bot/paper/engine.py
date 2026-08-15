"""Live paper-trading engine.

Polls public Coinbase candles, computes signals on the last CLOSED
candle, and executes pretend fills on that close (with fees + slippage).
Executing at the just-closed candle's close is the paper equivalent of
the backtester's next-bar-open fill (the next bar opens exactly when the
previous one closes); the REST poll adds at most ``poll_seconds`` of
staleness, which is honest paper-trading behavior.

Persistence: everything is written to SQLite (fills, closed trades,
equity snapshots, account state) so a run can be stopped and resumed.
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import pandas as pd

from bot.data.live import RestPollingFeed, WebSocketFeed
from bot.data.store import Store
from bot.paper.account import PaperAccount
from bot.paper.ledger import load_account, record_closed_trade, save_account
from bot.strategies.base import Strategy

WINDOW_BARS = 4000  # candles kept in memory per pair (1h ~ 5.5 months)


class PaperEngine:
    def __init__(self, config, strategies: List[Strategy], store: Optional[Store] = None):
        self.config = config
        self.strategies = strategies
        self.store = store or Store(config.db_path)
        self.feed = RestPollingFeed(
            config.pairs, config.granularity, poll_seconds=config.poll_seconds
        )
        self.accounts: Dict[str, PaperAccount] = {}
        self._baseline_qty: Dict[str, float] = {}
        self._baseline_price: Dict[str, float] = {}
        self._df_cache: Dict[str, pd.DataFrame] = {}
        self._prices: Dict[str, float] = {}
        self._start_ts = int(time.time())

    # ------------------------------------------------------------ lifecycle
    def _get_account(self, strategy: str) -> PaperAccount:
        if strategy not in self.accounts:
            acc = load_account(self.store, strategy) or PaperAccount(
                capital=self.config.paper_capital,
                taker_fee=self.config.taker_fee,
                slippage=self.config.slippage,
                position_fraction=self.config.position_fraction,
                max_positions=self.config.max_positions,
                cash_yield_apy=self.config.cash_yield_apy,
            )
            self.accounts[strategy] = acc
        return self.accounts[strategy]

    def _load_window(self, pair: str, now_ts: int) -> pd.DataFrame:
        start_ts = now_ts - WINDOW_BARS * self.config.candle_seconds
        df = self.store.load_candles(pair, self.config.granularity, start=start_ts)
        return df

    def _baseline_setup(self, prices: Dict[str, float]) -> None:
        if self._baseline_qty:
            return
        share = self.config.paper_capital / len(self.config.pairs)
        for pair in self.config.pairs:
            price = prices.get(pair)
            if price:
                self._baseline_qty[pair] = share / price
                self._baseline_price[pair] = price

    def _baseline_equity(self, prices: Dict[str, float]) -> float:
        return sum(qty * prices.get(pair, self._baseline_price.get(pair, 0.0))
                   for pair, qty in self._baseline_qty.items())

    # --------------------------------------------------------------- tick
    def tick(self) -> None:
        now = datetime.now(timezone.utc)
        now_ts = int(now.timestamp())
        prices: Dict[str, float] = {}
        for pair in self.config.pairs:
            try:
                fresh = self.feed.latest_candles(pair)
                if not fresh.empty:
                    self.store.upsert_candles(pair, self.config.granularity, fresh)
                df = self._load_window(pair, now_ts)
                if df.empty:
                    continue
                self._df_cache[pair] = df
                prices[pair] = float(df["close"].iloc[-1])
            except Exception as exc:  # noqa: BLE001
                print(f"[paper] {pair} data error: {exc}")
        self._prices = prices
        if not prices:
            return
        self._baseline_setup(prices)

        for strategy in self.strategies:
            account = self._get_account(strategy.name)
            for pair in self.config.pairs:
                df = self._df_cache.get(pair)
                if df is None or len(df) < 30:
                    continue
                try:
                    signals = strategy.compute_signals(df, live=True)
                    sig = signals.iloc[-1] if len(signals) else 0
                    if pd.isna(sig):
                        sig = 0
                    sig = int(sig)
                except Exception as exc:  # noqa: BLE001 - never kill the loop
                    print(f"[paper] {strategy.name}/{pair} signal error: {exc}")
                    continue
                price = prices[pair]
                if sig == 1:
                    pos = account.open_position(pair, price, now_ts)
                    if pos is not None:
                        self.store.record_fill(strategy.name, pair, "BUY", now_ts,
                                               price * (1 + self.config.slippage),
                                               pos.qty, pos.entry_fee)
                        print(f"[paper] {strategy.name} BUY  {pair} @ {price:.4g} "
                              f"qty={pos.qty:.6g} fee=${pos.entry_fee:.2f}")
                elif sig == -1:
                    closed = account.close_position(pair, price, now_ts)
                    if closed is not None:
                        record_closed_trade(self.store, strategy.name, closed)
                        self.store.record_fill(strategy.name, pair, "SELL", now_ts,
                                               price * (1 - self.config.slippage),
                                               closed["qty"], closed["exit_fee"])
                        print(f"[paper] {strategy.name} SELL {pair} @ {price:.4g} "
                              f"pnl=${closed['pnl']:.2f} ({closed['pnl_pct'] * 100:.2f}%)")
            account.accrue_yield(now_ts)   # idle USDC earns the risk-free rate
            save_account(self.store, strategy.name, account)

        # equity snapshots
        points = []
        for strategy in self.strategies:
            eq = self.accounts[strategy.name].equity(prices)
            points.append((strategy.name, "ALL", eq))
        points.append(("buy_hold", "ALL", self._baseline_equity(prices)))
        self.store.save_equity(now_ts, points)

    # ---------------------------------------------------------------- run
    async def run(self, duration_hours: Optional[float] = None) -> None:
        print(f"[paper] starting {len(self.strategies)} strategy(s) on {self.config.pairs} "
              f"({self.config.granularity}), capital ${self.config.paper_capital:,.0f}")
        deadline = (datetime.now(timezone.utc) + timedelta(hours=duration_hours)
                    if duration_hours else None)
        tick_i = 0
        while True:
            if deadline and datetime.now(timezone.utc) >= deadline:
                print("[paper] duration reached, stopping")
                break
            self.tick()
            tick_i += 1
            if tick_i % (600 // max(1, self.config.poll_seconds)) == 0:
                elapsed = (datetime.now(timezone.utc) - datetime.fromtimestamp(
                    self._start_ts, tz=timezone.utc)).total_seconds() / 3600
                print(f"[paper] running {elapsed:.1f}h | "
                      + " | ".join(f"{s.name}=${self.accounts.get(s.name, PaperAccount(self.config.paper_capital)).equity(self._prices):,.0f}"
                                   for s in self.strategies)
                      + f" | buy_hold=${self._baseline_equity(self._prices):,.0f}")
            await asyncio.sleep(self.config.poll_seconds)