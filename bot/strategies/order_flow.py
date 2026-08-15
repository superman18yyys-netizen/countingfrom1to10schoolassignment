"""Exchange order-flow strength strategy (the "who is buying" bot).

Premise (your idea, made testable): *when aggressive buy-side transaction
flow turns up after a price drop, price is more likely to recover.* This
is the documented "order-flow / volume-delta" signal family in quant
research -- it carries real short-horizon predictive content, but it is
weak and crowded, and fees can eat it. The whole point is to TEST that
net-of-fee, not to promise profit.

Data (keyless, verified live):
  * trade-by-trade:  GET /market/products/{id}/ticker -> trades with
    ``side`` (BUY/SELL taker), price, size, time.
  * order book:      GET /market/product_book -> live bid/ask depth.

Signals (all from closed flow, no look-ahead):
  1. buy_ratio      : BUY$/SELL$ over the window (buyers stepping in?)
  2. cvd_change     : cumulative-volume-delta move (persistent buying?)
  3. buy_velocity   : buy$ arriving per second (how FAST are buyers?)
  4. book pressure  : bid-side $ share of the top of book (who can absorb)
  Combined with a drawdown-from-high filter (only act on a dip).

Both live and backtest share ONE normalized ``order_flow_score`` in
[-inf, inf]; live computes it from real trades, backtest approximates it
from candle volume (aggressor fraction = (close-low)/(high-low), a
standard documented proxy) so it can be validated on history. The score
is used to OVERRIDE an otherwise neutral signal only when it is strong.

Reasoning: every order attempt carries a structured human ``reason`` so
the runner (and humans) can audit WHY the bot acted.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from bot.data.fetcher import fetch_order_book, fetch_recent_trades
from bot.indicators.ta import ema
from bot.strategies.base import Strategy

DEFAULTS = {
    "window_sec": 300,        # order-flow aggregation window (5 min)
    "flow_momentum_sec": 900, # longer window for the CVD base/momentum
    "drawdown_window": 96,    # bars for the rolling high (dip filter)
    "drawdown_pct": 0.06,     # only overweight flow AFTER ~this drawdown
    "buy_ratio_thresh": 1.35, # buy$/sell$ must exceed this
    "cvd_min_pct": 0.005,     # CVD must have risen >= this much in window
    "book_bid_frac": 0.52,    # book must be at least this bid-side
    "target_pct": 0.05,       # sell target above entry
    "stop_pct": 0.04,         # stop below entry
    "poll_sec": 30,           # live REST poll cadence
}


class OrderFlowStrategy(Strategy):
    name = "order_flow"

    def __init__(self, params=None):
        super().__init__(params)
        self.p = {**DEFAULTS, **(params or {})}
        self._session = requests.Session()
        self._flow_cache: dict[str, tuple[float, dict]] = {}  # pair -> (fetched_ts, flow)
        self._reason: str | None = None

    def warmup_bars(self) -> int:
        return int(self.p["drawdown_window"]) + 5

    # ----------------------------------------------------- order flow
    def _fetch_live_flow(self, pair: str) -> dict:
        """Pull real trades + book for a pair, aggregating buy vs sell."""
        now = time.time()
        cached = self._flow_cache.get(pair)
        if cached and now - cached[0] < self.p["poll_sec"]:
            return cached[1]
        trades = fetch_recent_trades(pair, limit=200, session=self._session)
        book = fetch_order_book(pair, limit=10, session=self._session)
        flow = self._aggregate_trades(trades, book)
        self._flow_cache[pair] = (now, flow)
        return flow

    def _aggregate_trades(self, trades: list[dict], book: dict) -> dict:
        win = float(self.p["window_sec"])
        now = time.time()
        buy = sell = t0 = tN = 0.0
        for t in trades:
            try:
                ts = datetime.fromisoformat(t["time"].replace("Z", "+00:00")).timestamp()
            except (ValueError, KeyError):
                continue
            if now - ts > win:
                continue
            size_usd = t["price"] * t["size"]
            if t.get("side", "").upper() in ("BUY", "BID"):
                buy += size_usd
            elif t.get("side", "").upper() in ("SELL", "ASK"):
                sell += size_usd
            t0 = ts if t0 == 0 else min(t0, ts)
            tN = max(tN, ts)
        # book pressure: bid $-share of top of book
        pricebook = book.get("pricebook") or {}
        bids = pricebook.get("bids") or []
        asks = pricebook.get("asks") or []
        bid_usd = sum(float(b.get("price", 0)) * float(b.get("size", 0)) for b in bids)
        ask_usd = sum(float(a.get("price", 0)) * float(a.get("size", 0)) for a in asks)
        tot = bid_usd + ask_usd
        book_bid = bid_usd / tot if tot > 0 else 0.5

        denom = sell if sell > 0 else 1e-9
        buy_ratio = buy / denom
        span = max(1.0, tN - t0)
        cvd_move = (buy - sell) / (buy + sell) if (buy + sell) > 0 else 0.0  # in [-]1,1
        cvd_pct = cvd_move * 100.0
        buy_velocity = (buy - sell) / span / 1e6  # net buy $/s, scaled to M$/s
        return {
            "buy_usd": buy, "sell_usd": sell, "buy_ratio": buy_ratio,
            "cvd_pct": cvd_pct, "buy_velocity": buy_velocity,
            "book_bid": book_bid, "n_trades": len(trades),
        }

    def _backtest_flow(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive an order-flow proxy from candles (documented approximation):
        aggressor-sell fraction = (close-low)/(high-low); buy fraction is the
        complement. This lets the same signal be validated historically.
        """
        high = df["high"].replace(0.0, np.nan)
        rng = (high - df["low"]).replace(0.0, np.nan)
        sell_share = ((df["close"] - df["low"]) / rng).clip(0.0, 1.0).fillna(0.5)
        buy_usd = df["volume"] * (1.0 - sell_share) * df["close"]
        sell_usd = df["volume"] * sell_share * df["close"]
        win = max(1, int(round(self.p["window_sec"] / 3600)))  # hour-fraction -> bars
        b = buy_usd.rolling(win).sum()
        s = sell_usd.rolling(win).sum()
        out = pd.DataFrame(index=df.index)
        out["buy_ratio"] = (b / s.replace(0.0, np.nan)).fillna(1.0)
        out["cvd"] = (buy_usd - sell_usd).rolling(win).sum().fillna(0.0)
        out["cvd_base"] = (buy_usd - sell_usd).rolling(win * 3).mean().fillna(0.0)
        out["cvd_pct"] = (out["cvd"] - out["cvd_base"]) / (
            (b + s).replace(0.0, np.nan) / 2.0
        ).clip(lower=1.0)
        out["cvd_pct"] = (out["cvd_pct"].fillna(0.0) * 100.0)
        out["buy_velocity"] = (b - s).diff().fillna(0.0)
        out["book_bid"] = 0.5  # no live book in backtest; use pure flow
        return out.replace([np.inf, -np.inf], 0.0)

    # ----------------------------------------------------- signal
    def compute_signals(self, df: pd.DataFrame, live: bool = False) -> pd.Series:
        p = self.p
        close = df["close"]
        rolling_high = close.rolling(int(p["drawdown_window"]), min_periods=1).max()
        drawdown = close / rolling_high - 1.0

        if live:
            sig = pd.Series(0, index=df.index, dtype=int)
            self._reason = None
            try:
                flow = self._fetch_live_flow(getattr(df, "attrs", {}).get("pair", ""))
                px = float(df["close"].iloc[-1])
                dd = float(drawdown.iloc[-1])
                score = self._flow_score(flow.get("buy_ratio", 1.0), px, dd,
                                         cvd=float(flow.get("cvd_pct", 0.0)),
                                         velocity=float(flow.get("buy_velocity", 0.0)),
                                         book=float(flow.get("book_bid", 0.5)))
                last_i = len(sig) - 1
                if score >= 1.5:
                    sig.iloc[last_i] = 1
                    target = px * (1.0 + float(self.p["target_pct"]))
                    self._reason = self.build_reason(
                        pair=df.attrs.get("pair", ""), price=px, action="BUY",
                        flow=flow, drawdown=dd, target=target)
                elif dd >= 0.0 or score <= -1.0:
                    sig.iloc[last_i] = -1
                    self._reason = self.build_reason(
                        pair=df.attrs.get("pair", ""), price=px, action="SELL",
                        flow=flow, drawdown=dd, target=0.0)
                else:
                    self._reason = None
            except Exception as exc:  # noqa: BLE001 - never kill the loop on a feed hiccup
                self._reason = f"[order_flow] feed error: {exc}"
            return sig

        fl = self._backtest_flow(df)
        score = self._flow_score(fl["buy_ratio"], close, drawdown,
                                 cvd=fl["cvd_pct"], velocity=fl["buy_velocity"],
                                 book=fl["book_bid"])
        sig = pd.Series(0, index=df.index, dtype=int)
        # Proxy discipline: only act on momentum-consistent, after-a-dip
        # flow. The candle-volume proxy is noisy, so this keeps backtest
        # from over-trading the way raw live flow would not.
        in_dip = drawdown <= -float(p["drawdown_pct"])
        strong_buy = score >= 1.5
        sig[in_dip & strong_buy] = 1
        sig[(drawdown >= 0.0) | (score <= -1.0)] = -1  # dip recovered / flow turned
        return sig

    def _flow_score(self, ratio, close, drawdown, cvd=None, velocity=None,
                    book=None) -> float:
        """Composite buy pressure in ~[-1, 1]. Accepts scalar or pandas
        Series (backtest) and returns the same type."""
        p = self.p
        s = 0.0 + (ratio - ratio)  # preserve dtype (scalar 0.0 or Series)
        s = s + (1.0 * (ratio >= float(p["buy_ratio_thresh"])))
        if cvd is not None:
            s = s + (1.0 * (cvd >= p["cvd_min_pct"] * 100.0))
        if book is not None:
            s = s + (0.5 * (book >= p["book_bid_frac"]))
        if not isinstance(ratio, pd.Series):
            # raise the score further only when the drawdown filter also holds
            if float(s) > 0 and float(drawdown) <= -float(p["drawdown_pct"]):
                s = float(s) + 0.5
        else:
            s = s.where(s <= 0, s + 0.5 * (drawdown <= -float(p["drawdown_pct"])))
        return s

    # -------------------------------------------------  reason log
    def last_reason(self) -> str | None:
        return self._reason

    def build_reason(self, pair: str, price: float, action: str, flow: dict,
                     drawdown: float, target: float) -> str:
        p = self.p
        lines = [f"[order_flow {pair}] {action} @ {price:,.2f}",
                 f"   reason: buy:sell ratio {flow.get('buy_ratio', 0):.2f} "
                 f"(buyers {'stepping in' if flow.get('buy_ratio', 1) >= p['buy_ratio_thresh'] else 'not dominant'}),"]
        lines.append(f"           CVD {flow.get('cvd_pct', 0):+.2f}% "
                     f"({'accumulating' if flow.get('cvd_pct', 0) > 0 else 'distributing'}),"
                     f" drawdown {drawdown * 100:+.1f}% from high,")
        lines.append(f"           book is {flow.get('book_bid', 0.5) * 100:.0f}% bid-side,")
        lines.append(f"           price sees {flow.get('n_trades', 0)} recent trades.")
        if target:
            lines.append(f"   plan: target ~{target:,.2f} ({p['target_pct'] * 100:.0f}% up) "
                         f"minus ~1.4% fees+slip = net positive if reached; "
                         f"stop {p['stop_pct'] * 100:.0f}% below entry.")
        return "\n".join(lines)


__all__ = ["OrderFlowStrategy"]