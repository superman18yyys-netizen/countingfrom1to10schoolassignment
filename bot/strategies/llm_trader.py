"""LLM-orchestrated trader (deepseek v4 flash, thinking enabled).

Layered design -- the LLM is a DELIBERATOR on top of hard statistics,
NOT a raw predictor. Deterministic code computes a real "situation
brief"; the LLM weighs pre-vetted candidate signals against a
research-grounded risk/fee/regime playbook and returns a structured
decision JSON. Every decision logs its reasoning (the "thinking").

Live path: calls DeepSeek (needs OPENCODE_GO_KEY secret) on a cadence.
Backtest/offline path: falls back deterministically to the underlying
signals, so the strategy is importable and testable without a key and
without hammering the API. The live reasoning is the real behavior.

The strategy still enforces hard risk rules LOCALLY (sizing, stop, max
trades), so a bad or malformed LLM call can never blow up the account.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone

import requests

from bot.data.llm_client import LLMClient, parse_decision_json
from bot.data.fetcher import fetch_recent_trades
from bot.indicators.ta import atr, ema, rsi, sma
from bot.strategies.base import Strategy

log = logging.getLogger("llm_trader")

DEFAULTS = {
    "base_url": "https://opencode.ai/zen/go/v1",
    "model": "deepseek-v4-flash",    # OpenCode Go 2x tier; thinking=max
    "thinking": True,
    "min_interval_sec": 900,   # don't call the LLM more often than this
    "drawdown_window": 96,
    "drawdown_pct": 0.10,    # offline fallback buy threshold (dip size)
    "target_pct": 0.05,
    "stop_pct": 0.04,
    "max_positions_total": 4,
    "risk_per_trade": 0.02,    # risk 2% of equity per trade on the stop
    "fee_rate": 0.014,         # ~1.4% round trip
}

PLAYBOOK = (
    "You are a disciplined senior crypto quant trader. You reason first, "
    "then reply ONLY with a JSON object, no prose outside it.\n"
    "Non-negotiable rules you MUST enforce:\n"
    "1. RISK FIRST: never risk more than the size implied by the stop and "
    "the account risk bound. If the setup does not clearly clear the ~1.4% "
    "round-trip fee (fee_rate) after slippage, answer HOLD.\n"
    "2. FEE LITERACY: buying/selling costs fee_rate. Only act when the "
    "expected edge in the brief clearly exceeds fee_rate (prefer a margin "
    ">= 2x the fee). Small expected moves are not worth trading.\n"
    "3. REGIME AWARENESS: if the brief says a downtrend is accelerating or "
    "volatility is extreme/spiky downwards, do NOT buy a falling knife; "
    "wait or answer HOLD.\n"
    "4. ORDER FLOW: prefer entries where buy-side flow is turning up after "
    "a dip (buyers stepping in), not when selling dominates.\n"
    "5. CONVICTION: only BUY with positive net expected edge after fees; "
    "SELL an existing winner to lock profit if the brief says distribution "
    "is starting; HOLD otherwise. Do not overtrade.\n"
    "Answer JSON schema:\n"
    "{\n"
    '  "action": "BUY" | "SELL" | "HOLD",\n'
    '  "symbol": "<pair or null>",\n'
    '  "reason": "<2-4 sentence explanation of the decision>",\n'
    '  "edge_pct": <expected net % after fees, or 0>,\n'
    '  "risk_assessment": "<one line about the downside>",\n'
    '  "confidence": <0.0 to 1.0>\n'
    "}\n"
)


class LLMTraderStrategy(Strategy):
    name = "llm_trader"

    def __init__(self, params=None):
        super().__init__(params)
        self.p = {**DEFAULTS, **(params or {})}
        self._client = LLMClient(base_url=self.p["base_url"],
                                 model=self.p["model"])
        self._last_llm_ts: dict[str, float] = {}
        self._reason: str | None = None
        self._session = requests.Session()

    def warmup_bars(self) -> int:
        return int(self.p["drawdown_window"]) + 5

    # ------------------------------------------------ situation brief
    def _brief(self, df) -> str:
        p = self.p
        close = df["close"]
        high, low = df["high"], df["low"]
        c = float(close.iloc[-1])
        span = (df.index[-1] - df.index[0]).total_seconds()
        bar_h = max(1.0, span / max(1, len(df) - 1) / 3600.0)
        n_day = max(1, int(round(24.0 / bar_h)))
        ret: dict[str, float] = {}
        for label, bars in (("~1h", max(1, n_day // 24)),
                            ("~24h", n_day),
                            ("~7d", n_day * 7)):
            if bars < len(close):
                ret[label] = round(float(close.iloc[-1] / close.iloc[-1 - bars] - 1.0) * 100, 2)
        rsi14 = float(rsi(close, 14).iloc[-1])
        ema_short = float(ema(close, 12).iloc[-1])
        s200 = sma(close, 200)
        sma200 = float(s200.iloc[-1]) if not s200.isna().iloc[-1] else 0.0
        rolling_high = close.rolling(int(p["drawdown_window"]), min_periods=1).max().iloc[-1]
        dd = (c / rolling_high - 1.0) * 100 if rolling_high else 0.0
        atr_pct = float(atr(high, low, close, 14).iloc[-1] / c * 100) if c else 0.0
        # order-flow buy ratio if available live
        flow_note = "n/a (offline)"
        pair = df.attrs.get("pair", "")
        if self._client.configured and pair:
            try:
                trades = fetch_recent_trades(pair, limit=50, session=self._session)
                buy = sum(t["price"] * t["size"] for t in trades
                          if t.get("side", "").upper() in ("BUY", "BID"))
                sell = sum(t["price"] * t["size"] for t in trades
                           if t.get("side", "").upper() in ("SELL", "ASK"))
                ratio = buy / sell if sell > 0 else (9.9 if buy > 0 else 1.0)
                flow_note = f"live buy:sell ${buy:.1f}:${sell:.1f} -> ratio {ratio:.2f}"
            except Exception as exc:  # noqa: BLE001
                flow_note = f"flow err: {exc}"
        return (
            f"Pair {pair} @ {c:,.2f}\n"
            f"drawdown from {p['drawdown_window']}-bar high: {dd:+.2f}%\n"
            f"returns: {ret}\n"
            f"RSI(14): {rsi14:.1f} | price vs EMA12: "
            f"{'above' if c >= ema_short else 'below'} | "
            f"vs SMA200: {'above' if sma200 else 'n/a'} | ATR% {atr_pct:.2f}\n"
            f"order flow: {flow_note}\n"
            f"risk bounds: fee round-trip {p['fee_rate']*100:.1f}%, "
            f"target +{p['target_pct']*100:.1f}%, stop -{p['stop_pct']*100:.1f}%\n"
        )

    # ------------------------------------------------ decision
    def _decide_llm(self, df, brief: str) -> tuple[str, str]:
        try:
            resp = self._client.complete(PLAYBOOK, brief, thinking=self.p["thinking"])
            decision = parse_decision_json(resp["content"])
        except Exception as exc:  # noqa: BLE001
            return "HOLD", f"LLM call failed -> no trade: {exc}"
        action = str(decision.get("action", "HOLD")).upper()
        if action not in ("BUY", "SELL", "HOLD"):
            action = "HOLD"
        reason = decision.get("reason", "")
        reason += (f" | risk: {decision.get('risk_assessment','')}"
                   f" | edge {decision.get('edge_pct','?')}% "
                   f"| conf {decision.get('confidence','?')}")
        return action, reason

    def compute_signals(self, df: pd.DataFrame, live: bool = False) -> "object":
        """Return a valid 1/-1/0 signal series per the Strategy contract.

        Offline/backtest: deterministic dip-recovery signal via
        _decide_offline. Live: entry/exit are driven by execute() (the
        LLM decision), so we emit 0 here to keep the contract satisfied
        without double-trading.
        """
        import pandas as pd
        if not live:
            state = self._decide_offline(df)
            code = {"BUY": 1, "SELL": -1}.get(state, 0)
            return pd.Series([code] * len(df), index=df.index, dtype=int)
        return pd.Series(0, index=df.index, dtype=int)

    def _decide_offline(self, df) -> str:
        """Deterministic fallback when no key / no network: buy a dip with
        turning momentum; sell after recovering toward the 200-SMA. Keeps
        the strategy honest and testable offline."""
        p = self.p
        close = df["close"]
        rolling_high = close.rolling(int(p["drawdown_window"]), min_periods=1).max()
        dd = float(close.iloc[-1] / rolling_high.iloc[-1] - 1.0)
        half = max(2, int(p["drawdown_window"]) // 2)
        mom = float(close.iloc[-1] / close.iloc[-half] - 1.0) if half < len(close) else 0.0
        sma_f = sma(close, 200)
        above_sma = (not sma_f.isna().iloc[-1]) and float(close.iloc[-1]) >= float(sma_f.iloc[-1])
        if dd <= -float(p["drawdown_pct"]) * 0.6 and mom > 0.02:
            return "BUY"
        if above_sma:
            return "SELL"
        return "HOLD"

    def decide(self, df) -> tuple[str, str]:
        """Live entry: throttle LLM calls, build a brief, ask the model
        (or fall back offline)."""
        pair = df.attrs.get("pair", "?")
        now = time.time()
        if now - self._last_llm_ts.get(pair, 0) < self.p["min_interval_sec"]:
            return "HOLD", "throttled (LLM call cadence)"
        self._last_llm_ts[pair] = now
        price = float(df["close"].iloc[-1])
        brief = self._brief(df)
        if self._client.configured:
            try:
                action, reason = self._decide_llm(df, brief)
            except Exception as exc:  # noqa: BLE001
                return "HOLD", f"LLM error -> no trade: {exc}"
        else:
            action, reason = "HOLD", "no OPENCODE_GO_KEY -> offline fallback (no LLM)"
        self._reason = (
            f"[llm_trader {pair}] {action} @ {price:,.2f}\n"
            f"   brief:\n{brief}   decision: {reason}")
        return action, self._reason

    def last_reason(self) -> str | None:
        return self._reason

    def execute(self, account, pair: str, df: pd.DataFrame,
                price: float, ts: int, live: bool = False) -> dict | None:
        p = self.p
        pos = account.positions.get(pair)
        # Manage exit deterministically (target/stop) so a bad LLM call
        # cannot blow the stop away.
        if pos is not None:
            avg = pos.entry_cost / pos.qty if pos.qty else price
            if price >= avg * (1.0 + p["target_pct"]):
                c = account.close_position(pair, price, ts)
                return {**c, "reason": f"target +{p['target_pct']*100:.0f}% hit (deterministic)"} if c else None
            if price <= avg * (1.0 - p["stop_pct"]):
                c = account.close_position(pair, price, ts)
                return {**c, "reason": f"stop -{p['stop_pct']*100:.0f}% hit (deterministic)"} if c else None
            return None
        # No position: get a decision (LLM in live, deterministic offline).
        action, reason = self.decide(df)
        self._reason = reason
        if action != "BUY" or len(account.positions) >= p["max_positions_total"]:
            if account.positions and action == "SELL":
                c = account.close_position(pair, price, ts)
                if c:
                    return {**c, "reason": reason}
            return None
        opened = account.open_position(pair, price, ts)
        if opened is not None:
            return {"action": "buy", "qty": opened.qty, "fee": opened.entry_fee,
                    "price": price, "reason": reason}
        return None


__all__ = ["LLMTraderStrategy"]
