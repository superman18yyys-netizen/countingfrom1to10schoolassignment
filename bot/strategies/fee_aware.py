"""FeeAwareStrategy: wraps any strategy so every trade decision clears
the round-trip cost (see docs/superpowers/specs/2026-08-15-fee-aware-trading-design.md).

Two adapters, one policy (bot.trade_gate.TradeGate):
  * compute_signals -> causal signal REWRITE for the backtest engine
    (suppress gated entries, defer sub-cost exits, inject stop/time exits)
  * execute -> GatedAccount proxy around the live/paper account, plus
    forced stop/time exits against the real position

The wrapper preserves the base's ``name`` so reports, boards and state
files are unchanged.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from bot.indicators.ta import atr
from bot.strategies.base import Strategy
from bot.trade_gate import (GatedAccount, GateContext, GateParams,
                            TradeGate, fee_pair, round_trip_cost)


def _bar_sec(df: pd.DataFrame) -> int:
    try:
        return int((df.index[-1] - df.index[-2]).total_seconds())
    except Exception:  # noqa: BLE001
        return 3600


class FeeAwareStrategy(Strategy):
    name = "fee_aware"

    def __init__(self, base: Strategy, gate_params: Optional[dict] = None):
        super().__init__({})
        self._base = base
        self.name = base.name          # reports/boards unchanged
        self.gate = TradeGate(GateParams(**(gate_params or {})))
        self._proxies: Dict[int, GatedAccount] = {}
        self._last_reason: Optional[str] = None

    # ---- forwarding ---------------------------------------------------
    def fit(self, df: pd.DataFrame) -> None:
        return self._base.fit(df)

    def warmup_bars(self) -> int:
        return self._base.warmup_bars()

    def last_reason(self) -> Optional[str]:
        return self._last_reason

    # ---- chassis hooks (base: permissive no-ops) ----------------------
    def _build_context(self, df: pd.DataFrame):
        """Chassis layer 1: override to compute MarketContext (cached)."""
        return None

    def _regime_ok(self, ctx, i: int) -> bool:
        """Chassis layer 2: override to apply the regime allowlist."""
        return True

    def _entry_fraction(self, ctx, i: int):
        """Chassis layer 5: override to size entries (vol-target x
        conviction). Return None for the account default."""
        return None

    def _configure_proxy(self, proxied, ctx, df: pd.DataFrame) -> None:
        """Chassis live-path hook: set entry block/fraction on the proxy
        before the base strategy executes."""
        return None

    # ---- backtest path: causal rewrite --------------------------------
    def compute_signals(self, df: pd.DataFrame, live: bool = False) -> pd.Series:
        base_sig = self._base.compute_signals(df, live=live)
        self._entry_fractions = None
        try:
            return self._rewrite(base_sig, df)
        except Exception:  # noqa: BLE001 — fail closed for entries,
            # fail open for exits: keep only the base's exit signals
            return base_sig.where(base_sig == -1, 0).astype(int)

    def _rewrite(self, sig: pd.Series, df: pd.DataFrame) -> pd.Series:
        gate, p = self.gate, self.gate.p
        rtc = round_trip_cost()
        fee, slip = fee_pair()
        close = df["close"]
        atr_pct = (atr(df["high"], df["low"], close, 14)
                   / close.replace(0.0, np.nan)).fillna(0.0)
        ctx = self._build_context(df)          # chassis layer 1 (or None)
        fractions = pd.Series(np.nan, index=df.index)
        out = pd.Series(0, index=df.index, dtype=int)
        in_pos = False
        entry_price = 0.0
        entry_i = 0
        recent_gross: list = []
        fee_ledger: list = []      # (ts, fee as fraction of capital)
        cooldown = 0
        for i in range(len(df)):
            if cooldown > 0:
                cooldown -= 1
            px = float(close.iloc[i])
            ts = df.index[i]
            fee_ledger = [(t, f) for (t, f) in fee_ledger
                          if (ts - t).total_seconds() <= 86400]
            s = int(sig.iloc[i]) if not pd.isna(sig.iloc[i]) else 0
            if in_pos:
                unreal = (px * (1.0 - slip) * (1.0 - fee)
                          / (entry_price * (1.0 + slip) * (1.0 + fee)) - 1.0)
                hold = i - entry_i
                forced, _ = gate.force_exit(GateContext(
                    atr_pct=float(atr_pct.iloc[i]), rtc=rtc,
                    unrealized_pct=unreal, hold_bars=hold))
                allowed = forced
                if not allowed and s == -1:
                    allowed, _ = gate.allow_exit(GateContext(
                        atr_pct=float(atr_pct.iloc[i]), rtc=rtc,
                        unrealized_pct=unreal, hold_bars=hold))
                if allowed:
                    out.iloc[i] = -1
                    in_pos = False
                    recent_gross.append(px / entry_price - 1.0)
                    recent_gross = recent_gross[-p.breaker_trades:]
                    fee_ledger.append((ts, rtc * p.position_fraction / 2.0))
                    if (len(recent_gross) >= p.breaker_trades
                            and sum(recent_gross) / len(recent_gross) < 0):
                        cooldown = p.cooldown_bars
            elif s == 1:
                ctx_ok = ctx is None or self._regime_ok(ctx, i)
                if not ctx_ok:
                    continue
                entry_ctx = GateContext(atr_pct=float(atr_pct.iloc[i]),
                                        rtc=rtc,
                                        recent_gross_pcts=recent_gross,
                                        fees_paid_window=sum(f for _, f in fee_ledger),
                                        capital=1.0,
                                        cooldown_bars_left=cooldown)
                ok, _ = gate.allow_entry(entry_ctx)
                if ok:
                    out.iloc[i] = 1
                    in_pos = True
                    entry_price = px
                    entry_i = i
                    fee_ledger.append((ts, rtc * p.position_fraction / 2.0))
                    frac = self._entry_fraction(ctx, i) if ctx is not None else None
                    if frac is not None:
                        fractions.iloc[i] = float(frac)
        if ctx is not None and fractions.notna().any():
            self._entry_fractions = fractions
        return out

    # ---- live/zoo path: proxy + forced exits ---------------------------
    def execute(self, account, pair: str, df: pd.DataFrame,
                price: float, ts: int, live: bool = False) -> Optional[dict]:
        key = id(account)
        proxied = self._proxies.get(key)
        if proxied is None or proxied._acc is not account:
            proxied = GatedAccount(account, self.gate, bar_sec=_bar_sec(df))
            self._proxies[key] = proxied
        try:
            atr_pct = float((atr(df["high"], df["low"], df["close"], 14)
                             / df["close"]).iloc[-1])
        except Exception:  # noqa: BLE001
            atr_pct = 0.0
        proxied.set_bar_context(atr_pct, ts=ts)
        ctx = self._build_context(df)          # chassis layer 1 (or None)
        self._configure_proxy(proxied, ctx, df)
        result = self._base.execute(proxied, pair, df, price, ts, live=live)
        # forced stop / time exits against the REAL position
        pos = account.positions.get(pair)
        if pos is not None:
            unreal = proxied._unrealized(pos, price)
            hold = max(0, int((ts - pos.entry_ts) / max(1, proxied.bar_sec)))
            forced, why = self.gate.force_exit(GateContext(
                atr_pct=atr_pct, rtc=round_trip_cost(),
                unrealized_pct=unreal, hold_bars=hold))
            if forced:
                closed = account.close_position(pair, price, ts)
                if closed is not None:
                    proxied._record_close(closed, ts)
                    result = {"action": "sell", "qty": closed["qty"],
                              "fee": closed["exit_fee"], "price": price,
                              "pnl": closed["pnl"],
                              "pnl_pct": closed["pnl_pct"]}
        self._last_reason = proxied.block_reason
        return result
