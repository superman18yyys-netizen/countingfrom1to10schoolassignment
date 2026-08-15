"""Fee-aware trade gate: the single source of truth for cost-aware
trade decisions (see docs/superpowers/specs/2026-08-15-fee-aware-trading-design.md).

Every bot's entries must clear the round-trip cost with margin, exits
never realize a sub-cost "win", and repeated fee-bleeding pauses the
bot. The gate itself is STATELESS — adapters (signal rewriter for
backtests, GatedAccount proxy for live) own all state.

Failure asymmetry (adapters): entries fail CLOSED (broken math never
opens a position), exits fail OPEN (selling is always possible).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

from bot.config import FeeAwareConfig

# Module fee model — entry points sync this from strategies.yaml so the
# gate's math always matches the engine's fills. Defaults = verified
# Coinbase Advanced Trade retail tier (0.6% taker) + 0.1% slippage.
_TAKER_FEE = 0.006
_SLIPPAGE = 0.001


def set_fee_model(taker_fee: float, slippage: float) -> None:
    """Sync the gate's cost model with the run's actual config."""
    global _TAKER_FEE, _SLIPPAGE
    _TAKER_FEE = float(taker_fee)
    _SLIPPAGE = float(slippage)


def round_trip_cost() -> float:
    """Full round-trip cost as a fraction (2 x (taker + slippage))."""
    return 2.0 * (_TAKER_FEE + _SLIPPAGE)


def fee_pair() -> Tuple[float, float]:
    """(taker_fee, slippage) from the synced fee model."""
    return (_TAKER_FEE, _SLIPPAGE)


@dataclass
class GateParams:
    margin: float = 1.5
    expected_hold_bars: int = 16
    min_profit_mult: float = 1.0
    stop_mult: float = 3.0
    max_hold_bars: int = 96
    breaker_trades: int = 8
    cooldown_bars: int = 24
    fee_budget_pct: float = 0.02
    position_fraction: float = 0.25

    @classmethod
    def from_config(cls, cfg: Optional[FeeAwareConfig] = None) -> "GateParams":
        c = cfg or FeeAwareConfig.from_yaml()
        return cls(margin=c.margin, expected_hold_bars=c.expected_hold_bars,
                   min_profit_mult=c.min_profit_mult, stop_mult=c.stop_mult,
                   max_hold_bars=c.max_hold_bars, breaker_trades=c.breaker_trades,
                   cooldown_bars=c.cooldown_bars, fee_budget_pct=c.fee_budget_pct,
                   position_fraction=c.position_fraction)


@dataclass
class GateContext:
    atr_pct: float              # ATR(14)/price on the decision bar
    rtc: float                  # round-trip cost fraction
    unrealized_pct: float = 0.0
    hold_bars: int = 0
    recent_gross_pcts: List[float] = field(default_factory=list)
    fees_paid_window: float = 0.0
    capital: float = 0.0
    cooldown_bars_left: int = 0


@dataclass
class TradeGate:
    """Stateless fee-brain. Pure decisions over a GateContext."""

    p: GateParams = field(default_factory=lambda: GateParams.from_config())

    def allow_entry(self, ctx: GateContext) -> Tuple[bool, str]:
        if ctx.cooldown_bars_left > 0:
            return False, f"circuit-breaker cooldown ({ctx.cooldown_bars_left} bars left)"
        if ctx.capital > 0 and ctx.fees_paid_window > self.p.fee_budget_pct * ctx.capital:
            return False, "fee budget exhausted (24h window)"
        hurdle = self.p.margin * ctx.rtc
        expected = math.sqrt(self.p.expected_hold_bars) * ctx.atr_pct
        if expected < hurdle:
            return False, (f"EV: expected move {expected * 100:.2f}% "
                           f"< hurdle {hurdle * 100:.2f}%")
        return True, "entry clears round-trip cost"

    def allow_exit(self, ctx: GateContext) -> Tuple[bool, str]:
        if ctx.unrealized_pct >= self.p.min_profit_mult * ctx.rtc:
            return True, "profit clears round-trip cost"
        if ctx.unrealized_pct <= -self.p.stop_mult * ctx.rtc:
            return True, "disaster stop"
        return False, (f"exit deferred: {ctx.unrealized_pct * 100:+.2f}% "
                       f"inside cost band")

    def force_exit(self, ctx: GateContext) -> Tuple[bool, str]:
        if ctx.unrealized_pct <= -self.p.stop_mult * ctx.rtc:
            return True, "disaster stop"
        if ctx.hold_bars >= self.p.max_hold_bars:
            return True, f"time stop ({ctx.hold_bars} bars)"
        return False, ""


class GatedAccount:
    """Live/zoo adapter: wraps a PaperAccount so every open/close the
    base strategy requests routes through the TradeGate.

    State (recent gross returns, 24h fee ledger, cooldown) lives here,
    in-process; the recent-returns window is re-seeded from the wrapped
    account's trade history so the circuit breaker survives CI restarts.
    """

    def __init__(self, account, gate: TradeGate, bar_sec: int = 3600):
        self._acc = account
        self.gate = gate
        self.bar_sec = bar_sec
        rtc = round_trip_cost()
        self.recent_gross: Deque[float] = deque(
            [p + rtc for p in account.trade_pcts[-gate.p.breaker_trades:]],
            maxlen=gate.p.breaker_trades)
        self.fee_ledger: List[Tuple[int, float]] = []  # (ts, fee$)
        self.cooldown = 0
        self._atr_pct = 0.0
        self.block_reason: Optional[str] = None
        # chassis hooks: per-entry sizing + hard entry block (regime)
        self.next_fraction: Optional[float] = None
        self.entries_blocked: bool = False

    # ---- context ----------------------------------------------------
    def set_bar_context(self, atr_pct: float, ts: int = 0) -> None:
        """Called once per closed candle with this bar's context."""
        self._atr_pct = float(atr_pct)
        if self.cooldown > 0:
            self.cooldown -= 1
        if ts:
            self._drop_stale_fees(ts)

    def _drop_stale_fees(self, now_ts: int, window_sec: int = 86400) -> None:
        self.fee_ledger = [(t, f) for (t, f) in self.fee_ledger
                           if now_ts - t <= window_sec]

    def _fees_window(self) -> float:
        return sum(f for _, f in self.fee_ledger)

    def _unrealized(self, pos, price: float) -> float:
        """Net-of-cost unrealized return — matches PaperAccount math."""
        fill = price * (1.0 - self._acc.slippage)
        proceeds = fill * pos.qty * (1.0 - self._acc.taker_fee)
        basis = pos.entry_cost + pos.entry_fee
        return proceeds / basis - 1.0 if basis else 0.0

    # ---- gated account API ------------------------------------------
    def open_position(self, pair: str, price: float, ts: int,
                      fraction: Optional[float] = None):
        if self.entries_blocked:
            self.block_reason = "chassis: regime blocks entries"
            return None
        frac = fraction if fraction is not None else self.next_fraction
        self.next_fraction = None
        try:
            ctx = GateContext(atr_pct=self._atr_pct, rtc=round_trip_cost(),
                              recent_gross_pcts=list(self.recent_gross),
                              fees_paid_window=self._fees_window(),
                              capital=self._acc.capital,
                              cooldown_bars_left=self.cooldown)
            ok, why = self.gate.allow_entry(ctx)
        except Exception:  # noqa: BLE001 — entries fail CLOSED
            self.block_reason = "gate error -> entry blocked (fail closed)"
            return None
        if not ok:
            self.block_reason = f"gate: {why}"
            return None
        self.block_reason = None
        pos = self._acc.open_position(pair, price, ts, fraction=frac)
        if pos is not None:
            self.fee_ledger.append((ts, pos.entry_fee))
        return pos

    def close_position(self, pair: str, price: float, ts: int):
        pos = self._acc.positions.get(pair)
        if pos is None:
            return None
        try:
            ctx = GateContext(atr_pct=self._atr_pct, rtc=round_trip_cost(),
                              unrealized_pct=self._unrealized(pos, price),
                              recent_gross_pcts=list(self.recent_gross),
                              fees_paid_window=self._fees_window(),
                              capital=self._acc.capital,
                              cooldown_bars_left=self.cooldown)
            ok, why = self.gate.allow_exit(ctx)
        except Exception:  # noqa: BLE001 — exits fail OPEN
            ok, why = True, "gate error -> exit allowed (fail open)"
        if not ok:
            self.block_reason = f"gate: {why}"
            return None
        self.block_reason = None
        closed = self._acc.close_position(pair, price, ts)
        if closed is not None:
            self._record_close(closed, ts)
        return closed

    def _record_close(self, closed: dict, ts: int = 0) -> None:
        """Update breaker state after any realized close (also used by
        FeeAwareStrategy for forced stop/time exits)."""
        self.fee_ledger.append((ts, closed.get("exit_fee", 0.0)))
        gross = closed["pnl_pct"] + round_trip_cost()
        self.recent_gross.append(gross)
        if (len(self.recent_gross) >= self.gate.p.breaker_trades
                and sum(self.recent_gross) / len(self.recent_gross) < 0):
            self.cooldown = self.gate.p.cooldown_bars

    # ---- delegation --------------------------------------------------
    def __getattr__(self, name: str):
        acc = self.__dict__.get("_acc")
        if acc is None:
            raise AttributeError(name)
        return getattr(acc, name)
