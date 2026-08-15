"""ChassisStrategy: the mechanical trading rule system (see
docs/superpowers/specs/2026-08-15-chassis-design.md).

Six fixed layers around every base strategy:
  1. CONTEXT   — MarketContext, the 1-year look (bot/market/context.py)
  2. REGIME    — each play family trades only in its regimes
  3. FEE GATE  — inherited TradeGate (EV / breaker / budget)
  4. SIGNAL    — the base strategy (the ONLY evolvable surface)
  5. SIZING    — vol-target x conviction: risk ~1% of equity per trade,
                 sized up (max 1.5x) when context favors the play
  6. ORDER     — execute with that fraction; exits always pass

Humans own layers 1-3 and 5-6; evolution tunes only layer 4.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from bot.market import (build_context, classify_regime, CRASH, DOWN,
                        RANGE, UP)
from bot.strategies.base import Strategy
from bot.strategies.fee_aware import FeeAwareStrategy

# ---- frozen chassis constants (NOT tunable — by design) ----------------
TARGET_RISK = 0.01          # 1% of equity risked per trade
FRAC_MIN, FRAC_MAX = 0.05, 0.50
CONVICTION_MAX_MULT = 0.5   # conviction can add at most +50% size
VOL_DROUGHT_PCTL = 0.25     # entries blocked when ATR pctile (90d) below this

# ---- play families and their regime allowlists --------------------------
FAMILY_ALLOW: dict = {
    "trend": {UP},
    "range": {RANGE, UP},
    "value": {DOWN, RANGE},
    "capitulation": {CRASH},
    "self": None,            # None = any regime (sizing still applies)
}

STRATEGY_FAMILY: dict = {
    # trend
    "momentum": "trend", "macd_cross": "trend", "golden_cross": "trend",
    "donchian_breakout": "trend", "trend_runner": "trend",
    "hold_cycle": "trend", "ml_trend": "trend",
    # range
    "mean_reversion": "range", "rsi2": "range",
    "stochastic_reversion": "range", "grid_trader": "range",
    "bbands_breakout": "range", "adaptive_grid": "range",
    # value (fade_extreme additionally buys CRASH via "capitulation")
    "deep_value": "value", "deep_recovery": "value",
    "deep_recovery_v2": "value", "dca_bot": "value",
    "fade_extreme": "value+capitulation",
    # self-directed (chassis sizes but never blocks)
    "llm_trader": "self", "order_flow": "self", "consensus": "self",
}


def family_of(strategy_name: str) -> str:
    fam = STRATEGY_FAMILY.get(strategy_name)
    if fam is None and strategy_name.startswith("guarded_"):
        fam = STRATEGY_FAMILY.get(strategy_name[len("guarded_"):], "trend")
    return fam or "self"


def _allowed_regimes(fam: str) -> Optional[set]:
    """Effective allowlist; composite families union their rows."""
    if "+" in fam:
        out: set = set()
        for part in fam.split("+"):
            allowed = FAMILY_ALLOW[part]
            if allowed is None:
                return None
            out |= allowed
        return out
    return FAMILY_ALLOW[fam]


def context_score(fam: str, row: dict) -> float:
    """How strongly the context favors THIS family's play, 0..1."""
    if fam in ("trend",):
        return max(0.0, min(1.0, (row.get("trend_frac_pos", 0.5) - 0.5) / 0.25))
    if fam == "range":
        return max(0.0, min(1.0, 1.0 - abs(row.get("sma_dist", 0.0)) / 0.03))
    if fam == "value":
        return max(0.0, min(1.0, -row.get("dd_from_high", 0.0) / 0.40))
    if fam == "capitulation":
        return max(0.0, min(1.0, row.get("atr_pctile", 0.5)))
    return 0.5   # self-directed: neutral conviction


def size_fraction(fam: str, row: dict) -> float:
    """Layer 5: vol-target base x conviction multiplier, clamped."""
    atr_pct = max(row.get("atr_pct", 0.0), 1e-6)
    base = TARGET_RISK / atr_pct
    score = context_score(fam.split("+")[0], row)
    conviction = 1.0 + CONVICTION_MAX_MULT * score * row.get(
        "context_confidence", 1.0)
    return max(FRAC_MIN, min(FRAC_MAX, base * conviction))


def in_vol_drought(row: dict) -> bool:
    """True when ATR is in its lowest quartile vs the trailing 90 days
    (empirically the regime where entries cannot clear fees; see
    context.py). Blocks entries for every family — in a drought the
    correct human decision is to stand aside.

    NOTE: a surge-override variant (>=5% surge defeats the veto) was
    A/B tested on 2y data (Aug 2026) and REJECTED: fleet total went
    +89.6% -> +88.2% with worse drawdown for 2 of 5 bots. The
    drought-blocked swings are not worth the false entries it lets in."""
    return float(row.get("atr_pctile_long", 1.0)) < VOL_DROUGHT_PCTL


class ChassisStrategy(FeeAwareStrategy):
    """Layers 2 and 5 bolted onto the fee-aware wrapper. Pass
    ``chassis_off: True`` in gate_params to disable both (used by
    machinery tests and A/B baselines)."""
    name = "chassis"

    def __init__(self, base: Strategy, gate_params: Optional[dict] = None):
        gate_params = dict(gate_params or {})
        self.chassis_off = bool(gate_params.pop("chassis_off", False))
        super().__init__(base, gate_params)
        self.name = base.name
        self._ctx_cache: dict = {}

    # ---- layer 1 (context, cached per df identity) ----------------------
    def _build_context(self, df: pd.DataFrame):
        if self.chassis_off:
            return None
        key = (id(df), len(df))
        cached = self._ctx_cache.get(key)
        if cached is None:
            cached = build_context(df)
            self._ctx_cache.clear()      # one window at a time
            self._ctx_cache[key] = cached
        return cached

    # ---- layer 2 (regime allowlist + vol-drought block) ----------------
    def _regime_ok(self, ctx, i: int) -> bool:
        if self.chassis_off:
            return True
        row = ctx.iloc[i]
        if in_vol_drought(row):
            return False
        allowed = _allowed_regimes(family_of(self._base.name))
        if allowed is None:
            return True
        reg = int(classify_regime(ctx).iloc[i])
        return reg in allowed

    # ---- layer 5 (sizing) -------------------------------------------------
    def _entry_fraction(self, ctx, i: int):
        if self.chassis_off:
            return None
        return size_fraction(family_of(self._base.name),
                             {k: float(v) for k, v in ctx.iloc[i].items()})

    def _configure_proxy(self, proxied, ctx, df: pd.DataFrame) -> None:
        if self.chassis_off or ctx is None or len(ctx) == 0:
            return
        fam = family_of(self._base.name)
        allowed = _allowed_regimes(fam)
        row = {k: float(v) for k, v in ctx.iloc[-1].items()}
        reg = int(classify_regime(ctx).iloc[-1])
        if in_vol_drought(row):
            proxied.entries_blocked = True
        elif allowed is not None and reg not in allowed:
            proxied.entries_blocked = True
        else:
            proxied.entries_blocked = False
            proxied.next_fraction = size_fraction(fam, row)
