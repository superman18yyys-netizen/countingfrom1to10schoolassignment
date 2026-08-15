"""Bot genomes: strategy choice + tunable parameters + mutation.

The swarm evolves parameter tunings, not code: every bot keeps the same
strategy type (per the experiment design) and only its numeric tuning
mutates. Bounds keep mutations inside sane, research-backed ranges.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

# param name -> (low, high, type)
PARAM_BOUNDS: Dict[str, Dict[str, Tuple[float, float, type]]] = {
    "mean_reversion": {
        "rsi_period": (7, 28, int),
        "bb_period": (10, 40, int),
        "bb_std": (1.5, 3.0, float),
        "oversold": (20, 40, int),
        "overbought": (60, 80, int),
        "exit_rsi": (45, 70, int),
    },
    "momentum": {
        "ema_fast": (5, 24, int),
        "ema_slow": (15, 60, int),
        "trend_ema": (50, 400, int),
    },
    "ml_trend": {
        "horizon": (3, 24, int),
        "min_gain": (0.008, 0.025, float),
        "regime_horizon": (12, 72, int),
        "regime_tol": (0.002, 0.012, float),
        "regime_up": (0.40, 0.65, float),
        "buy": (0.50, 0.75, float),
        "exit": (0.35, 0.55, float),
    },
    "trend_runner": {
        "trend_sma": (40, 400, int),
        "atr_period": (7, 30, int),
        "atr_mult": (1.5, 5.0, float),
        "trail_bars": (24, 300, int),
        "atr_hurdle_pct": (0.002, 0.010, float),
    },
    "donchian_breakout": {
        "entry_period": (10, 55, int),
        "exit_period": (5, 20, int),
    },
    "rsi2": {
        "entry_rsi": (5, 25, int),
        "exit_rsi": (60, 95, int),
        "trend_sma": (100, 300, int),
    },
    "golden_cross": {
        "fast": (20, 80, int),
        "slow": (120, 300, int),
    },
    "bbands_breakout": {
        "bb_period": (15, 30, int),
        "bb_std": (1.8, 2.5, float),
        "squeeze_pct": (0.1, 0.3, float),
    },
    "grid_trader": {
        "reference_bars": (48, 168, int),
        "grid_step": (0.01, 0.03, float),
    },
}


@dataclass
class Genome:
    id: str
    strategy: str
    params: Dict[str, Any] = field(default_factory=dict)
    lineage: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"id": self.id, "strategy": self.strategy,
                "params": self.params, "lineage": self.lineage}

    @classmethod
    def from_dict(cls, d: dict) -> "Genome":
        return cls(id=d["id"], strategy=d["strategy"],
                   params=dict(d.get("params", {})),
                   lineage=list(d.get("lineage", [])))


def random_params(strategy: str, rng: random.Random) -> Dict[str, Any]:
    bounds = PARAM_BOUNDS[strategy]
    out: Dict[str, Any] = {}
    for name, (lo, hi, typ) in bounds.items():
        if typ is int:
            out[name] = rng.randint(int(lo), int(hi))
        else:
            out[name] = round(rng.uniform(lo, hi), 3)
    return out


def make_genome(strategy: str, genome_id: str, rng: random.Random,
                params: Dict[str, Any] | None = None,
                lineage: list[str] | None = None) -> Genome:
    return Genome(id=genome_id, strategy=strategy,
                  params=params if params is not None else random_params(strategy, rng),
                  lineage=lineage or [])


def mutate(genome: Genome, genome_id: str, rng: random.Random,
           strength: float = 0.15) -> Genome:
    """Copy a genome with slightly perturbed tuning (bounded)."""
    bounds = PARAM_BOUNDS[genome.strategy]
    new_params: Dict[str, Any] = {}
    for name, value in genome.params.items():
        if name not in bounds or rng.random() > 0.75:
            new_params[name] = value
            continue
        lo, hi, typ = bounds[name]
        if typ is int:
            span = max(1, int(round((hi - lo) * strength)))
            v = int(value) + rng.randint(-span, span)
            new_params[name] = int(min(hi, max(lo, v)))
        else:
            v = float(value) * (1.0 + rng.uniform(-strength, strength * 1.2))
            new_params[name] = round(float(min(hi, max(lo, v))), 3)
    lineage = [genome.id] + genome.lineage[:4]
    new_params = _repair(genome.strategy, new_params)
    return Genome(id=genome_id, strategy=genome.strategy,
                  params=new_params, lineage=lineage)


def _repair(strategy: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Enforce ordering constraints between related parameters."""
    p = dict(params)
    if strategy == "momentum" and "ema_fast" in p and "ema_slow" in p:
        if p["ema_fast"] >= p["ema_slow"]:
            lo, hi, _ = PARAM_BOUNDS[strategy]["ema_slow"]
            p["ema_slow"] = int(min(hi, p["ema_fast"] + 5))
    if strategy == "mean_reversion" and all(k in p for k in
                                            ("oversold", "exit_rsi", "overbought")):
        if not (p["oversold"] < p["exit_rsi"] < p["overbought"]):
            p["exit_rsi"] = (p["oversold"] + p["overbought"]) // 2
    if strategy == "ml_trend":
        # regime_horizon must exceed horizon (regime is the slower view)
        if p.get("regime_horizon", 0) <= p.get("horizon", 0):
            p["regime_horizon"] = p.get("horizon", 6) + 6
        # buy threshold must exceed exit threshold (enter higher, leave lower)
        if p.get("buy", 0.5) <= p.get("exit", 0.5):
            p["buy"] = min(0.75, p.get("exit", 0.5) + 0.05)
    if strategy == "trend_runner":
        # trail window must comfortably exceed the ATR lookback so the
        # rolling-high reference actually spans the holding period.
        if p.get("trail_bars", 0) <= p.get("atr_period", 0):
            p["trail_bars"] = int(p.get("atr_period", 14)) * 2
    if strategy == "donchian_breakout":
        # exit channel must be faster than the entry channel
        if p.get("exit_period", 0) >= p.get("entry_period", 0):
            p["exit_period"] = max(5, int(p.get("entry_period", 20)) // 2)
    if strategy == "rsi2":
        if p.get("entry_rsi", 0) >= p.get("exit_rsi", 0):
            p["entry_rsi"] = max(5, int(p.get("exit_rsi", 70)) - 45)
    return p