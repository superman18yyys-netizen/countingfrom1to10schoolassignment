"""Zoo roster: one bot per community-classic model, fixed canonical
parameters, each starting with the same capital. No evolution here —
the zoo measures how published strategies perform AS-IS on live data.
"""
from __future__ import annotations

from typing import List, Tuple

from bot.paper.account import PaperAccount
from bot.swarm.genome import Genome
from bot.swarm.population import Agent, Population

# (bot id, strategy name, params, account overrides)
ROSTER: List[Tuple[str, str, dict, dict]] = [
    ("momentum", "momentum", {}, {}),
    ("mean_reversion", "mean_reversion", {}, {}),
    ("macd_cross", "macd_cross", {}, {}),
    ("golden_cross", "golden_cross", {}, {}),
    ("donchian_breakout", "donchian_breakout", {}, {}),
    ("rsi2", "rsi2", {}, {}),
    ("stochastic_reversion", "stochastic_reversion", {}, {}),
    ("bbands_breakout", "bbands_breakout", {}, {}),
    ("grid_trader", "grid_trader", {}, {}),
    ("dca_bot", "dca_bot", {}, {"allow_averaging": True}),
    ("ml_trend", "ml_trend", {}, {}),       # best-practice two-stage model (diagnostic)
    ("deep_value", "deep_value", {}, {}),   # drawdown-reversion dip buyer (diagnostic)
    ("order_flow", "order_flow", {}, {}),   # 'who is buying' order-flow strength
    ("llm_trader", "llm_trader", {}, {}),   # LLM-orchestrated trader (needs OPENCODE_GO_KEY)
    ("hold_cycle", "hold_cycle", {}, {}),   # few/big/slow trend-cycle holder
    ("fade_extreme", "fade_extreme", {}, {}),  # buy capitulation, sell distribution
    ("deep_recovery", "deep_recovery", {}, {}),  # aggressive confirmed-dip recovery
    ("guarded_momentum", "guarded_momentum", {}, {}),  # momentum + fee/regime guards
    ("guarded_macd", "guarded_macd", {}, {}),      # macd + guards
    ("guarded_donchian", "guarded_donchian", {}, {}),  # donchian + guards
    ("guarded_rsi2", "guarded_rsi2", {}, {}),      # rsi2 + guards (range mode)
    ("guarded_stochastic", "guarded_stochastic", {}, {}),  # stochastic + guards
    ("guarded_grid", "guarded_grid", {}, {}),      # grid + guards (range mode)
    ("deep_recovery_v2", "deep_recovery_v2", {}, {}),  # v2: adaptive + volume confirm
    ("adaptive_grid", "adaptive_grid", {}, {}),    # v2: ATR-scaled fee-clearing grid
    ("consensus", "consensus", {}, {}),            # multi-signal agreement combiner
    ("trend_runner", "trend_runner", {}, {}),      # trailing, vol-scaled trend follower
]


def build_zoo_population(pairs: List[str], granularity: str,
                         capital: float, fee_cfg: dict) -> Population:
    pop = Population(pairs=pairs, granularity=granularity,
                     capital=capital, fee_cfg=fee_cfg, strategy="zoo")
    pop.agents = []
    for bot_id, strategy_name, params, overrides in ROSTER:
        genome = Genome(id=bot_id, strategy=strategy_name, params=dict(params))
        acc = PaperAccount(capital=capital, **fee_cfg)
        for key, value in overrides.items():
            setattr(acc, key, value)
        pop.agents.append(Agent(genome=genome, account=acc, equity=capital))
    return pop
