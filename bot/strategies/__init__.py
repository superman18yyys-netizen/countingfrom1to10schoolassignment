"""Strategy registry."""
from __future__ import annotations

from typing import Dict, Optional, Type

from bot.strategies.base import Strategy
from bot.strategies.chassis import ChassisStrategy
from bot.strategies.fee_aware import FeeAwareStrategy
from bot.strategies.community import (BBandsBreakout, DCABot, DonchianBreakout,
                                      GoldenCross, GridTrader, MACDCross,
                                      RSI2, StochasticReversion)
from bot.strategies.deep_value import DeepValueStrategy
from bot.strategies.mean_reversion import MeanReversionStrategy
from bot.strategies.ml import MLOptimizerStrategy
from bot.strategies.ml_trend import MLTrendStrategy
from bot.strategies.momentum import MomentumStrategy
from bot.strategies.order_flow import OrderFlowStrategy
from bot.strategies.llm_trader import LLMTraderStrategy
from bot.strategies.hold_cycle import HoldCycleStrategy
from bot.strategies.fade_extreme import FadeExtremeStrategy
from bot.strategies.deep_recovery import DeepRecoveryStrategy
from bot.strategies.guarded import (GuardedDonchian, GuardedGrid, GuardedMACD,  # noqa: F401
                                    GuardedMomentum, GuardedRSI2,
                                    GuardedStochastic, GuardedStrategy)
from bot.strategies.trend_runner import TrendRunner
from bot.strategies.winners_v2 import AdaptiveGrid, Consensus, DeepRecoveryV2

REGISTRY: Dict[str, Type[Strategy]] = {
    cls.name: cls for cls in (
        MomentumStrategy, MeanReversionStrategy, MLOptimizerStrategy,
        MLTrendStrategy, DeepValueStrategy, OrderFlowStrategy, LLMTraderStrategy,
        HoldCycleStrategy, FadeExtremeStrategy, DeepRecoveryStrategy,
        DeepRecoveryV2, AdaptiveGrid, Consensus, TrendRunner,
        GuardedMomentum, GuardedMACD, GuardedDonchian,
        GuardedRSI2, GuardedStochastic, GuardedGrid,
        MACDCross, GoldenCross, DonchianBreakout, RSI2,
        StochasticReversion, BBandsBreakout, GridTrader, DCABot,
    )
}


def build_strategy(name: str, params: Optional[dict] = None) -> Strategy:
    """Build a strategy inside the chassis (context -> regime -> fee
    gate -> signal -> sizing -> order). Base params flow to the base
    class; chassis/gate params live under the ``fee_aware`` key."""
    if name not in REGISTRY:
        raise KeyError(f"unknown strategy '{name}'; available: {sorted(REGISTRY)}")
    params = dict(params or {})
    gate_params = params.pop("fee_aware", {}) or {}
    base = REGISTRY[name](params or None)
    return ChassisStrategy(base, gate_params)


def build_strategies(config: dict) -> list[Strategy]:
    """Build strategies from a config section like {'momentum': {...}, 'ml': {...}}."""
    out = []
    for name, params in (config or {}).items():
        out.append(build_strategy(name, params or None))
    return out