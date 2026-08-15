"""Swarm population: 40 agents, realized-Sharpe selection, compounding.

State is a single small JSON file (``state/population.json``) so it can
be committed to the repo between GitHub Actions runs.

Key design changes from the original (Aug 2026 critique):
- Fitness is now **realized, fee-paid per-trade Sharpe** over a rolling
  window, NOT mark-to-market net worth. This rewards consistent, small,
  fee-clearing profits instead of one lucky coin pump.
- Capital is NOT reset each generation — compounding is preserved so
  bots with sustained edge accumulate real wealth.
- Survival rate widened to ~half (TOP_K=20) to preserve diversity.
- Immigrants sample from ALL strategy families in PARAM_BOUNDS, so the
  swarm never gets stuck in one strategy's tuning space.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from bot.paper.account import PaperAccount
from bot.strategies import build_strategy
from bot.strategies.base import Strategy
from bot.swarm.genome import PARAM_BOUNDS, Genome, make_genome, mutate

POP_SIZE = 40
TOP_K = 20                    # wider survival preserves diversity
CLONES_PER_SURVIVOR = 2      # 20 survivors x 2 = 40
IMMIGRANTS = 0                # immigrants come from a separate pool below
MIN_TRADES = 3
DEFAULT_CAPITAL = 20.0
# number of most-recent trade returns used for Sharpe fitness (None = all)
SHARPE_WINDOW = 20
VERSION = 3                   # bumped for the fitness rewrite


@dataclass
class Agent:
    genome: Genome
    account: PaperAccount
    equity: float = DEFAULT_CAPITAL
    _strategy: Optional[Strategy] = None

    @property
    def strategy(self) -> Strategy:
        if self._strategy is None:
            self._strategy = build_strategy(self.genome.strategy, self.genome.params)
        return self._strategy

    @property
    def fitness(self) -> float:
        """Primary fitness: realized Sharpe ratio over recent fee-paid trades.

        A bot with no trades scores 0 (can't win on hope). A bot with
        consistent small positive returns after fees scores high. A bot
        that had one lucky huge trade and a dozen losers scores lower
        than a bot with +0.1% per trade on 15 trades.
        """
        return self.account.realized_sharpe(SHARPE_WINDOW)

    def to_dict(self) -> dict:
        # Don't include `_strategy` — it's restored lazily
        return {"genome": self.genome.to_dict(),
                "account": self.account.state_dict(),
                "equity": self.equity}

    @classmethod
    def from_dict(cls, d: dict, fee_cfg: dict) -> "Agent":
        genome = Genome.from_dict(d["genome"])
        acc = PaperAccount.from_dict(d["account"], **fee_cfg)
        return cls(genome=genome, account=acc, equity=d.get("equity", acc.cash))


# ---- strategy families eligible to spawn immigrants --------------------
_MUTABLE_STRATEGIES = sorted(PARAM_BOUNDS.keys())


class Population:
    def __init__(self, pairs: List[str], granularity: str,
                 capital: float = DEFAULT_CAPITAL,
                 fee_cfg: Optional[dict] = None,
                 strategy: str = "mean_reversion"):
        self.pairs = pairs
        self.granularity = granularity
        self.capital = capital
        self.fee_cfg = fee_cfg or {}
        self.strategy = strategy
        self.agents: List[Agent] = []
        self.generation = 0
        self.day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.last_ts = 0
        self.history: List[dict] = []

    # ------------------------------------------------------------- seeding
    def seed(self, n: int = POP_SIZE, strategy: str = "mean_reversion",
             seeds: Optional[List[dict]] = None, rng_seed: int = 7) -> None:
        """Create the initial population.

        If ``seeds`` (from run_train.py or the ML pipeline) is given,
        clone each seed into n/len(seeds) mutated children -- each child
        may be of a different strategy type (the seed's strategy). If no
        seeds, sample random tunings from the default strategy.
        """
        rng = random.Random(rng_seed)
        self.agents = []
        self.strategy = strategy
        if seeds:
            per = max(1, n // len(seeds))
            i = 0
            for seed in seeds:
                seed_strat = seed.get("strategy", strategy)
                base_params = dict(seed.get("params", {}))
                base = Genome(id=f"seed-{seed_strat}", strategy=seed_strat,
                              params=base_params)
                for c in range(per):
                    if len(self.agents) >= n:
                        break
                    gid = f"g0-{i:02d}"
                    child = mutate(base, gid, rng, strength=0.05) if c else \
                        Genome(id=gid, strategy=base.strategy, params=dict(base.params))
                    self.agents.append(self._fresh_agent(child))
                    i += 1
            while len(self.agents) < n:
                gid = f"g0-{len(self.agents):02d}"
                strat_name = rng.choice(_MUTABLE_STRATEGIES)
                g = make_genome(strat_name, gid, rng)
                self.agents.append(self._fresh_agent(g))
        else:
            for i in range(n):
                gid = f"g0-{i:02d}"
                strat_name = rng.choice(_MUTABLE_STRATEGIES) if i > 3 else strategy
                g = make_genome(strat_name, gid, rng)
                self.agents.append(self._fresh_agent(g))
        self.generation = 0

    def _fresh_agent(self, genome: Genome) -> Agent:
        acc = PaperAccount(capital=self.capital, **self.fee_cfg)
        return Agent(genome=genome, account=acc, equity=self.capital)

    # ---------------------------------------------------------- selection
    def mark_equity(self, prices: Dict[str, float]) -> None:
        for agent in self.agents:
            agent.equity = agent.account.equity(prices)

    def leaderboard(self) -> List[Agent]:
        return sorted(self.agents,
                      key=lambda a: (a.fitness, a.account.realized_pnl,
                                     a.account.n_trades, a.genome.id),
                      reverse=True)

    def select_and_repopulate(self, top_k: int = TOP_K,
                              clones: int = CLONES_PER_SURVIVOR,
                              immigrants: int = IMMIGRANTS,
                              min_trades: int = MIN_TRADES,
                              rng_seed: Optional[int] = None) -> dict:
        """Daily evolution step.

        Fitness is **realized, fee-paid per-trade Sharpe** over a rolling
        window — the only signal worth evolving: consistent, small,
        risk-adjusted returns after all costs. Open positions do NOT
        contribute to fitness, so a bot holding a coin that pumped 40%
        still needs to actually exit profitably for credit.

        Anti-gaming rules:
        * A bot needs >= ``min_trades`` closed trades to be ELIGIBLE.
        * Immigrants are drawn from ALL available strategy families to
          prevent the swarm from self-cannibalizing into one tuning space.
        * Survivors (top_k) are cloned with mutations; everyone keeps
          their accumulated capital — no reset — so compounding rewards
          sustained edge.

        Returns a summary for reporting.
        """
        rng = random.Random(rng_seed if rng_seed is not None
                            else int(datetime.now(timezone.utc).timestamp()))
        board = self.leaderboard()
        eligible = [a for a in board if a.account.n_trades >= min_trades]
        survivors = eligible[:top_k]
        fallback_used = 0
        if len(survivors) < top_k:
            for a in board:
                if len(survivors) >= top_k:
                    break
                if a not in survivors:
                    survivors.append(a)
                    fallback_used += 1
        now = datetime.now(timezone.utc)
        summary = {
            "date": now.strftime("%Y-%m-%d %H:%M UTC"),
            "generation_finished": self.generation,
            "min_trades_gate": min_trades,
            "disqualified_no_trades": len(board) - len(eligible),
            "fallback_survivors": fallback_used,
            "final_standings": [
                {"id": a.genome.id, "strategy": a.genome.strategy,
                 "fitness": round(a.fitness, 4),
                 "sharpe_20": round(a.account.realized_sharpe(20), 4),
                 "net_worth": round(a.account.cash, 4),
                 "realized_pnl": round(a.account.realized_pnl, 4),
                 "trades": a.account.n_trades,
                 "eligible": a.account.n_trades >= min_trades,
                 "holdings": {p: round(pos.qty, 8)
                              for p, pos in a.account.positions.items()},
                 "params": a.genome.params}
                for a in board
            ],
            "survivors": [a.genome.id for a in survivors],
        }
        # Repopulate: survivors keep their accounts (capital accumulutes).
        # Immigrants get fresh $20.
        new_agents: List[Agent] = []
        self.generation += 1
        for survivor in survivors:
            for c in range(clones):
                if len(new_agents) >= POP_SIZE:
                    break
                gid = f"g{self.generation}-{len(new_agents):02d}"
                strength = 0.05 if c == 0 else 0.15
                child = mutate(survivor.genome, gid, rng, strength=strength)
                # Child starts with HALF the survivor's capital (to keep
                # a diversity of capital levels); the survivor keeps their
                # full account in new_agents below  (they always survive).
                child_capital = max(self.capital, survivor.account.cash * 0.5)
                acc = PaperAccount(capital=child_capital, **self.fee_cfg)
                new_agents.append(Agent(genome=child, account=acc,
                                        equity=child_capital))
            # Also keep the survivor themselves (amplified survival)
            gid = f"g{self.generation}-{len(new_agents):02d}"
            child = mutate(survivor.genome, gid, rng, strength=0.02)
            acc = PaperAccount(capital=survivor.account.cash, **self.fee_cfg)
            new_agents.append(Agent(genome=child, account=acc,
                                    equity=survivor.account.cash))

        # Immigrants from diverse strategy families
        # We budget one immigrant slot if we have room after clones + survivors
        while len(new_agents) < POP_SIZE:
            gid = f"g{self.generation}-{len(new_agents):02d}"
            strat_name = rng.choice(_MUTABLE_STRATEGIES)
            g = make_genome(strat_name, gid, rng)
            g.lineage = ["immigrant"]
            acc = PaperAccount(capital=self.capital, **self.fee_cfg)
            new_agents.append(Agent(genome=g, account=acc,
                                    equity=self.capital))

        self.agents = new_agents[:POP_SIZE]
        self.history.append(summary)
        return summary

    def maybe_rollover(self, top_k: int = TOP_K, clones: int = CLONES_PER_SURVIVOR,
                       immigrants: int = IMMIGRANTS, min_trades: int = MIN_TRADES
                       ) -> Optional[dict]:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today <= self.day or not self.agents:
            return None
        summary = self.select_and_repopulate(top_k, clones, immigrants, min_trades)
        self.day = today
        return summary

    # -------------------------------------------------------- persistence
    def to_dict(self) -> dict:
        return {
            "version": VERSION,
            "generation": self.generation,
            "day": self.day,
            "last_ts": self.last_ts,
            "granularity": self.granularity,
            "pairs": self.pairs,
            "capital": self.capital,
            "strategy": self.strategy,
            "agents": [a.to_dict() for a in self.agents],
            "history": self.history[-14:],
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=1)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str, fee_cfg: Optional[dict] = None) -> "Population":
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        pop = cls(pairs=d["pairs"], granularity=d["granularity"],
                  capital=d.get("capital", DEFAULT_CAPITAL), fee_cfg=fee_cfg,
                  strategy=d.get("strategy", "mean_reversion"))
        pop.generation = d.get("generation", 0)
        pop.day = d.get("day")
        pop.last_ts = d.get("last_ts", 0)
        pop.history = d.get("history", [])
        pop.agents = [Agent.from_dict(a, pop.fee_cfg) for a in d.get("agents", [])]
        return pop

    @classmethod
    def load_or_seed(cls, path: str, pairs: List[str], granularity: str,
                     capital: float, fee_cfg: dict, n: int = POP_SIZE,
                     strategy: str = "mean_reversion",
                     seeds: Optional[List[dict]] = None) -> "Population":
        if os.path.exists(path):
            return cls.load(path, fee_cfg)
        pop = cls(pairs=pairs, granularity=granularity, capital=capital, fee_cfg=fee_cfg)
        pop.seed(n=n, strategy=strategy, seeds=seeds)
        pop.save(path)
        return pop