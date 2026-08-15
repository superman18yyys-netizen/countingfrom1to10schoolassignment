"""Deep Time: an accelerated evolutionary world on multi-year history.

The idea: bots "live" in a sped-up world where years of real Coinbase
candles play back in minutes. Inside that world the same rules as the
live swarm apply -- realized-Sharpe selection, fees + slippage + cash
yield on every fill, multi-pair accounts, max_positions -- so tunings
that win here are tunings that would have won live.

Honesty architecture (the part that makes a "skill win" meaningful):

  * EVOLUTION ZONE = the oldest (1 - validation_frac) of history. Selection,
    mutation and reproduction only ever happen here.
  * VALIDATION GAUNTLET = the most recent validation_frac of history. The
    evolution NEVER touches it. Each epoch, the top candidate genomes are
    replayed once through the gauntlet with fresh accounts; their score is
    OOS by construction (they were never selected on this data).
  * Champions must clear the gauntlet AFTER fees, with enough closed
    trades, and beat mean buy & hold by a margin -- only then are they
    promoted to the live bots (state/champions.json).

Engine design (why epochs are fast): strategies are stateless functions
of the candle window, so signals for one (genome, pair) are computed
ONCE over the full history (vectorised) and cached; the expensive
indicator work never repeats per bar. The per-bar loop only does cheap
account operations, exactly matching the live runner's semantics
(signal on a closed candle, fill at that close, fees inside the
account). One epoch over ~3y of 4H candles x 6 pairs x 40 bots runs in
seconds, so a CI window can accumulate hundreds of simulated years.
"""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from bot.paper.account import PaperAccount
from bot.strategies import build_strategy
from bot.swarm.genome import PARAM_BOUNDS, Genome, make_genome, mutate
from bot.swarm.population import Agent, Population

# Strategies the deep world can evolve. ml_* need fitted models and
# llm/order_flow need live network feeds, so they are excluded: the deep
# world evolves parameter tunings of signal-based models only.
DEEP_STRATEGIES = sorted(
    s for s in PARAM_BOUNDS if s not in ("ml_trend", "ml"))

DEFAULT_STATE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "state", "deep")

MIN_VALIDATION_TRADES = 6     # a champion must have actually traded OOS
MIN_CHAMPION_EXCESS = 0.5     # % mean excess vs buy&hold to be promotable
BEAT_MARGIN = 0.25            # % improvement to count as a new best epoch
PATIENCE = 6                  # epochs without improvement -> converged


def _key(genome: Genome) -> Tuple[str, str]:
    return (genome.strategy, json.dumps(genome.params, sort_keys=True))


@dataclass
class DeepReport:
    epoch: int
    elapsed_sec: float
    validation_span: Tuple[str, str]
    candidates: List[dict] = field(default_factory=list)
    converged: bool = False
    total_trades: int = 0          # closed trades across the epoch's evolution
    open_positions: int = 0        # positions still riding at epoch end


class DeepWorld:
    """The accelerated world. Construct with per-pair history, then run
    epochs until the time budget is spent or convergence is reached."""

    def __init__(self, data: Dict[str, pd.DataFrame], pairs: List[str],
                 granularity: str, capital: float = 20.0, fee_cfg: Optional[dict] = None,
                 n_agents: int = 40, top_k: int = 10, clones: int = 3,
                 min_trades: int = 3, segment_bars: int = 2190,
                 validation_frac: float = 0.25, rng_seed: int = 7):
        if not data or any(df.empty for df in data.values()):
            raise ValueError("deep world needs non-empty history per pair")
        self.pairs = [p for p in pairs if p in data]
        self.granularity = granularity
        self.capital = capital
        self.fee_cfg = fee_cfg or {"taker_fee": 0.006, "slippage": 0.001,
                                   "position_fraction": 0.50, "max_positions": 3}
        self.n_agents = n_agents
        self.top_k = top_k
        self.clones = clones
        self.min_trades = min_trades
        self.segment_bars = segment_bars
        self.validation_frac = validation_frac
        self.rng_seed = rng_seed
        self._dfs: Dict[str, pd.DataFrame] = dict(data)
        self._sel_count = 0

        # ---- per-pair arrays + global aligned timeline -------------------
        self._ts: Dict[str, np.ndarray] = {}
        self._pos: Dict[str, Dict[int, int]] = {}
        self._close: Dict[str, np.ndarray] = {}
        self._pairs_at: Dict[int, List[str]] = {}
        timeline: set[int] = set()
        for pair, df in data.items():
            # normalise to epoch seconds robustly across resolutions:
            # Store.load_candles returns tz-aware datetime64[s] (where
            # int64 is ALREADY seconds); the fetcher returns [ns].
            ts_arr = np.array(
                [int(t.timestamp()) for t in df.index], dtype=np.int64)
            self._ts[pair] = ts_arr
            self._close[pair] = df["close"].to_numpy(dtype=float)
            self._pos[pair] = {int(t): i for i, t in enumerate(ts_arr)}
            timeline.update(int(t) for t in ts_arr)
        self.timeline = np.array(sorted(timeline), dtype=np.int64)
        for t in self.timeline:
            self._pairs_at[int(t)] = [p for p in self.pairs if int(t) in self._pos[p]]

        n = len(self.timeline)
        n_valid = max(200, int(n * validation_frac))
        if n - n_valid < segment_bars:
            raise ValueError(
                f"history too short: {n} bars total, need > segment_bars "
                f"({segment_bars}) + validation ({n_valid})")
        self.train_idx = (0, n - n_valid)
        self.valid_idx = (n - n_valid, n)
        self.last_ts = int(self.timeline[-1])

        # ---- state --------------------------------------------------------
        self._sig: Dict[Tuple[str, str, str], np.ndarray] = {}
        self._warmup: Dict[Tuple[str, str], int] = {}
        self.epoch = 0
        self.no_improve = 0
        self.converged = False
        self.best: Optional[dict] = None
        self.champion_history: List[dict] = []
        self.pop = self._fresh_population(rng_seed)

    # ------------------------------------------------------------- seeding
    def _fresh_population(self, seed: int) -> Population:
        rng = random.Random(seed)
        pop = Population(pairs=self.pairs, granularity=self.granularity,
                         capital=self.capital, fee_cfg=self.fee_cfg,
                         strategy="deep")
        agents = []
        for i in range(self.n_agents):
            strat = rng.choice(DEEP_STRATEGIES)
            g = make_genome(strat, f"d{self.epoch}-{i:02d}", rng)
            acc = PaperAccount(capital=self.capital, **self.fee_cfg)
            agents.append(Agent(genome=g, account=acc, equity=self.capital))
        pop.agents = agents
        return pop

    def _reseed_from(self, genomes: List[Genome], seed: int) -> None:
        """Start a new epoch from the given genomes (mutated clones fill
        the roster, plus fresh immigrants for diversity)."""
        rng = random.Random(seed)
        agents: List[Agent] = []
        if genomes:
            per = max(1, self.n_agents // len(genomes))
            i = 0
            for g in genomes:
                for c in range(per):
                    if len(agents) >= self.n_agents:
                        break
                    gid = f"d{self.epoch + 1}-{i:02d}"
                    child = mutate(g, gid, rng, strength=0.05 if c else 0.0) \
                        if c else Genome(id=gid, strategy=g.strategy,
                                         params=dict(g.params), lineage=list(g.lineage))
                    agents.append(self._agent(child))
                    i += 1
        while len(agents) < self.n_agents:
            gid = f"d{self.epoch + 1}-{len(agents):02d}"
            g = make_genome(rng.choice(DEEP_STRATEGIES), gid, rng)
            g.lineage = ["immigrant"]
            agents.append(self._agent(g))
        self.pop.agents = agents

    def _agent(self, genome: Genome) -> Agent:
        acc = PaperAccount(capital=self.capital, **self.fee_cfg)
        return Agent(genome=genome, account=acc, equity=self.capital)

    # ------------------------------------------------------ signal caching
    def _signals(self, genome: Genome, pair: str) -> Optional[np.ndarray]:
        """Signals for one (genome, pair) over its FULL history, cached.

        Broken/unfittable strategies degrade to all-zeros (they never
        trade and get selected out) -- one bad genome can't kill a run.
        """
        k = _key(genome)
        cache_key = (k[0], k[1], pair)
        if cache_key in self._sig:
            return self._sig[cache_key]
        if k not in self._warmup:
            try:
                strat = build_strategy(genome.strategy, genome.params)
                self._warmup[k] = max(1, strat.warmup_bars())
            except Exception:  # noqa: BLE001
                self._warmup[k] = 10**9
        if self._warmup[k] >= 10**9:
            arr = np.zeros(len(self._close[pair]), dtype=np.int8)
            self._sig[cache_key] = arr
            return arr
        try:
            df = self._dfs[pair]
            series = build_strategy(genome.strategy, genome.params) \
                .compute_signals(df)
            arr = series.fillna(0).astype(int).to_numpy(dtype=np.int8)
        except Exception:  # noqa: BLE001
            arr = np.zeros(len(self._close[pair]), dtype=np.int8)
        self._sig[cache_key] = arr
        return arr

    # ----------------------------------------------------------- replaying
    def _replay(self, t0: int, t1: int, evolve: bool) -> None:
        """Walk every agent through timeline[t0:t1) with live-runner
        semantics: signal on the closed candle, fill at that close.

        With ``evolve=True``, selection runs at every segment boundary;
        survivors KEEP their accounts and open positions (never a forced
        sell -- children are the ones born flat).
        """
        agents = self.pop.agents
        keys = [ _key(a.genome) for a in agents ]
        boundaries = set()
        if evolve:
            seg = self.segment_bars
            boundaries = set(range(t0 + seg, t1, seg))
        for step, ts in enumerate(self.timeline[t0:t1]):
            ts = int(ts)
            first_pair = True
            for pair in self._pairs_at[ts]:
                i = self._pos[pair][ts]
                price = float(self._close[pair][i])
                if first_pair:
                    for a in agents:
                        a.account.accrue_yield(ts)
                    first_pair = False
                for a, k in zip(agents, keys):
                    arr = self._signals(a.genome, pair)
                    sig = arr[i] if i >= self._warmup.get(k, 0) else 0
                    if sig == 1:
                        a.account.open_position(pair, price, ts)
                    elif sig == -1:
                        a.account.close_position(pair, price, ts)
            if evolve and (step + 1) in boundaries:
                self._sel_count += 1
                self.pop.select_and_repopulate(
                    top_k=self.top_k, clones=self.clones,
                    min_trades=self.min_trades,
                    rng_seed=self.rng_seed * 1000 + self._sel_count)
                agents = self.pop.agents
                keys = [_key(a.genome) for a in agents]

    # ------------------------------------------------------------- scoring
    def _validation_span(self) -> Tuple[str, str]:
        t0, t1 = self.valid_idx
        return (str(pd.to_datetime(self.timeline[t0], unit="s", utc=True)),
                str(pd.to_datetime(self.timeline[t1 - 1], unit="s", utc=True)))

    def _benchmark_return(self) -> float:
        """Mean buy & hold return across pairs over the validation span."""
        t0, t1 = self.valid_idx
        start_ts, end_ts = int(self.timeline[t0]), int(self.timeline[t1 - 1])
        rets = []
        for pair in self.pairs:
            ts_arr = self._ts[pair]
            inside = np.where((ts_arr >= start_ts) & (ts_arr <= end_ts))[0]
            if len(inside) < 2:
                continue
            a, b = inside[0], inside[-1]
            if self._close[pair][a] > 0:
                rets.append(self._close[pair][b] / self._close[pair][a] - 1.0)
        return float(np.mean(rets)) if rets else 0.0

    def validate(self, candidates: List[Genome]) -> List[dict]:
        """Replay candidates once through the gauntlet with fresh accounts.
        No selection, no adaptation -- a pure out-of-sample exam."""
        t0, t1 = self.valid_idx
        start_ts, end_ts = int(self.timeline[t0]), int(self.timeline[t1 - 1])
        bench = self._benchmark_return()
        out = []
        for g in candidates:
            acc = PaperAccount(capital=self.capital, **self.fee_cfg)
            k = _key(g)
            holder = Agent(genome=g, account=acc, equity=self.capital)
            for ts in self.timeline[t0:t1]:
                ts = int(ts)
                acc.accrue_yield(ts)
                for pair in self._pairs_at[ts]:
                    i = self._pos[pair][ts]
                    price = float(self._close[pair][i])
                    arr = self._signals(g, pair)
                    sig = arr[i] if i >= self._warmup.get(k, 0) else 0
                    if sig == 1:
                        acc.open_position(pair, price, ts)
                    elif sig == -1:
                        acc.close_position(pair, price, ts)
            prices = {p: float(self._close[p][self._pos[p][end_ts]])
                      for p in self.pairs if end_ts in self._pos[p]}
            equity = acc.equity(prices)
            excess = (equity / self.capital - 1.0) - bench
            out.append({
                "strategy": g.strategy, "params": dict(g.params),
                "equity": round(equity, 4),
                "return_pct": round((equity / self.capital - 1.0) * 100.0, 2),
                "benchmark_pct": round(bench * 100.0, 2),
                "excess_pct": round(excess * 100.0, 2),
                "sharpe": round(acc.realized_sharpe(20), 4),
                "trades": acc.n_trades,
                "fees": round(acc.fee_take, 4),
                "eligible": acc.n_trades >= MIN_VALIDATION_TRADES,
            })
        out.sort(key=lambda r: (r["eligible"], r["excess_pct"]), reverse=True)
        return out

    # --------------------------------------------------------------- epoch
    def run_epoch(self) -> DeepReport:
        started = time.time()
        t0, t1 = self.train_idx
        self._replay(t0, t1, evolve=True)

        # candidates: top unique genomes by end-of-epoch TRAIN fitness
        board = self.pop.leaderboard()
        seen, candidates = set(), []
        for a in board:
            k = _key(a.genome)
            if k in seen:
                continue
            seen.add(k)
            candidates.append(a.genome)
            if len(candidates) >= 8:
                break
        results = self.validate(candidates)

        self.epoch += 1
        best = next((r for r in results if r["eligible"]), None)
        improved = False
        if best and (self.best is None or
                     best["excess_pct"] > self.best["excess_pct"] + BEAT_MARGIN):
            self.best = {**best, "epoch": self.epoch}
            self.no_improve = 0
            improved = True
        else:
            self.no_improve += 1
        if self.no_improve >= PATIENCE:
            self.converged = True
        self.champion_history.append({
            "epoch": self.epoch, "improved": improved,
            "best_excess_pct": (best or {}).get("excess_pct"),
            "top": results[0] if results else None,
        })
        self.champion_history = self.champion_history[-50:]

        # next epoch: evolve from the best TRAIN performers (NOT from
        # validation scores -- the gauntlet must stay untouched).
        seed_genomes = [a.genome for a in board[:8]]
        self._reseed_from(seed_genomes, self.rng_seed + self.epoch)

        return DeepReport(epoch=self.epoch, elapsed_sec=time.time() - started,
                          validation_span=self._validation_span(),
                          candidates=results, converged=self.converged,
                          total_trades=sum(a.account.n_trades for a in board),
                          open_positions=sum(a.account.n_positions for a in board))

    # ---------------------------------------------------------- persistence
    def to_dict(self) -> dict:
        return {
            "version": 1,
            "epoch": self.epoch,
            "converged": self.converged,
            "no_improve": self.no_improve,
            "best": self.best,
            "champion_history": self.champion_history,
            "population": self.pop.to_dict(),
            "config": {
                "pairs": self.pairs, "granularity": self.granularity,
                "capital": self.capital, "n_agents": self.n_agents,
                "top_k": self.top_k, "clones": self.clones,
                "min_trades": self.min_trades, "segment_bars": self.segment_bars,
                "validation_frac": self.validation_frac, "rng_seed": self.rng_seed,
            },
            "timeline_ends": [int(self.timeline[0]), int(self.timeline[-1])],
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=1)
        os.replace(tmp, path)

    @classmethod
    def load(cls, data: Dict[str, pd.DataFrame], path: str,
             fee_cfg: Optional[dict] = None) -> "DeepWorld":
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        cfg = d["config"]
        world = cls(data, pairs=cfg["pairs"], granularity=cfg["granularity"],
                    capital=cfg["capital"], fee_cfg=fee_cfg, n_agents=cfg["n_agents"],
                    top_k=cfg["top_k"], clones=cfg["clones"],
                    min_trades=cfg["min_trades"], segment_bars=cfg["segment_bars"],
                    validation_frac=cfg["validation_frac"], rng_seed=cfg["rng_seed"])
        world.epoch = d["epoch"]
        world.converged = d["converged"]
        world.no_improve = d["no_improve"]
        world.best = d["best"]
        world.champion_history = d["champion_history"]
        pop = world.pop
        pop.agents = [Agent.from_dict(a, world.fee_cfg) for a in d["population"]["agents"]]
        pop.generation = d["population"].get("generation", 0)
        pop.history = d["population"].get("history", [])
        return world

    def bind_data(self, data: Dict[str, pd.DataFrame]) -> None:
        """Re-bind history dataframes after a checkpoint load (candles are
        not serialized in the state file)."""
        self._dfs = dict(data)
        self._sig = {}   # signal cache is cheap to rebuild
