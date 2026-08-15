"""Virtual USDC wallet with fee/slippage modeling (pretend money)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from bot.config import RISK_FREE_APY, SECONDS_PER_YEAR


@dataclass
class Position:
    pair: str
    qty: float
    entry_cost: float      # fill price * qty (slippage included)
    entry_fee: float
    entry_ts: int


@dataclass
class PaperAccount:
    capital: float = 10_000.0
    taker_fee: float = 0.006
    slippage: float = 0.001
    position_fraction: float = 0.25
    max_positions: int = 3
    allow_averaging: bool = False   # if True, buying a held pair adds to it (DCA)
    cash_yield_apy: float = RISK_FREE_APY  # APY earned on idle USDC cash
    cash: float = field(init=False)
    positions: Dict[str, Position] = field(default_factory=dict)
    fee_take: float = field(default=0.0, init=False)
    realized_pnl: float = field(default=0.0, init=False)
    n_trades: int = field(default=0, init=False)
    trade_pcts: list = field(default_factory=list, init=False)  # per-trade return %

    def __post_init__(self) -> None:
        self.cash = self.capital
        self._last_yield_ts = 0  # no prior accrual yet; see accrue_yield

    def accrue_yield(self, ts: int) -> None:
        """Compound the risk-free APY on idle cash since the last step.

        Only accrues when we have a *prior* timestamp (``_last_yield_ts > 0``).
        The very first call just records the start timestamp; otherwise a
        fresh account would compound the APY for the entire Unix-epoch
        span (~56 years) and inflate cash by ~3.5x.
        """
        if self.cash_yield_apy <= 0 or ts <= 0:
            return
        last = self._last_yield_ts
        if last > 0 and ts > last:
            elapsed = ts - last
            self.cash *= (1.0 + self.cash_yield_apy * elapsed / SECONDS_PER_YEAR)
        self._last_yield_ts = ts

    @property
    def n_positions(self) -> int:
        return len(self.positions)

    def can_open(self) -> bool:
        return self.n_positions < self.max_positions and self.cash > 1.0

    def equity(self, prices: Dict[str, float]) -> float:
        value = self.cash
        for pair, pos in self.positions.items():
            price = prices.get(pair)
            if price is not None:
                value += pos.qty * price * (1.0 - self.taker_fee)
        return value

    def open_position(self, pair: str, price: float, ts: int,
                      fraction: Optional[float] = None) -> Optional[Position]:
        """Buy with position_fraction of equity (or the caller-supplied
        ``fraction`` — chassis sizing); returns the Position or None."""
        held = self.positions.get(pair)
        if held is not None and not self.allow_averaging:
            return None  # one position per pair; never silently overwrite
        if self.n_positions >= self.max_positions and held is None:
            return None
        if self.cash <= 1.0:
            return None
        fill = price * (1.0 + self.slippage)
        frac = self.position_fraction if fraction is None else float(fraction)
        size = self.equity({pair: price}) * frac
        qty = size / fill
        cost = qty * fill
        fee = cost * self.taker_fee
        if cost + fee > self.cash:
            return None
        self.cash -= cost + fee
        self.fee_take += fee
        if held is not None:  # average into the existing position (DCA)
            held.qty += qty
            held.entry_cost += cost
            held.entry_fee += fee
            held.entry_ts = ts
            return held
        pos = Position(pair=pair, qty=qty, entry_cost=cost, entry_fee=fee, entry_ts=ts)
        self.positions[pair] = pos
        return pos

    def close_position(self, pair: str, price: float, ts: int) -> Optional[dict]:
        """Sell the full position; returns trade record dict or None."""
        pos = self.positions.get(pair)
        if pos is None:
            return None
        fill = price * (1.0 - self.slippage)
        fee = fill * pos.qty * self.taker_fee
        proceeds = fill * pos.qty - fee
        self.cash += proceeds
        self.fee_take += fee
        pnl = proceeds - (pos.entry_cost + pos.entry_fee)
        pnl_pct = pnl / (pos.entry_cost + pos.entry_fee) if pos.entry_cost else 0.0
        self.realized_pnl += pnl
        self.n_trades += 1
        self.trade_pcts.append(pnl_pct)
        del self.positions[pair]
        return {
            "pair": pair, "entry_ts": pos.entry_ts, "entry_price": pos.entry_cost / pos.qty,
            "exit_ts": ts, "exit_price": fill, "qty": pos.qty,
            "entry_fee": pos.entry_fee, "exit_fee": fee, "pnl": pnl, "pnl_pct": pnl_pct,
        }

    def state_dict(self) -> dict:
        return {
            "capital": self.capital, "cash": self.cash,
            "taker_fee": self.taker_fee, "slippage": self.slippage,
            "position_fraction": self.position_fraction,
            "max_positions": self.max_positions,
            "cash_yield_apy": self.cash_yield_apy,
            "positions": {p: {**pos.__dict__} for p, pos in self.positions.items()},
            "fee_take": self.fee_take, "realized_pnl": self.realized_pnl,
            "n_trades": self.n_trades, "trade_pcts": list(self.trade_pcts),
            "_last_yield_ts": getattr(self, "_last_yield_ts", 0),
        }

    @classmethod
    def from_dict(cls, data: dict, **overrides) -> "PaperAccount":
        acc = cls(
            capital=data["capital"],
            taker_fee=overrides.get("taker_fee", data.get("taker_fee", 0.006)),
            slippage=overrides.get("slippage", data.get("slippage", 0.001)),
            position_fraction=data.get("position_fraction", 0.25),
            max_positions=data.get("max_positions", 3),
            cash_yield_apy=overrides.get("cash_yield_apy",
                                         data.get("cash_yield_apy", RISK_FREE_APY)),
        )
        acc.cash = data["cash"]
        acc.fee_take = data.get("fee_take", 0.0)
        acc.realized_pnl = data.get("realized_pnl", 0.0)
        acc.n_trades = data.get("n_trades", 0)
        acc.trade_pcts = list(data.get("trade_pcts", []))
        acc._last_yield_ts = data.get("_last_yield_ts", 0)
        for pair, raw in (data.get("positions") or {}).items():
            acc.positions[pair] = Position(**raw)
        return acc

    def realized_sharpe(self, window: Optional[int] = None) -> float:
        """Risk-adjusted return quality over the last ``window`` trades.

        Scale-free (depends on per-trade returns, not absolute capital),
        so bots with different principal are directly comparable. This is
        the swarm's fitness: it reflects consistent, fee-paid, realized
        returns instead of a single lucky mark-to-market event.
        """
        pcts = self.trade_pcts[-window:] if window else self.trade_pcts
        if len(pcts) < 2:
            return 0.0
        mean = sum(pcts) / len(pcts)
        var = sum((p - mean) ** 2 for p in pcts) / (len(pcts) - 1)
        if var <= 0:
            return 0.0
        return round(mean / (var ** 0.5), 4)