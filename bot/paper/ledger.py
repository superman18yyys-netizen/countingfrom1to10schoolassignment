"""Persistence helpers for paper trading (wraps bot.data.store)."""
from __future__ import annotations

from typing import List

from bot.data.store import Store
from bot.paper.account import PaperAccount


def save_account(store: Store, strategy: str, account: PaperAccount) -> None:
    store.set_state(f"paper_account:{strategy}", account.state_dict())


def load_account(store: Store, strategy: str) -> PaperAccount | None:
    data = store.get_state(f"paper_account:{strategy}")
    return PaperAccount.from_dict(data) if data else None


def record_closed_trade(store: Store, strategy: str, trade: dict) -> None:
    store.record_trade(
        strategy=strategy, pair=trade["pair"],
        entry_ts=trade["entry_ts"], entry_price=trade["entry_price"],
        exit_ts=trade["exit_ts"], exit_price=trade["exit_price"],
        qty=trade["qty"], entry_fee=trade["entry_fee"], exit_fee=trade["exit_fee"],
        pnl=trade["pnl"], pnl_pct=trade["pnl_pct"],
    )


def record_fills(store: Store, strategy: str, fills: List[tuple[str, str, int, float, float, float]]) -> None:
    for pair, side, ts, price, qty, fee in fills:
        store.record_fill(strategy, pair, side, ts, price, qty, fee)