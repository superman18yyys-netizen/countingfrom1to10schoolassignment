"""P&L reporting: backtest comparison tables, equity charts, paper reports."""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from bot.backtest.engine import BacktestResult  # noqa: E402
from bot.data.store import Store  # noqa: E402

plt.rcParams["figure.dpi"] = 110


# ------------------------------------------------------------ backtesting
def backtest_table(results: List[BacktestResult]) -> str:
    if not results:
        return "(no results)"
    rows = [r.summary_row() for r in results]
    df = pd.DataFrame(rows)
    lines = [df.to_string(index=False)]
    best = max(results, key=lambda r: r.excess_return)
    lines.append("")
    lines.append(f"Best vs buy&hold: {best.strategy} on {best.pair} "
                 f"(excess {best.excess_return * 100:+.2f}%)")
    return "\n".join(lines)


def save_backtest_results(results: List[BacktestResult], outdir: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "backtest_results.json")
    payload = [dict(r.summary_row(), start=str(r.start), end=str(r.end)) for r in results]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path


def plot_backtest_equity(results: List[BacktestResult], outdir: str) -> Optional[str]:
    if not results:
        return None
    os.makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    by_pair: Dict[str, List[BacktestResult]] = {}
    for r in results:
        by_pair.setdefault(r.pair, []).append(r)
    for pair, rs in by_pair.items():
        ax2 = ax
        for r in rs:
            ax2.plot(r.equity_curve.index, r.equity_curve.values,
                     label=f"{r.strategy} ({pair})", linewidth=1.1)
    # buy & hold overlay per pair from the first result's data
    ax.set_title("Backtest equity curves (vs $10k start)")
    ax.set_ylabel("Equity ($)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    path = os.path.join(outdir, "backtest_equity.png")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


# ------------------------------------------------------------ paper trading
def paper_report(store: Store, strategies: List[str], pairs: List[str]) -> str:
    lines: List[str] = []
    for strategy in strategies:
        trades = store.load_trades(strategy=strategy)
        if trades.empty:
            lines.append(f"## {strategy}\nNo closed trades yet.\n")
            continue
        win = (trades["pnl"] > 0).mean()
        total_pnl = float(trades["pnl"].sum())
        lines.append(f"## {strategy}")
        lines.append(f"- closed trades: {len(trades)}")
        lines.append(f"- win rate: {win * 100:.1f}%")
        lines.append(f"- total realized P&L: ${total_pnl:,.2f}")
        lines.append("")
        lines.append(trades[["pair", "entry_ts", "exit_ts", "entry_price",
                             "exit_price", "qty", "pnl", "pnl_pct"]]
                     .to_string(index=False))
        lines.append("")
    eq = store.load_equity("buy_hold", "ALL")
    if not eq.empty:
        lines.append(f"## buy_hold baseline (last): ${float(eq.iloc[-1]):,.2f}")
    return "\n".join(lines)


def save_paper_report(text: str, outdir: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "paper_report.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def plot_paper_equity(store: Store, strategies: List[str], outdir: str) -> Optional[str]:
    os.makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    plotted = False
    for strategy in strategies:
        ser = store.load_equity(strategy, "ALL")
        if not ser.empty:
            ax.plot(ser.index, ser.values, label=strategy, linewidth=1.2)
            plotted = True
    base = store.load_equity("buy_hold", "ALL")
    if not base.empty:
        ax.plot(base.index, base.values, label="buy_hold", linewidth=1.2, linestyle="--")
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.set_title("Paper trading equity (pretend money)")
    ax.set_ylabel("Equity ($)")
    ax.legend()
    ax.grid(alpha=0.3)
    path = os.path.join(outdir, "paper_equity.png")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path