"""Walk-forward validation with purge gaps (anti-overfitting discipline).

The standard protocol from the research (Bailey & Lopez de Prado 2021;
AlgoXpert IS->WFA->OOS 2026): parameters must never peek past the
training window, and a purge gap removes the autocorrelation seam
between train and test folds.
"""
from __future__ import annotations

from typing import Callable, Optional

import pandas as pd

from bot.backtest.engine import BacktestResult, run_backtest
from bot.strategies.base import Strategy


def split_train_test(df: pd.DataFrame, train_frac: float = 0.7,
                     purge_bars: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split history into train/test by time with a purge gap between them."""
    n = len(df)
    split = int(n * train_frac)
    train = df.iloc[:split]
    test = df.iloc[split + purge_bars:] if split + purge_bars < n else df.iloc[split:]
    return train, test


def walkforward(df: pd.DataFrame, strategy_factory: Callable[[], Strategy],
                n_folds: int = 4, fold_bars: Optional[int] = None,
                purge_bars: int = 12, **engine_kwargs) -> list[BacktestResult]:
    """Run the strategy on each of ``n_folds`` rolling test windows.

    Each fold fits a fresh strategy instance (so ML models retrain) on
    the trailing window and evaluates on the following fold window,
    separated by a purge gap. Results are returned per fold.
    """
    n = len(df)
    if fold_bars is None:
        fold_bars = max(100, n // (n_folds + 2))
    results: list[BacktestResult] = []
    pos = n - 2 * fold_bars
    end = n - fold_bars
    while pos + fold_bars <= end:
        train = df.iloc[max(0, pos - 10 * fold_bars):pos]
        test = df.iloc[pos + purge_bars: pos + purge_bars + fold_bars]
        if len(test) < 50:
            break
        strat = strategy_factory()
        strat.fit(train)
        results.append(run_backtest(test, strat, **engine_kwargs))
        pos += fold_bars
    return results