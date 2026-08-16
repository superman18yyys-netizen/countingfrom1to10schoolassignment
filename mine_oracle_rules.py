"""Mine mechanical rules from the optimal path.

The optimal path uses future knowledge. But at every BUY bar, only
the past is observable. This script computes, at each oracle buy bar,
the features A-000 CAN see (momentum, RSI, ATR percentile, drawdown,
cross-sectional rank, surge) and compares their distribution against
random bars — extracting the mechanical rules that discriminate
oracle entries from noise. These rules become A-000's entry filter.

Output: reports/oracle-rules.json + printed rule summary.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bot.data.store import Store  # noqa: E402
from bot.indicators.ta import atr, rsi  # noqa: E402

RULES = {
    "mom_40": lambda df, i: float(df["close"].iloc[i]
                                  / df["close"].iloc[max(0, i - 40)] - 1.0),
    "mom_90d_pctile": lambda df, i: _mom_pctile(df, i),
    "rsi14": lambda df, i: float(_rsi(df, i)),
    "atr_pct": lambda df, i: float(_atr(df, i)),
    "atr_pctile_90d": lambda df, i: float(_atr_pctile(df, i)),
    "dd_from_high": lambda df, i: float(_dd(df, i)),
    "surge_12": lambda df, i: float(df["close"].iloc[i]
                                    / df["close"].iloc[max(0, i - 12)] - 1.0),
}

_CACHE: dict = {}


def _frame(df: pd.DataFrame):
    key = id(df)
    if key not in _CACHE:
        close = df["close"]
        _CACHE[key] = {
            "mom_pct": (close / close.shift(40) - 1.0).rolling(
                2160, min_periods=300).apply(
                lambda v: (v <= v[-1]).mean(), raw=True),
            "rsi": rsi(close, 14),
            "atr": atr(df["high"], df["low"], close, 14) / close,
            "atr_pct": (atr(df["high"], df["low"], close, 14)
                        / close).rolling(540, min_periods=100).apply(
                lambda v: (v <= v[-1]).mean(), raw=True),
            "dd": close / close.rolling(2160, min_periods=100).max() - 1.0,
        }
    return _CACHE[key]


def _mom_pctile(df, i):
    v = _frame(df)["mom_pct"].iloc[i]
    return float(v) if v == v else 0.5


def _rsi(df, i):
    v = _frame(df)["rsi"].iloc[i]
    return float(v) if v == v else 50.0


def _atr(df, i):
    v = _frame(df)["atr"].iloc[i]
    return float(v) if v == v else 0.0


def _atr_pctile(df, i):
    v = _frame(df)["atr_pct"].iloc[i]
    return float(v) if v == v else 0.5


def _dd(df, i):
    v = _frame(df)["dd"].iloc[i]
    return float(v) if v == v else 0.0


def main() -> None:
    with open("reports/optimal-path.json") as fh:
        path = json.load(fh)

    store = Store("data/universe.db")
    frames = {}
    pair_list = [r[0] for r in store.conn.execute(
        "SELECT DISTINCT pair FROM candles WHERE granularity='FOUR_HOUR'")]
    total = len(pair_list)
    done = 0
    for pair in pair_list:
        df = store.load_candles(pair, "FOUR_HOUR")
        if df is not None and len(df) >= 400:
            frames[pair] = df.dropna()
        done += 1
        if done % 50 == 0 or done == total:
            print(f"[rule-mine] frames: {done}/{total} coins loaded "
                  f"({len(frames)} usable)", flush=True)

    # oracle feature vectors + random baseline
    oracle_f = {k: [] for k in RULES}
    rand_f = {k: [] for k in RULES}
    rng = np.random.default_rng(7)
    print(f"[rule-mine] computing features for "
          f"{len(path['ledger'])} oracle buys...", flush=True)
    for n, t in enumerate(path["ledger"]):
        df = frames.get(t["pair"])
        if df is None:
            continue
        i = df.index.get_indexer([pd.Timestamp(t["buy_ts"], unit="s",
                                               tz="UTC")],
                                 method="nearest")[0]
        for k, fn in RULES.items():
            oracle_f[k].append(fn(df, i))
        # random baseline from the same coin
        j = int(rng.integers(100, len(df) - 100))
        for k, fn in RULES.items():
            rand_f[k].append(fn(df, j))
        if (n + 1) % 500 == 0:
            print(f"[rule-mine] {n + 1}/{len(path['ledger'])} buys "
                  f"processed", flush=True)

    print(f"== RULE MINING: {len(oracle_f['mom_40'])} oracle buys vs "
          f"random bars ==\n")
    print(f"{'feature':<16}{'oracle med':>12}{'random med':>12}"
          f"{'oracle q25':>12}{'oracle q75':>12}{'rule':>28}")
    rules_out = {}
    for k in RULES:
        o = np.array(oracle_f[k])
        r = np.array(rand_f[k])
        med_o, med_r = np.median(o), np.median(r)
        q25, q75 = np.percentile(o, 25), np.percentile(o, 75)
        if med_o > med_r:
            rule = f"{k} above ~{med_r:.2f} (med {med_o:.2f})"
        else:
            rule = f"{k} below ~{med_r:.2f} (med {med_o:.2f})"
        print(f"{k:<16}{med_o:>12.3f}{med_r:>12.3f}{q25:>12.3f}"
              f"{q75:>12.3f}{rule:>28}")
        rules_out[k] = {"oracle_median": round(float(med_o), 4),
                        "random_median": round(float(med_r), 4),
                        "q25": round(float(q25), 4),
                        "q75": round(float(q75), 4)}

    out = {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
           "n_oracle_trades": len(oracle_f["mom_40"]),
           "rules": rules_out}
    os.makedirs("reports", exist_ok=True)
    with open("reports/oracle-rules.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print("\n[rule-mine] -> reports/oracle-rules.json")


if __name__ == "__main__":
    main()
