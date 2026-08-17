"""Daily fusion bot vs gen2b champion — same $100, same discipline.

Walk-forward compounding backtest of the daily mechanical-fusion model:
  - 9y 1H history -> 4H resample, top-13 coins
  - fusion features (raw + mechanical rule outputs), realistic segment
    labels (latency + fees in the labels)
  - fold k: train on folds < k, trade fold k (never peeks)
  - ONE account, one position, 95% stakes, hard stop -2%, trail 2%,
    30-bar cap, fees + slippage both sides (1.4% round trip)
  - equity compounds across folds: fold k starts at fold k-1's equity

Reference: gen2b champion folds +38.5 / +89.5 / +308.9, total +973%
($100 -> $1,073.17, 9.9y, walk-forward, fees on). This script reports
the daily bot's numbers on the same footing for a direct comparison.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from nn_lab.final_train import (  # noqa: E402
    SLIP, FEE, TOP_N, DB_1H,
    realistic_segments, build_features_pos, build_features_ladder,
    mechanical_layer)

from bot.data.store import Store  # noqa: E402

OUT = "reports/daily_vs_champion"
THR = 0.30
STAKE = 0.95
HARD_STOP = 0.02
TRAIL = 0.02
CAP_BARS = 30
GEN2B = {"folds": [38.5, 89.5, 308.9], "total_pct": 973.0,
         "end": 1073.17, "years": 9.9, "name": "gen2b champion (week-scale)"}


def log(m):
    print(m, flush=True)


def main():
    log("loading full 1H history...")
    store = Store(DB_1H)
    cand = {}
    for pair in [r[0] for r in store.conn.execute(
            "SELECT DISTINCT pair FROM candles WHERE granularity='ONE_HOUR'")]:
        df = store.load_candles(pair, "ONE_HOUR")
        if df is not None and len(df) >= 3000:
            cand[pair] = df.dropna()
    med = {p: float((d["volume"] * d["close"]).tail(1000).median())
           for p, d in cand.items()
           if (d["volume"] * d["close"]).tail(1000).notna().any()}
    top = sorted(med, key=lambda p: -med[p])[:TOP_N]
    r4 = {}
    for p in top:
        d = cand[p]
        r4[p] = d.resample("4h").agg({"open": "first", "high": "max",
                                      "low": "min", "close": "last",
                                      "volume": "sum"}).dropna()
    closes = {p: r4[p]["close"] for p in top}
    highs = {p: r4[p]["high"] for p in top}
    lows = {p: r4[p]["low"] for p in top}
    vols = {p: r4[p]["volume"] for p in top}
    span_y = (max(c.index[-1] for c in closes.values())
              - min(c.index[0] for c in closes.values())).total_seconds() \
        / (365.25 * 86400)
    log(f"4H span {span_y:.1f}y, top-{TOP_N}")
    segs = realistic_segments(closes)
    log(f"segments: {sum(len(v) for v in segs.values())}")

    mech = mechanical_layer(closes, highs, lows, vols, closes.get("BTC-USDC"))
    f1 = build_features_pos(closes, highs, lows, vols)
    f3 = build_features_ladder(closes, highs, lows, vols)

    from lightgbm import LGBMClassifier

    Xraw, Xmech, ybin, keys, times = [], [], [], [], []
    for p in top:
        fa = f1[p]; fc = f3[p].reindex(fa.index); fm = mech[p].reindex(fa.index)
        c = closes[p].to_numpy()
        lb = np.zeros(len(c), dtype=np.int8)
        for (ei, xi, ep, xp, net, dur) in segs[p]:
            lb[ei] = 1
        ok = fa.notna().all(axis=1).to_numpy() \
            & fm.notna().all(axis=1).to_numpy()
        idx = np.where(ok)[0]
        Xraw.append(np.concatenate([fa[ok].values, fc[ok].values], axis=1)
                    .astype(np.float64))
        Xmech.append(fm[ok].values.astype(np.float64))
        ybin.append(lb[idx])
        keys += [(p, int(i)) for i in idx]
        times += [t.timestamp() for t in fa[ok].index]
    X = np.concatenate([np.vstack(Xraw), np.vstack(Xmech)], axis=1)
    ybin = np.concatenate(ybin)
    pos = np.array(times)
    log(f"rows {len(ybin)} pos {ybin.mean():.4f} feats {X.shape[1]}")

    bnds = np.linspace(pos.min(), pos.max() + 1, 6, dtype=int)
    equity = 100.0
    peak = equity
    max_dd = 0.0
    fold_rets = []
    n_trades_total = 0
    bar_order = sorted(range(len(times)), key=lambda j: times[j])

    for k in range(1, 5):
        trm = pos < bnds[k - 1]
        tem = (pos >= bnds[k - 1]) & (pos < bnds[k])
        log(f"--- fold {k}: train {trm.sum()} test {tem.sum()} ---")
        clf = LGBMClassifier(n_estimators=300, learning_rate=0.05,
                             max_depth=7, num_leaves=63, subsample=0.8,
                             colsample_bytree=0.8, verbose=-1, n_jobs=4)
        clf.fit(X[trm], ybin[trm])
        p = clf.predict_proba(X)[:, 1]

        cash = equity
        in_pos = False
        units = 0.0
        entry_px = 0.0
        run_hi = 0.0
        held_pair = None
        bars_held = 0
        fold_trades = 0

        for j in bar_order:
            if not tem[j]:
                continue
            p_ = keys[j][0]
            i = keys[j][1]
            c = closes[p_].to_numpy()
            if in_pos and p_ == held_pair:
                bars_held += 1
                px = c[i]
                stop_px = None
                if px <= entry_px * (1 - HARD_STOP):
                    stop_px = entry_px * (1 - HARD_STOP)
                else:
                    run_hi = max(run_hi, px)
                    if px <= run_hi * (1 - TRAIL):
                        stop_px = px
                    elif bars_held >= CAP_BARS:
                        stop_px = px
                if stop_px is not None:
                    cash += stop_px * units * (1 - FEE)
                    in_pos = False
                    held_pair = None
                    fold_trades += 1
            elif (not in_pos) and p[j] >= THR and i + 1 < len(c):
                stake = cash * STAKE
                entry_px = c[i] * (1 + SLIP)
                units = stake * (1 - FEE) / entry_px
                cash -= stake
                in_pos = True
                held_pair = p_
                run_hi = entry_px
                bars_held = 0
        if in_pos:
            px = closes[held_pair].iloc[-1]
            cash += px * units * (1 - FEE)
            in_pos = False
            fold_trades += 1
        fold_ret = cash / equity - 1.0
        equity = cash
        peak = max(peak, equity)
        max_dd = min(max_dd, equity / peak - 1.0)
        fold_rets.append(round(fold_ret * 100, 1))
        n_trades_total += fold_trades
        log(f"  fold {k}: {fold_ret:+.1%} (trades {fold_trades}, "
            f"equity ${equity:.2f})")

    total_pct = round((equity / 100 - 1) * 100, 1)
    log("")
    log(f"DAILY FUSION BOT: $100 -> ${equity:.2f} ({total_pct:+.1f}%) "
        f"over {span_y:.1f}y, folds {fold_rets}, trades {n_trades_total}, "
        f"max_dd {max_dd:.1%}")
    log(f"GEN2B CHAMPION:   $100 -> ${GEN2B['end']:.2f} "
        f"({GEN2B['total_pct']:+.1f}%) over {GEN2B['years']}y, "
        f"folds {GEN2B['folds']}")
    report = {
        "daily_bot": {"end": round(equity, 2), "total_pct": total_pct,
                      "folds_pct": fold_rets, "trades": n_trades_total,
                      "max_dd": round(max_dd, 4), "years": round(span_y, 2),
                      "thr": THR, "stake": STAKE,
                      "note": "exits checked on held coin's bars only"},
        "gen2b": GEN2B}
    os.makedirs("reports", exist_ok=True)
    with open(f"{OUT}.json", "w") as fh:
        json.dump(report, fh, indent=1)
    log("report written")


if __name__ == "__main__":
    main()
