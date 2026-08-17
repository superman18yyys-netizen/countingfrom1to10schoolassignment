"""Oracle-mirror daily bot — the oracle's decision procedure as stage 0.

The oracle never scores every bar; it WAITS for a confirmed pivot low
then acts. This bot mirrors that:
  Stage 0 (mechanical, 95% recall measured): candidate bars = from the
    first close >= low x 1.01 after a zigzag pivot low, until the high.
    Base rate on candidates: ~7% (vs 2.5% global) — 2.8x concentration.
  Stage 1 (fusion model on candidates only): raw + mechanical features
    PLUS the oracle's own zigzag state (bars since confirm, distance
    above the low, leg age, return since confirmation).
  Stage 2: threshold on candidate precision, hard stop, trail, cap.
Then the same walk-forward compounding head-to-head vs gen2b on $100.
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
    SLIP, FEE, TOP_N, DB_1H, ENTRY_CONFIRM, ZIG,
    _zigzag, realistic_segments, build_features_pos,
    build_features_ladder, mechanical_layer)

from bot.data.store import Store  # noqa: E402

OUT = "reports/oracle_mirror"
THR = 0.30
STAKE = 0.95
HARD_STOP = 0.02
TRAIL = 0.02
CAP_BARS = 30
GEN2B = {"folds": [38.5, 89.5, 308.9], "total_pct": 973.0,
         "end": 1073.17, "years": 9.9, "name": "gen2b champion"}


def log(m):
    print(m, flush=True)


def candidate_masks(closes):
    """Relaxed oracle candidates: bars from first-1%-confirm off a
    zigzag low until the swing high. Returns {pair: bool array} and
    the zigzag-state features per candidate bar."""
    masks, states = {}, {}
    for p, close in closes.items():
        c = close.to_numpy()
        piv = _zigzag(c, ZIG)
        mask = np.zeros(len(c), dtype=bool)
        st = {"bars_since_confirm": np.full(len(c), np.nan),
              "dist_above_low": np.full(len(c), np.nan),
              "leg_age": np.full(len(c), np.nan),
              "ret_since_confirm": np.full(len(c), np.nan)}
        pending = None
        for i, px, kind in piv:
            if kind == "lo":
                pending = (i, px)
            elif kind == "hi" and pending is not None:
                lo_i, lo_p = pending
                ei = lo_i
                while ei < i and c[ei] < lo_p * (1 + ENTRY_CONFIRM):
                    ei += 1
                if ei < i:
                    conf_px = c[ei]
                    for t in range(ei, i):
                        mask[t] = True
                        st["bars_since_confirm"][t] = t - ei
                        st["dist_above_low"][t] = c[t] / lo_p - 1.0
                        st["leg_age"][t] = t - lo_i
                        st["ret_since_confirm"][t] = c[t] / conf_px - 1.0
                pending = None
        masks[p] = mask
        states[p] = pd.DataFrame(st, index=close.index)
    return masks, states


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
    masks, states = candidate_masks(closes)
    tot_bars = sum(len(closes[p]) for p in top)
    tot_cand = sum(int(masks[p].sum()) for p in top)
    tot_entries = sum(len(v) for v in segs.values())
    hits = sum(len(set(np.where(masks[p])[0]) & set(s[0] for s in segs[p]))
               for p in top)
    log(f"candidates: {tot_cand}/{tot_bars} bars, base rate {hits/tot_cand:.1%}"
        f" (global {tot_entries/tot_bars:.2%}), recall "
        f"{hits/tot_entries:.0%}")

    mech = mechanical_layer(closes, highs, lows, vols, closes.get("BTC-USDC"))
    f1 = build_features_pos(closes, highs, lows, vols)
    f3 = build_features_ladder(closes, highs, lows, vols)

    from lightgbm import LGBMClassifier

    X, ybin, keys, times = [], [], [], []
    for p in top:
        fa = f1[p]; fc = f3[p].reindex(fa.index); fm = mech[p].reindex(fa.index)
        fs = states[p].reindex(fa.index)
        c = closes[p].to_numpy()
        lb = np.zeros(len(c), dtype=np.int8)
        for (ei, xi, ep, xp, net, dur) in segs[p]:
            lb[ei] = 1
        ok = (fa.notna().all(axis=1) & fm.notna().all(axis=1)
              & fs.notna().all(axis=1) & masks[p]).to_numpy()
        idx = np.where(ok)[0]
        X.append(np.concatenate([fa[ok].values, fc[ok].values,
                                 fm[ok].values, fs[ok].values], axis=1)
                 .astype(np.float64))
        ybin.append(lb[idx])
        keys += [(p, int(i)) for i in idx]
        times += [t.timestamp() for t in fa[ok].index]
    X = np.vstack(X)
    ybin = np.concatenate(ybin)
    pos = np.array(times)
    log(f"candidate rows {len(ybin)} pos {ybin.mean():.4f} feats {X.shape[1]}")

    bnds = np.linspace(pos.min(), pos.max() + 1, 6, dtype=int)
    equity = 100.0
    peak = equity
    max_dd = 0.0
    fold_rets = []
    n_trades_total = 0
    bar_order = sorted(range(len(times)), key=lambda j: times[j])
    frontier = {}
    preds = {}

    for k in range(2, 5):
        trm = pos < bnds[k - 1]
        tem = (pos >= bnds[k - 1]) & (pos < bnds[k])
        clf = LGBMClassifier(n_estimators=300, learning_rate=0.05,
                             max_depth=7, num_leaves=63, subsample=0.8,
                             colsample_bytree=0.8, verbose=-1, n_jobs=4)
        clf.fit(X[trm], ybin[trm])
        p = clf.predict_proba(X)[:, 1]
        preds[k] = (p, tem)
        for t in (0.20, 0.25, 0.30, 0.35, 0.40):
            sel = tem & (p >= t)
            if sel.sum() < 8:
                continue
            frontier.setdefault(t, []).append(float(ybin[sel].mean()))
        log(f"fold {k} predictions cached")

    results = {}
    for THR in (0.30, 0.40):
        equity = 100.0
        peak = equity
        max_dd = 0.0
        fold_rets = []
        n_trades_total = 0
        for k in range(2, 5):
            p, tem = preds[k]
            cash = equity
            free_ts = -1.0
            fold_trades = 0
            for j in bar_order:
                if not tem[j]:
                    continue
                if times[j] <= free_ts:
                    continue
                p_ = keys[j][0]
                i = keys[j][1]
                c = closes[p_].to_numpy()
                if p[j] >= THR and i + 1 < len(c):
                    stake = cash * STAKE
                    entry_px = c[i] * (1 + SLIP)
                    units = stake * (1 - FEE) / entry_px
                    # exact exit scan over EVERY 4H bar of the pair
                    n = len(c)
                    run_hi = entry_px
                    exit_px = None
                    exit_k = i + 1
                    for kk in range(1, min(CAP_BARS, n - i - 1) + 1):
                        px = c[i + kk]
                        exit_k = i + kk
                        if px <= entry_px * (1 - HARD_STOP):
                            exit_px = entry_px * (1 - HARD_STOP)
                            break
                        run_hi = max(run_hi, px)
                        if px <= run_hi * (1 - TRAIL):
                            exit_px = px
                            break
                        if kk == min(CAP_BARS, n - i - 1):
                            exit_px = px
                    if exit_px is None:
                        exit_px = c[min(i + CAP_BARS, n - 1)]
                    cash -= stake
                    cash += exit_px * units * (1 - FEE)
                    fold_trades += 1
                    free_ts = closes[p_].index[
                        min(exit_k, n - 1)].timestamp()
                    peak = max(peak, cash)
                    max_dd = min(max_dd, cash / peak - 1.0)
            fold_ret = cash / equity - 1.0
            equity = cash
            peak = max(peak, equity)
            max_dd = min(max_dd, equity / peak - 1.0)
            fold_rets.append(round(fold_ret * 100, 1))
            n_trades_total += fold_trades
            log(f"THR {THR:.2f} fold {k}: {fold_ret:+.1%} "
                f"(trades {fold_trades}, equity ${equity:.2f})")
        results[THR] = {"end": round(equity, 2),
                        "total_pct": round((equity / 100 - 1) * 100, 1),
                        "folds_pct": fold_rets, "trades": n_trades_total,
                        "max_dd": round(max_dd, 4)}

    log("")
    log("candidate precision (mean over folds): "
        + ", ".join(f"{t:.2f}->{np.mean(v):.0%}" for t, v in frontier.items()))
    for THR, res in results.items():
        log(f"ORACLE-MIRROR thr {THR:.2f}: $100 -> ${res['end']:.2f} "
            f"({res['total_pct']:+.1f}%) folds {res['folds_pct']} "
            f"trades {res['trades']} max_dd {res['max_dd']:.1%}")
    log(f"GEN2B CHAMPION:    $100 -> ${GEN2B['end']:.2f} "
        f"({GEN2B['total_pct']:+.1f}%) over {GEN2B['years']}y")
    report = {"candidate_precision_by_thr":
              {str(t): round(float(np.mean(v)), 4)
               for t, v in frontier.items()},
              "years": round(span_y, 2), "stake": STAKE,
              "daily_bot": results,
              "gen2b": GEN2B}
    os.makedirs("reports", exist_ok=True)
    with open(f"{OUT}.json", "w") as fh:
        json.dump(report, fh, indent=1)
    log("report written")


if __name__ == "__main__":
    main()
