"""Oracle-ranking lab — learn the oracle's DECISION, not just its entries.

The oracle's daily segments are not uniform: median net 5.7%, mean 7.7%,
fat tail to 13.5x. A binary buy/no-buy classifier never sees the size of
the opportunity. This lab trains models to PREDICT THE SEGMENT NET GAIN
at the entry bar, then evaluates selection quality as the oracle does:
of the bars the model picks, how many are real oracle entries, and what
does the realized net compound to (the oracle itself: 60bn %/yr).

Self-contained research mirror — the Athena champion stays local.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from bot.data.store import Store
from bot.indicators.ta import atr, rsi, sma

SLIP, FEE = 0.001, 0.006
ENTRY_CONFIRM, EXIT_TRAIL, MIN_NET, MIN_DUR, MAX_DUR = 0.010, 0.020, 0.015, 4, 120
ZIG = 0.02
SEED = 7
np.random.seed(SEED)

DB = "data/universe.db"
OUT = "reports/oracle_rank_lab"


def log(m):
    print(m, flush=True)


def _zigzag(close: np.ndarray, pct: float):
    piv, mode, ext_i, ext_p = [], "init", 0, float(close[0])
    for i in range(len(close)):
        p = float(close[i])
        if mode in ("init", "up"):
            if mode == "init":
                if p > ext_p * (1 + pct):
                    mode, ext_i, ext_p = "up", i, p
                    continue
                if p < ext_p:
                    ext_i, ext_p = i, p
            elif p > ext_p:
                ext_i, ext_p = i, p
            elif p <= ext_p * (1 - pct):
                piv.append((ext_i, ext_p, "hi"))
                mode, ext_i, ext_p = "down", i, p
        if mode == "down":
            if p < ext_p:
                ext_i, ext_p = i, p
            elif p >= ext_p * (1 + pct):
                piv.append((ext_i, ext_p, "lo"))
                mode, ext_i, ext_p = "up", i, p
    return piv


def realistic_segments(closes, pct=ZIG):
    out = {}
    for p, close in closes.items():
        c = close.to_numpy()
        piv = _zigzag(c, pct)
        segs = []
        pending = None
        for i, px, kind in piv:
            if kind == "lo":
                pending = (i, px)
            elif kind == "hi" and pending is not None:
                lo_i, lo_p = pending
                ei = lo_i
                while ei < i and c[ei] < lo_p * (1 + ENTRY_CONFIRM):
                    ei += 1
                if ei >= i:
                    pending = None
                    continue
                entry_p = c[ei]
                run_hi = entry_p
                xi = ei + 1
                while xi <= i:
                    run_hi = max(run_hi, c[xi])
                    if c[xi] <= run_hi * (1 - EXIT_TRAIL):
                        break
                    xi += 1
                xi = min(xi, i)
                exit_p = c[xi]
                net = (exit_p / entry_p) * (1 - SLIP) * (1 - FEE) \
                    / ((1 + SLIP) * (1 + FEE)) - 1.0
                dur = xi - ei
                if net >= MIN_NET and MIN_DUR <= dur <= MAX_DUR:
                    segs.append((ei, xi, entry_p, exit_p, net, dur))
                pending = None
        out[p] = segs
    return out


def build_features_pos(closes, highs, lows, vols):
    feats = {}
    for pair in sorted(closes):
        close, high, low = closes[pair], highs[pair], lows[pair]
        v = vols[pair].replace(0.0, np.nan)
        a = atr(high, low, close, 14) / close
        atr_pctl = a.rolling(540, min_periods=100).apply(
            lambda x: (x <= x[-1]).mean(), raw=True)
        rng = (high - low).replace(0.0, np.nan)
        f = pd.DataFrame(index=close.index)
        f["mom40"] = close / close.shift(40) - 1.0
        f["mom90"] = close / close.shift(540) - 1.0
        f["mom12"] = close / close.shift(12) - 1.0
        f["rsi14"] = rsi(close, 14)
        f["atr_pct"] = a
        f["atr_pctl"] = atr_pctl
        f["dd1y"] = close / close.rolling(2160, min_periods=100).max() - 1.0
        f["surge12"] = close / close.shift(12) - 1.0
        f["sma_dist"] = close / sma(close, 200) - 1.0
        f["close_pos"] = ((close - low) / rng).fillna(0.5)
        f["buy_pressure"] = v * f["close_pos"]
        f["net_flow"] = f["buy_pressure"] - v * (1 - f["close_pos"])
        f["flow_z"] = (f["net_flow"] - f["net_flow"].rolling(336).mean()) \
            / f["net_flow"].rolling(336).std().replace(0.0, np.nan)
        f["flow_mom"] = f["net_flow"].rolling(24).mean() \
            / f["net_flow"].rolling(168).mean().replace(0.0, np.nan)
        f["vol_z"] = (np.log(v) - np.log(v).rolling(336).mean()) \
            / np.log(v).rolling(336).std().replace(0.0, np.nan)
        f["dist_hi20"] = close / high.rolling(20).max() - 1.0
        feats[pair] = f
    return feats


def build_features_ladder(closes, highs, lows, vols):
    pairs = sorted(closes)
    feats = {}
    for pair in pairs:
        close, high, low = closes[pair], highs[pair], lows[pair]
        v = vols[pair] if pair in vols else None
        a = atr(high, low, close, 14) / close
        atr_pctl = a.rolling(540, min_periods=100).apply(
            lambda x: (x <= x[-1]).mean(), raw=True)
        f = pd.DataFrame(index=close.index)
        f["mom40"] = close / close.shift(40) - 1.0
        f["mom90"] = close / close.shift(540) - 1.0
        f["mom6"] = close / close.shift(6) - 1.0
        f["mom12"] = close / close.shift(12) - 1.0
        f["mom168"] = close / close.shift(168) - 1.0
        f["mom360"] = close / close.shift(360) - 1.0
        f["rsi14"] = rsi(close, 14)
        f["rsi6"] = rsi(close, 6)
        f["atr_pct"] = a
        f["atr_pctl"] = atr_pctl
        f["dd1y"] = close / close.rolling(2160, min_periods=100).max() - 1.0
        f["surge12"] = close / close.shift(12) - 1.0
        f["sma_dist"] = close / sma(close, 200) - 1.0
        f["ret_std"] = close.pct_change().rolling(120).std(ddof=0)
        rng = (high - low).replace(0.0, np.nan)
        f["close_pos"] = ((close - low) / rng).fillna(0.5)
        f["range_pct"] = ((high - low) / close).fillna(0.0)
        if v is not None:
            vv = v.replace(0.0, np.nan)
            f["vol_z"] = (np.log(vv) - np.log(vv).rolling(336).mean()) \
                / np.log(vv).rolling(336).std().replace(0.0, np.nan)
            f["vol_mom"] = vv.rolling(24).mean() \
                / vv.rolling(168).mean().replace(0.0, np.nan)
        feats[pair] = f
    tl = pd.DatetimeIndex(sorted(set().union(
        *[set(f.index) for f in feats.values()])))
    for hor in ("mom12", "mom40", "mom90"):
        mat = pd.DataFrame({p: feats[p][hor].reindex(tl) for p in pairs})
        ranks = mat.rank(axis=1, pct=True)
        for p in pairs:
            feats[p][f"rank_{hor}"] = ranks[p].reindex(feats[p].index)
    return feats


def main():
    log("loading 4H universe...")
    store = Store(DB)
    cand = {}
    for pair in [r[0] for r in store.conn.execute(
            "SELECT DISTINCT pair FROM candles WHERE granularity='FOUR_HOUR'")]:
        df = store.load_candles(pair, "FOUR_HOUR")
        if df is not None and len(df) >= 8000:
            cand[pair] = df.dropna()
    med = {p: float((d["volume"] * d["close"]).tail(1000).median())
           for p, d in cand.items()
           if (d["volume"] * d["close"]).tail(1000).notna().any()}
    top = sorted(med, key=lambda p: -med[p])[:20]
    closes = {p: cand[p]["close"] for p in top}
    highs = {p: cand[p]["high"] for p in top}
    lows = {p: cand[p]["low"] for p in top}
    vols = {p: cand[p]["volume"] for p in top}
    segs = realistic_segments(closes)
    log(f"segments: {sum(len(v) for v in segs.values())}")
    f1 = build_features_pos(closes, highs, lows, vols)
    f3 = build_features_ladder(closes, highs, lows, vols)

    from lightgbm import LGBMClassifier, LGBMRegressor

    X, ybin, ynet, keys, times = [], [], [], [], []
    for p in top:
        fa = f1[p]; fc = f3[p].reindex(fa.index)
        c = closes[p].to_numpy()
        lb = np.zeros(len(c), dtype=np.int8)
        g = np.full(len(c), np.nan)
        for (ei, xi, ep, xp, net, dur) in segs[p]:
            lb[ei] = 1
            g[ei] = net
        ok = fa.notna().all(axis=1).to_numpy()
        idx = np.where(ok)[0]
        X.append(np.concatenate([fa[ok].values, fc[ok].values], axis=1)
                 .astype(np.float64))
        ybin.append(lb[idx])
        ynet.append(g[idx])
        keys += [(p, int(i)) for i in idx]
        times += [t.timestamp() for t in fa[ok].index]
    X = np.vstack(X)
    ybin = np.concatenate(ybin)
    ynet = np.concatenate(ynet)
    pos = np.array(times)
    b = np.linspace(pos.min(), pos.max() + 1, 6, dtype=int)
    tr = pos < b[3]
    te = (pos >= b[3]) & (pos < b[4])
    # train target: real segments only get their net; others are never
    # used as positive regression samples (NaN dropped for regression)
    log(f"rows {len(ybin)} train {tr.sum()} test {te.sum()} "
        f"pos {ybin.mean():.4f}")

    # --- binary classifier (frontier reference) ---
    bc = LGBMClassifier(n_estimators=250, learning_rate=0.05, max_depth=7,
                        num_leaves=63, subsample=0.8, colsample_bytree=0.8,
                        verbose=-1, n_jobs=4)
    bc.fit(X[tr], ybin[tr])
    p_bin = bc.predict_proba(X)[:, 1]

    # --- regression on segment net (only oracle bars as targets) ---
    reg_tr = tr & (ynet > 0)
    rg = LGBMRegressor(n_estimators=250, learning_rate=0.05, max_depth=7,
                       num_leaves=63, subsample=0.8, colsample_bytree=0.8,
                       verbose=-1, n_jobs=4)
    rg.fit(X[reg_tr], ynet[reg_tr])
    p_reg = rg.predict(X)          # predicted net for EVERY bar
    p_reg = np.nan_to_num(p_reg, nan=0.0)

    # --- ranking: take the top-K predicted-net bars in each fold window ---
    # simple approach: per coin, rank bars by p_reg in test fold, take top K
    import json as _json
    report = {"rows": int(len(ybin)), "train": int(tr.sum()),
              "test": int(te.sum())}
    for K in (10, 20, 40, 80):
        # top-K per coin in test
        sel = np.zeros(len(ybin), dtype=bool)
        for p in top:
            m = (pos >= b[3]) & (pos < b[4]) & np.array(
                [k[0] == p for k in keys])
            if m.sum() == 0:
                continue
            idx = np.where(m)[0]
            order = np.argsort(-p_reg[idx])[:K]
            sel[idx[order]] = True
        n_sel = int(sel.sum())
        n_hit = int((sel & (ybin == 1)).sum())
        prec = n_hit / max(1, n_sel)
        if n_hit > 0:
            realized = float(np.nanmean(ynet[sel & (ybin == 1)]))
        else:
            realized = 0.0
        report[f"top{K}"] = {"n": n_sel, "hits": n_hit,
                             "precision": prec, "mean_realized_net": realized}
        log(f"top-{K}/coin test: {n_sel} picks, {n_hit} oracle hits "
            f"({prec:.0%} precision), realized mean net {realized:.1%}")

    # binary-classifier precision reference at high thresholds
    for t in (0.20, 0.25, 0.30):
        sel = (p_bin >= t) & te
        if sel.sum() < 8:
            continue
        log(f"binary thr {t:.2f}: n {sel.sum()} precision "
            f"{ybin[sel].mean():.0%}")

    os.makedirs("reports", exist_ok=True)
    with open(f"{OUT}.json", "w") as fh:
        _json.dump(report, fh, indent=1)
    np.savez_compressed(f"{OUT}_probs.npz", y_te=ybin[te], p_bin=p_bin[te],
                        p_reg=p_reg[te])
    log("done")


if __name__ == "__main__":
    main()
