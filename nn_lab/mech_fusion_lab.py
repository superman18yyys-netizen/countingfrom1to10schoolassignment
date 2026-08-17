"""Mechanical-fusion lab — the model gets raw data AND calculated data.

Layer 1 (mechanical, per bar, causal): evaluate a battery of rules —
the DISCOVERED reasoning gates (mom90>=5%, atr_pctl>=0.60, dd1y>=-30%),
trend stacks, breakouts, pullbacks, volume spikes, BTC regime, weekly
opportunity density, time cycles. Each rule output is a number.

Layer 2 (fusion): LightGBM receives [raw features + mechanical rule
outputs] and decides. The model composes the rules instead of
relearning them — the "never built before" step: mechanical knowledge
as inputs, learned composition on top.

Ablations measured on the same test fold:
  A) pure mechanical (all discovered gates AND'd)      — rule-only baseline
  B) model on raw features only                        — 37-41% reference
  C) model on raw + mechanical features                — the fusion
  D) top-K fat-tail selection WITHIN C's gated set     — the oracle-decision
Self-contained research mirror; the Athena champion stays local.
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
OUT = "reports/mech_fusion_lab"


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


# ------------------- LAYER 1: mechanical rule outputs -------------------
def mechanical_layer(closes, highs, lows, vols, btc_close=None):
    """Per-bar rule evaluation — the 'data calculated'. All causal."""
    out = {}
    btc_sma = None
    if btc_close is not None:
        btc_sma = sma(btc_close, 200)
    for pair in sorted(closes):
        close, high, low = closes[pair], highs[pair], lows[pair]
        v = vols[pair].replace(0.0, np.nan)
        a = atr(high, low, close, 14) / close
        atr_pctl = a.rolling(540, min_periods=100).apply(
            lambda x: (x <= x[-1]).mean(), raw=True)
        m = pd.DataFrame(index=close.index)
        # discovered reasoning gates (from 737 winning-week buys)
        m["g_mom90"] = (close / close.shift(540) - 1.0 >= 0.05).astype(np.int8)
        m["g_atr"] = (atr_pctl >= 0.60).astype(np.int8)
        m["g_dd"] = (close / close.rolling(2160, min_periods=100).max() - 1.0
                     >= -0.30).astype(np.int8)
        # trend stack
        s50, s200 = sma(close, 50), sma(close, 200)
        m["r_trend"] = ((close > s50) & (s50 > s200)).astype(np.int8)
        # breakouts (new N-bar high, exclusive of current bar)
        m["r_brk20"] = (close > high.rolling(21).max().shift(1)).astype(np.int8)
        m["r_brk55"] = (close > high.rolling(56).max().shift(1)).astype(np.int8)
        # pullback in uptrend
        m["r_pull"] = ((close < s50 * 1.03) & (close > s50 * 0.97)
                       & (close > s200)).astype(np.int8)
        # RSI zone (established, not exhausted)
        r14 = rsi(close, 14)
        m["r_rsi"] = ((r14 >= 45) & (r14 <= 65)).astype(np.int8)
        # volume expansion
        m["r_vol"] = (v > 1.5 * v.rolling(20).mean()).astype(np.int8)
        # weekly opportunity density (up-moves in prior week)
        up = (close.pct_change() > 0.01).astype(np.int8)
        m["r_density"] = up.rolling(168, min_periods=20).sum()
        # vol-of-vol
        m["r_vov"] = a.rolling(168, min_periods=20).std() / (
            a.rolling(168, min_periods=20).mean().replace(0.0, np.nan))
        # time cycles
        m["r_dow"] = close.index.dayofweek.astype(np.int8)
        m["r_hod"] = (close.index.hour // 4).astype(np.int8)
        # BTC regime (market beta), if provided
        if btc_sma is not None:
            b = btc_close.reindex(close.index)
            bs = btc_sma.reindex(close.index)
            m["r_btc"] = ((b > bs)).astype(np.int8)
        # mechanical committee vote: sum of binary rules
        cols = [c for c in m.columns if c.startswith(("g_", "r_"))
                and c not in ("r_density", "r_vov", "r_dow", "r_hod")]
        m["mech_vote"] = m[cols].sum(axis=1)
        out[pair] = m
    return out


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
    btc = closes.get("BTC-USDC")
    log(f"segments: {sum(len(v) for v in realistic_segments(closes).values())}")

    mech = mechanical_layer(closes, highs, lows, vols, btc)
    f1 = build_features_pos(closes, highs, lows, vols)
    f3 = build_features_ladder(closes, highs, lows, vols)

    from lightgbm import LGBMClassifier, LGBMRegressor

    Xraw, Xmech, ybin, ynet, keys, times = [], [], [], [], [], []
    for p in top:
        fa = f1[p]; fc = f3[p].reindex(fa.index); fm = mech[p].reindex(fa.index)
        c = closes[p].to_numpy()
        lb = np.zeros(len(c), dtype=np.int8)
        g = np.full(len(c), np.nan)
        for (ei, xi, ep, xp, net, dur) in realistic_segments(closes)[p]:
            lb[ei] = 1
            g[ei] = net
        ok = fa.notna().all(axis=1).to_numpy() \
            & fm.notna().all(axis=1).to_numpy()
        idx = np.where(ok)[0]
        Xraw.append(np.concatenate([fa[ok].values, fc[ok].values], axis=1)
                    .astype(np.float64))
        Xmech.append(fm[ok].values.astype(np.float64))
        ybin.append(lb[idx])
        ynet.append(g[idx])
        keys += [(p, int(i)) for i in idx]
        times += [t.timestamp() for t in fa[ok].index]
    Xraw = np.vstack(Xraw); Xmech = np.vstack(Xmech)
    ybin = np.concatenate(ybin); ynet = np.concatenate(ynet)
    pos = np.array(times)
    b = np.linspace(pos.min(), pos.max() + 1, 6, dtype=int)
    tr = pos < b[3]
    te = (pos >= b[3]) & (pos < b[4])
    log(f"rows {len(ybin)} train {tr.sum()} test {te.sum()} "
        f"pos {ybin.mean():.4f} | raw {Xraw.shape[1]}f mech {Xmech.shape[1]}f")

    # A) pure mechanical: all discovered gates
    mech_gates = (Xmech[:, 0] == 1) & (Xmech[:, 1] == 1) & (Xmech[:, 2] == 1)
    nA = int((mech_gates & te).sum())
    pA = float(ybin[mech_gates & te].mean()) if nA else 0.0
    log(f"[A] pure mechanical (3 gates): n {nA} precision {pA:.0%}")

    # B) raw-only model (reference frontier)
    bc = LGBMClassifier(n_estimators=250, learning_rate=0.05, max_depth=7,
                        num_leaves=63, subsample=0.8, colsample_bytree=0.8,
                        verbose=-1, n_jobs=4)
    bc.fit(Xraw[tr], ybin[tr])
    pB = bc.predict_proba(Xraw)[:, 1]

    # C) fusion model: raw + mechanical
    Xf = np.concatenate([Xraw, Xmech], axis=1)
    fc_model = LGBMClassifier(n_estimators=250, learning_rate=0.05,
                              max_depth=7, num_leaves=63, subsample=0.8,
                              colsample_bytree=0.8, verbose=-1, n_jobs=4)
    fc_model.fit(Xf[tr], ybin[tr])
    pC = fc_model.predict_proba(Xf)[:, 1]

    report = {"rows": int(len(ybin)), "train": int(tr.sum()),
              "test": int(te.sum()), "pos_rate": float(ybin.mean()),
              "pure_mech": {"n": nA, "precision": pA}}
    log("--- B (raw only) vs C (raw+mech): precision at thresholds ---")
    for name, p in (("B_raw", pB), ("C_fusion", pC)):
        tab = []
        for t in (0.20, 0.25, 0.30, 0.35, 0.40):
            sel = (p >= t) & te
            if sel.sum() < 8:
                continue
            prec = float(ybin[sel].mean())
            tab.append({"thr": t, "precision": prec,
                        "n": int(sel.sum()),
                        "recall": float(ybin[sel].sum() / ybin[te].sum())})
            log(f"  {name} thr {t:.2f}: prec {prec:.0%} n {sel.sum()}")
        report[name] = tab

    # D) fat-tail: rank C's gated set by predicted net, take top-K
    reg_tr = tr & (ynet > 0)
    rg = LGBMRegressor(n_estimators=250, learning_rate=0.05, max_depth=7,
                       num_leaves=63, subsample=0.8, colsample_bytree=0.8,
                       verbose=-1, n_jobs=4)
    rg.fit(Xf[reg_tr], ynet[reg_tr])
    pnet = rg.predict(Xf)
    for gate_t in (0.25, 0.30):
        gated = te & (pC >= gate_t)
        n_g = int(gated.sum())
        if n_g < 8:
            continue
        # within gated: rank by predicted net, top half
        idx = np.where(gated)[0]
        order = idx[np.argsort(-pnet[idx])[: max(1, n_g // 2)]]
        hit = float((ybin[order] == 1).mean())
        rn = float(np.nanmean(ynet[order][ybin[order] == 1])) \
            if (ybin[order] == 1).any() else 0.0
        report[f"D_gate{gate_t}"] = {"n": int(len(order)),
                                     "precision": hit,
                                     "mean_realized_net": rn}
        log(f"[D] gate {gate_t:.2f} n {len(order)}: precision {hit:.0%} "
            f"realized net of hits {rn:.1%}")

    os.makedirs("reports", exist_ok=True)
    with open(f"{OUT}.json", "w") as fh:
        json.dump(report, fh, indent=1)
    np.savez_compressed(f"{OUT}_probs.npz", y_te=ybin[te], pB=pB[te],
                        pC=pC[te], pnet=pnet[te])
    log("done")


if __name__ == "__main__":
    main()
