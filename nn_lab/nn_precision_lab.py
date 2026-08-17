"""NN precision lab — self-contained research mirror for CI compute.

Vendored research modules (realistic-segment builder + three feature
views) mirror athena-local's daily-bot R&D so heavy neural-network
training can run on GitHub Actions. The Athena champion and the live
bot stay local-only; nothing in this directory trades.

Question under test: can neural networks with sign-asymmetric loss
push entry precision beyond the LightGBM frontier (41% at 4H)?
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from bot.data.store import Store
from bot.indicators.ta import atr, rsi, sma

# ---------------- vendored constants (mirror of athena-local) ----------------
SLIP, FEE = 0.001, 0.006
ENTRY_CONFIRM = 0.010
EXIT_TRAIL = 0.020
MIN_NET = 0.015
MIN_DUR, MAX_DUR = 4, 120
ZIG = 0.02
SEED = 7

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(4)

DB = "data/universe.db"
OUT = "reports/nn_precision"


def log(m):
    print(m, flush=True)


# -------------------------- vendored: athena.py _zigzag ----------------------
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


# ------------------- vendored: v3_realistic.realistic_segments ---------------
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


# ------------- vendored: athena_committee feature views ----------------------
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


def build_features_flow(closes, highs, lows, vols):
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
        f["body_pos"] = ((close - close.shift(1)) / rng).fillna(0.0)
        f["buy_pressure"] = v * f["close_pos"]
        f["sell_pressure"] = v * (1 - f["close_pos"])
        f["net_flow"] = f["buy_pressure"] - f["sell_pressure"]
        f["flow_z"] = (f["net_flow"] - f["net_flow"].rolling(336).mean()) \
            / f["net_flow"].rolling(336).std().replace(0.0, np.nan)
        f["flow_mom"] = f["net_flow"].rolling(24).mean() \
            / f["net_flow"].rolling(168).mean().replace(0.0, np.nan)
        f["vol_z"] = (np.log(v) - np.log(v).rolling(336).mean()) \
            / np.log(v).rolling(336).std().replace(0.0, np.nan)
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


# ----------------------------- raw price windows -----------------------------
def window(c, h, l, v, t):
    wc = c[t - 47:t + 1]; wh = h[t - 47:t + 1]
    wl = l[t - 47:t + 1]; wv = v[t - 47:t + 1]
    if len(wc) < 48 or wv.min() <= 0 or wc.min() <= 0:
        return None
    r = np.log(wc / wc[0])
    rng = (wh - wl) / wc
    vz = np.log(wv / wv.mean())
    vol = np.log(wv / wv[-1])
    d20 = wc[-1] / wc[-20:].mean() - 1
    d50 = wc[-1] / wc[-50:].mean() - 1
    return np.concatenate([r, rng, vz, vol, [d20, d50, d20 - d50]])


# ---------------------------------- models -----------------------------------
class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.35),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.35),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(128, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)


class CNN(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2))
        self.fc = nn.Sequential(
            nn.Linear(128 * (d // 4), 256), nn.ReLU(), nn.Dropout(0.35),
            nn.Linear(256, 1))

    def forward(self, x):
        return self.fc(self.conv(x.unsqueeze(1)).flatten(1)).squeeze(1)


def fit(model, Xt, yt, Xe, pos_w=0.35, epochs=120, bs=4096):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w))
    n = Xt.shape[0]
    for e in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            l = lossf(model(Xt[idx]), yt[idx])
            l.backward()
            opt.step()
            tot += float(l)
        sched.step()
        if e % 30 == 0 or e == epochs - 1:
            log(f"    epoch {e}: loss {tot / max(1, n // bs):.4f}")
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(Xe)).numpy()


def prep(M, tr, te, mu=None, sd=None):
    if mu is None:
        mu = M[tr].mean(0, keepdims=True)
        sd = M[tr].std(0, keepdims=True) + 1e-9
    Xt = torch.tensor(((M[tr] - mu) / sd).astype(np.float32))
    Xe = torch.tensor(((M[te] - mu) / sd).astype(np.float32))
    return Xt, torch.tensor(y[tr].astype(np.float32)), Xe


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
    log(f"top-20: {top[:5]} ...")
    closes = {p: cand[p]["close"] for p in top}
    highs = {p: cand[p]["high"] for p in top}
    lows = {p: cand[p]["low"] for p in top}
    vols = {p: cand[p]["volume"] for p in top}
    segs = realistic_segments(closes)
    log(f"segments: {sum(len(v) for v in segs.values())}")
    log("building feature views...")
    f1 = build_features_pos(closes, highs, lows, vols)
    f2 = build_features_flow(closes, highs, lows, vols)
    f3 = build_features_ladder(closes, highs, lows, vols)

    X1, X2, X3, W, y, times = [], [], [], [], [], []
    for p in top:
        fa = f1[p]; fb = f2[p].reindex(fa.index); fc = f3[p].reindex(fa.index)
        c = closes[p].to_numpy(); h = highs[p].to_numpy()
        l = lows[p].to_numpy(); v = vols[p].to_numpy()
        lb = np.zeros(len(c), dtype=np.int8)
        for (ei, xi, ep, xp, net, dur) in segs[p]:
            lb[ei] = 1
        ok = fa.notna().all(axis=1).to_numpy()
        idx = np.where(ok)[0]
        X1.append(fa[ok].values.astype(np.float64))
        X2.append(fb[ok].values.astype(np.float64))
        X3.append(fc[ok].values.astype(np.float64))
        wrows = [window(c, h, l, v, i) if i >= 200 else None for i in idx]
        keep = [j for j, w in enumerate(wrows) if w is not None]
        W.append(np.array([wrows[j] for j in keep], dtype=np.float64))
        y.append(lb[idx])
        times += [t.timestamp() for t in fa[ok].index]
    X1 = np.vstack(X1); X2 = np.vstack(X2); X3 = np.vstack(X3)
    W = np.vstack(W)
    y = np.concatenate(y)
    pos = np.array(times)
    b = np.linspace(pos.min(), pos.max() + 1, 6, dtype=int)
    tr = pos < b[3]
    te = (pos >= b[3]) & (pos < b[4])
    log(f"rows {len(y)} | train {tr.sum()} | test {te.sum()} | "
        f"pos_rate {y.mean():.4f} | window dim {W.shape[1]}")

    from lightgbm import LGBMClassifier
    log("LightGBM baseline (3 views)...")
    def mk(Xa):
        return LGBMClassifier(n_estimators=250, learning_rate=0.05,
                              max_depth=7, num_leaves=63, subsample=0.8,
                              colsample_bytree=0.8, verbose=-1,
                              n_jobs=4).fit(Xa[tr], y[tr])
    m1 = mk(X1); m2 = mk(X2); m3 = mk(X3)
    pb = (0.35 * m1.predict_proba(X1)[:, 1]
          + 0.35 * m2.predict_proba(X2)[:, 1]
          + 0.30 * m3.predict_proba(X3)[:, 1])

    log("MLP tabular (full data, sign-asymmetric)...")
    F = np.concatenate([X1, X2, X3], axis=1)
    Ft, Fyt, Fe = prep(F, tr, te)
    p_mlp = fit(MLP(Ft.shape[1]), Ft, Fyt, Fe)

    log("CNN raw 48-bar windows (full data, sign-asymmetric)...")
    Wt, Wyt, We = prep(W, tr, te)
    p_cnn = fit(CNN(Wt.shape[1]), Wt, Wyt, We)

    ens = (pb[te] + p_mlp + p_cnn) / 3.0

    from sklearn.metrics import roc_auc_score
    report = {"rows": int(len(y)), "train": int(tr.sum()),
              "test": int(te.sum()), "pos_rate": float(y.mean())}
    report["auc"] = {
        "gbdt": float(roc_auc_score(y[te], pb[te])),
        "mlp": float(roc_auc_score(y[te], p_mlp)),
        "cnn": float(roc_auc_score(y[te], p_cnn)),
        "ensemble": float(roc_auc_score(y[te], ens))}
    log("AUC: " + json.dumps(report["auc"], indent=1))
    for name, p in (("gbdt", pb[te]), ("mlp", p_mlp),
                    ("cnn", p_cnn), ("ensemble", ens)):
        tab = []
        for t in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40):
            sel = p >= t
            if sel.sum() < 8:
                continue
            tab.append({"thr": t, "precision": float(y[te][sel].mean()),
                        "recall": float(y[te][sel].sum() / y[te].sum()),
                        "n": int(sel.sum())})
        report[name] = tab
        log(f"--- {name} ---")
        for row in tab:
            log(f"  {row['thr']:.2f} | prec {row['precision']:.0%} | "
                f"rec {row['recall']:.0%} | n {row['n']}")

    os.makedirs("reports", exist_ok=True)
    with open(f"{OUT}.json", "w") as fh:
        json.dump(report, fh, indent=1)
    np.savez_compressed(f"{OUT}_probs.npz", y_te=y[te], gbdt=pb[te],
                        mlp=p_mlp, cnn=p_cnn, ens=ens)
    log("done")


if __name__ == "__main__":
    main()
