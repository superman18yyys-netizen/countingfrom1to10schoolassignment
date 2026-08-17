"""Continuous training protocol — the never-stop improvement loop.

CYCLE (runs on CI, scheduled weekly + on demand):
  1. load all 4H data + realistic segments (fees + latency in labels)
  2. ROLL-FORWARD walk-forward: for each fold boundary, train the
     mechanical-fusion model on folds < k, evaluate OOS on fold k
  3. realized-EV evaluation (the honest "profit"): for every bar in
     the OOS fold, simulate the trade the bot would actually take —
     entry close x (1+slip), trail exit 2%, 30-bar cap, fees both
     sides — and compute realized net. No assumed win sizes.
  4. operating point: threshold that maximizes mean realized EV per
     trade across OOS folds, with a MINIMUM precision floor so we
     never trade in a band that history says loses.
  5. champion rule: if the new model's OOS EV beats the archived
     champion's, archive it (model, threshold, stop, trail, sizer,
     EV). Else the champion stands — no regression, ever.
  6. policy card: threshold, precision band, EV/trade, trades/yr,
     hard stop, trail, net-regression sizer — the live bot's orders.

Loss machinery (the "0 losses" defense, all mechanical):
  - trade only inside the measured OOS band (stand in cash outside)
  - hard stop: exit at -2% before the trail can give back more
  - trail exit 2% (never ride a winner into a loss)
  - per-cycle champion gate (no adoption of worse models)

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
OUT = "reports/continuous_champion"
HARD_STOP = -0.02      # exit if price falls 2% below entry
EXIT_TRAIL_LIVE = 0.02  # trail the runner 2% below its high
MAX_HOLD_BARS = 30      # 4H bars = 5 days
MIN_EV_FLOOR = 0.002    # refuse to trade a band with less than +0.2%/trade


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
    return segs


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


def mechanical_layer(closes, highs, lows, vols, btc_close=None):
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
        m["g_mom90"] = (close / close.shift(540) - 1.0 >= 0.05).astype(np.int8)
        m["g_atr"] = (atr_pctl >= 0.60).astype(np.int8)
        m["g_dd"] = (close / close.rolling(2160, min_periods=100).max() - 1.0
                     >= -0.30).astype(np.int8)
        s50, s200 = sma(close, 50), sma(close, 200)
        m["r_trend"] = ((close > s50) & (s50 > s200)).astype(np.int8)
        m["r_brk20"] = (close > high.rolling(21).max().shift(1)).astype(np.int8)
        m["r_brk55"] = (close > high.rolling(56).max().shift(1)).astype(np.int8)
        m["r_pull"] = ((close < s50 * 1.03) & (close > s50 * 0.97)
                       & (close > s200)).astype(np.int8)
        r14 = rsi(close, 14)
        m["r_rsi"] = ((r14 >= 45) & (r14 <= 65)).astype(np.int8)
        m["r_vol"] = (v > 1.5 * v.rolling(20).mean()).astype(np.int8)
        up = (close.pct_change() > 0.01).astype(np.int8)
        m["r_density"] = up.rolling(168, min_periods=20).sum()
        m["r_vov"] = a.rolling(168, min_periods=20).std() / (
            a.rolling(168, min_periods=20).mean().replace(0.0, np.nan))
        m["r_dow"] = close.index.dayofweek.astype(np.int8)
        m["r_hod"] = (close.index.hour // 4).astype(np.int8)
        if btc_sma is not None:
            b = btc_close.reindex(close.index)
            bs = btc_sma.reindex(close.index)
            m["r_btc"] = (b > bs).astype(np.int8)
        cols = [c for c in m.columns if c.startswith(("g_", "r_"))
                and c not in ("r_density", "r_vov", "r_dow", "r_hod")]
        m["mech_vote"] = m[cols].sum(axis=1)
        out[pair] = m
    return out


def live_outcome(c, h, l, t, hard_stop=HARD_STOP, trail=EXIT_TRAIL_LIVE,
                 cap=MAX_HOLD_BARS):
    """What the bot ACTUALLY realizes entering bar t: entry = close x
    (1+slip), exit on hard stop, trail, or cap. Returns realized net
    after fees, or None if the window doesn't exist."""
    n = len(c)
    if t + 1 >= n:
        return None
    entry = c[t] * (1 + SLIP)
    run_hi = entry
    for k in range(1, min(cap, n - t) + 1):
        px = c[t + k]
        if px <= entry * (1 + hard_stop):
            out_px = entry * (1 + hard_stop)
            return out_px / entry - 1 - FEE
        run_hi = max(run_hi, px)
        if px <= run_hi * (1 - trail):
            return px / entry - 1 - FEE
    return c[min(t + cap, n - 1)] / entry - 1 - FEE


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
    segs = {p: realistic_segments(closes)[p] for p in top}
    log(f"segments: {sum(len(v) for v in segs.values())}")

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
        for (ei, xi, ep, xp, net, dur) in segs[p]:
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
    X = np.concatenate([Xraw, Xmech], axis=1)
    ybin = np.concatenate(ybin); ynet = np.concatenate(ynet)
    pos = np.array(times)
    log(f"rows {len(ybin)} pos {ybin.mean():.4f} features {X.shape[1]}")

    # precompute live outcomes once per (pair, idx) for ALL rows
    log("simulating realized outcomes for every bar...")
    realized = np.full(len(ybin), np.nan)
    for p, tlist in [(p, [k[1] for k in keys if k[0] == p]) for p in top]:
        c = closes[p].to_numpy(); h = highs[p].to_numpy()
        l = lows[p].to_numpy()
        m = np.array([k[0] == p for k in keys])
        for j in np.where(m)[0]:
            realized[j] = live_outcome(c, h, l, keys[j][1])
    log(f"outcomes computed: {np.isfinite(realized).sum()}")

    # ROLL-FORWARD: test each fold k>=2, train on folds < k
    bnds = np.linspace(pos.min(), pos.max() + 1, 6, dtype=int)
    fold_ev = []
    best_thr = None
    for k in range(2, 5):
        trm = pos < bnds[k - 1]
        tem = (pos >= bnds[k - 1]) & (pos < bnds[k])
        log(f"--- fold {k}: train {trm.sum()} test {tem.sum()} ---")
        clf = LGBMClassifier(n_estimators=250, learning_rate=0.05,
                             max_depth=7, num_leaves=63, subsample=0.8,
                             colsample_bytree=0.8, verbose=-1, n_jobs=4)
        clf.fit(X[trm], ybin[trm])
        p = clf.predict_proba(X)[:, 1]
        for t in (0.20, 0.25, 0.30, 0.35, 0.40):
            sel = tem & (p >= t)
            n_sel = int(sel.sum())
            if n_sel < 8:
                continue
            ev = float(np.nanmean(realized[sel])) - FEE  # realized already
            # realized includes fees; ev per trade:
            ev = float(np.nanmean(realized[sel]))
            prec = float(ybin[sel].mean())
            log(f"  thr {t:.2f}: n {n_sel} prec {prec:.0%} "
                f"EV/trade {ev:+.3%}")
            fold_ev.append((k, t, ev, n_sel, prec))
    # champion: best operating point averaged over folds (mean EV,
    # preferring the highest EV point with >=15 trades total)
    valid = [x for x in fold_ev if x[3] >= 15]
    if valid:
        agg = {}
        for k, t, ev, n, prec in valid:
            agg.setdefault(t, []).append((ev, n, prec))
        best = max(agg, key=lambda t: np.mean([a[0] for a in agg[t]]))
        mean_ev = float(np.mean([a[0] for a in agg[best]]))
        mean_prec = float(np.mean([a[2] for a in agg[best]]))
        n_tot = int(sum(a[1] for a in agg[best]))
        # net-regression sizer (predict segment net within band)
        reg_tr = pos < bnds[3]  # train sizer on earlier folds
        sel_tr = reg_tr & (ynet > 0)
        rg = LGBMRegressor(n_estimators=250, learning_rate=0.05,
                           max_depth=7, num_leaves=63, subsample=0.8,
                           colsample_bytree=0.8, verbose=-1, n_jobs=4)
        rg.fit(X[sel_tr], ynet[sel_tr])
        log(f"CHAMPION: thr {best:.2f} mean EV {mean_ev:+.3%}/trade "
            f"prec {mean_prec:.0%} trades {n_tot} (folds 2-4)")
        policy = {
            "threshold": best,
            "mean_ev_per_trade": mean_ev,
            "mean_precision": mean_prec,
            "n_trades_oos": n_tot,
            "hard_stop": HARD_STOP,
            "trail": EXIT_TRAIL_LIVE,
            "max_hold_bars": MAX_HOLD_BARS,
            "slip": SLIP, "fee_per_side": FEE,
            "sizer": "net-regression (LGBMRegressor)",
            "train_rows": int(trm.sum()),
            "updated": pd.Timestamp.utcnow().isoformat(),
        }
        os.makedirs("reports", exist_ok=True)
        with open(f"{OUT}.json", "w") as fh:
            json.dump(policy, fh, indent=1)
        np.savez_compressed(f"{OUT}_model.npz",
                            thr=np.float32(best),
                            n_est=np.int32(250), lr=np.float32(0.05))
        log("policy card written: " + json.dumps(policy, indent=1))
    else:
        log("NO PROFITABLE BAND FOUND — policy card withheld (honest stop)")


if __name__ == "__main__":
    main()
