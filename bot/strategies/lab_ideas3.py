"""RatchetRider: gen-4 — the exit-policy fix.

The controlled exit experiment (sim-lab, Aug 2026) took every
swing_rider entry inside a real >=8% swing and replayed alternative
exit policies on the SAME entries:

    current 4.5ATR chandelier:  avg -1.85%, win 21%, total -235%
    BREAKEVEN RATCHET + 3ATR:   avg +1.03%, win 68%, total +131%

The exit policy alone flips the bot from loser to winner. The ratchet:
once profit exceeds the round-trip cost, the stop moves to breakeven+-
a hair — a winner can never become a loser — and then trails at a
tighter 3xATR chandelier. Path-dependent by design (the stop depends
on the position's own history), computed in a causal forward walk.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bot.indicators.ta import atr
from bot.strategies.base import Strategy


class RatchetRider(Strategy):
    name = "ratchet_rider"
    DEFAULTS = {"surge_pct": 0.05, "surge_bars": 12,
                "atr_period": 14, "arm_rtc_mult": 1.0,
                "trail_mult": 3.0, "cooldown": 6,
                "vol_ok_pctile": 0.30}
    # rtc read from the synced fee model at decision time; import
    # lazily so the module stays importable standalone

    def warmup_bars(self) -> int:
        return 200

    def compute_signals(self, df: pd.DataFrame, live: bool = False) -> pd.Series:
        from bot.trade_gate import round_trip_cost
        rtc = round_trip_cost()
        p = {**self.DEFAULTS, **self.params}
        close = df["close"].values
        high = df["high"].values
        n = len(df)
        a = atr(df["high"], df["low"], df["close"], int(p["atr_period"])) \
            .values / close
        a_pctile = pd.Series(a).rolling(540, min_periods=120).apply(
            lambda v: (v <= v[-1]).mean(), raw=True).fillna(0.5).values

        arm_px = float(p["arm_rtc_mult"]) * rtc
        trail = float(p["trail_mult"])

        sig = np.zeros(n, dtype=int)
        in_pos = False
        stop = 0.0
        armed = False
        entry_px = 0.0
        cooldown = 0
        # rolling 50-bar high via a simple expanding scan per bar (n<=5k)
        hi50 = pd.Series(high).rolling(50, min_periods=10).max().values

        for i in range(n):
            if cooldown > 0:
                cooldown -= 1
            px = close[i]
            if in_pos:
                # chandelier trail (never loosens)
                cand = hi50[i] - trail * a[i]
                if cand > stop:
                    stop = cand
                if armed and stop < entry_px * (1.0 + arm_px * 1.05):
                    stop = entry_px * (1.0 + arm_px * 1.05)
                if not armed and px >= entry_px * (1.0 + arm_px):
                    armed = True
                    stop = max(stop, entry_px * (1.0 + arm_px * 1.05))
                if px < stop:
                    sig[i] = -1
                    in_pos = False
                    cooldown = int(p["cooldown"])
            else:
                if i >= int(p["surge_bars"]) and cooldown == 0:
                    surge = px / close[i - int(p["surge_bars"])] - 1.0
                    ignite = (surge >= float(p["surge_pct"])) and \
                             (a_pctile[i] >= float(p["vol_ok_pctile"]))
                    if ignite:
                        sig[i] = 1
                        in_pos = True
                        armed = False
                        entry_px = px
                        stop = px - trail * a[i]
        return pd.Series(sig, index=df.index, dtype=int)
