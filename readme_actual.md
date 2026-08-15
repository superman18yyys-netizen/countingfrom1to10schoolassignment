# SimLab — the simulation twin of StockTradeBot

Two-repo system:

- **StockTradeBot (live repo)** — runs bots on real *current* market
  data on GitHub Actions (the live zoo).
- **SimLab (this repo)** — the offline laboratory. Races every bot
  (plus evolved param variants) on REAL *historical* windows, ranks
  them, and exports champion candidates (`reports/champions.json`)
  that the live repo can promote.

## Why split?

Time in the live zoo is expensive: one verdict per 6-hour window.
Here, a full tournament over 90 days of 4H candles runs in minutes on
a CI runner. Iterate here; promote winners there.

## Harness

- Same `bot/` engine as the live repo (copy-synced; chassis, fee gate,
  drought gate, regime allowlists, vol-target sizing — identical
  decisions in sim and live).
- `run_sim.py` — the tournament runner:
  - splits the range into N-day windows (default 7),
  - backtests every contestant on every window × pair, fees ON,
  - ranks by mean excess return (min 5 trades to be eligible),
  - writes `reports/tournament-*.md` (board), `tournament-*.json`
    (full data), and `champions.json` (promotion candidate).

## Usage

```bash
# local (needs candles in data/sim.db or network access)
python run_sim.py --start 2026-05-01 --end 2026-08-01 --window 7

# with 40 random evolved variants racing too
python run_sim.py --variants 40
```

CI (`simulate` workflow): weekly Monday run, or dispatch with inputs
(days_back / window / variants). Results commit to `reports/`.

## Promotion flow

1. Run a tournament here (long enough range to trust — 60d+).
2. Copy `reports/champions.json` into the live repo's
   `state/champions.json` (or let a future automation PR do it).
3. The live zoo applies the champion at its next window
   (see `bot/train/champions.py` in the live repo).

## Engine sync

`bot/` here is a copy of the live repo's engine at sync time. After
live-repo changes, re-copy `bot/`, `requirements-swarm.txt`, and
`strategies.yaml` so sim verdicts stay valid for live behavior.
