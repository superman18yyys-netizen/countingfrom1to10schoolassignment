"""Champion promotion: apply Deep Time's validated winners to the live bots.

state/champions.json is written by run_deep_train.py after each epoch.
A champion is only promotable when it beat mean buy & hold on the
VALIDATION GAUNTLET (data evolution never touched) by a margin, AFTER
fees, with enough closed trades. This module applies such champions to
a live population (the zoo): the bot keeps its identity but receives
the champion's proven tuning.

If the champion is not promotable (or the file is absent/stale), the
live bots are left exactly as they were -- never regress on an
unproven tuning.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import List, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHAMPIONS_PATH = os.path.join(BASE_DIR, "state", "champions.json")
MAX_AGE_DAYS = 14.0   # a champion older than this is stale -> ignored


def load_champion(path: Optional[str] = None) -> Optional[dict]:
    """Return the promotable champion entry, or None."""
    try:
        with open(path or CHAMPIONS_PATH, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception:  # noqa: BLE001
        return None
    if not payload.get("promotable"):
        return None
    champ = payload.get("champion") or {}
    if not champ.get("params"):
        return None
    try:
        updated = datetime.strptime(payload["updated_at"], "%Y-%m-%d %H:%M UTC") \
            .replace(tzinfo=timezone.utc)
    except Exception:  # noqa: BLE001
        return champ
    if (datetime.now(timezone.utc) - updated).days > MAX_AGE_DAYS:
        return None
    return champ


def apply_champion(population, path: Optional[str] = None) -> List[str]:
    """Apply the champion tuning to every live agent of the same strategy.

    Agents keep their identity/accounts; only the genome params (and the
    lazily-built strategy) are refreshed. Returns the bot ids updated.
    """
    champ = load_champion(path)
    if not champ:
        return []
    strategy, params = champ["strategy"], champ["params"]
    updated: List[str] = []
    for agent in population.agents:
        if agent.genome.strategy != strategy:
            continue
        if agent.genome.params == params:
            continue
        agent.genome.params = dict(params)
        agent._strategy = None   # rebuild lazily with the champion tuning
        updated.append(agent.genome.id)
    return updated
