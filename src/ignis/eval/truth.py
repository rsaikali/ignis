"""Load per-appliance ON/OFF truth aligned to the common grid.

Reads the ``appliance_onoff`` view (Meross switch -> bool) and forward-fills it
onto the same grid as ``training.source.load_aligned``, so it lines up tick for
tick with predictions and the power-truth series.
"""

from __future__ import annotations

from datetime import datetime

import psycopg
from loguru import logger

from ignis.nilm.config import settings

_AGG_TICKS = """
SELECT DISTINCT (FLOOR(EXTRACT(EPOCH FROM time) / %(step)s) * %(step)s) AS tick
FROM linky_realtime
WHERE time >= %(start)s AND time < %(end)s
ORDER BY tick;
"""

_ONOFF = """
SELECT EXTRACT(EPOCH FROM time) AS ts, is_on
FROM appliance_onoff
WHERE appliance = %(app)s AND time >= %(start)s AND time < %(end)s
ORDER BY time;
"""


def load_truth_onoff(start: datetime, end: datetime, step: int) -> dict[str, list[bool]]:
    """Forward-fill switch ON/OFF truth onto the aggregate grid, per appliance.

    Returns ``{appliance: [bool per tick]}`` aligned to the same ordered ticks
    that ``load_aligned`` produces for the aggregate.
    """
    params = {"start": start, "end": end, "step": step}
    out: dict[str, list[bool]] = {}
    with psycopg.connect(settings.database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(_AGG_TICKS, params)
            ticks = [float(r[0]) for r in cur.fetchall()]
            for app in settings.nilm_appliances:
                cur.execute(_ONOFF, {**params, "app": app})
                changes = [(float(ts), bool(v)) for ts, v in cur.fetchall()]
                out[app] = _forward_fill_bool(changes, ticks)
    logger.info("Loaded ON/OFF truth for {} appliances over {} ticks", len(out), len(ticks))
    return out


def _forward_fill_bool(changes: list[tuple[float, bool]], ticks: list[float]) -> list[bool]:
    """Hold the last switch state at/under each tick; default OFF before first."""
    out: list[bool] = []
    i = 0
    n = len(changes)
    last = False
    for tick in ticks:
        while i < n and changes[i][0] <= tick:
            last = changes[i][1]
            i += 1
        out.append(last)
    return out
