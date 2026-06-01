"""Load aligned training data from the lab TimescaleDB (sync, for native train).

Reads the compatibility views (created by ha_ingest):
- ``linky_realtime(time, papp)``    -> aggregate samples
- ``appliance_power(time, appliance, power_w)`` -> per-appliance truth

Uses a plain psycopg connection (training runs native, not async). Returns the
raw ``(epoch, value)`` sample lists; alignment is done by ``dataset``.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import psycopg
from loguru import logger

from ignis.nilm.config import settings

from .dataset import AlignedDataset, Sample, build_aligned

_AGG_SQL = """
SELECT EXTRACT(EPOCH FROM time) AS ts, papp
FROM linky_realtime
WHERE time >= %(start)s AND time < %(end)s
ORDER BY time;
"""

_APP_SQL = """
SELECT EXTRACT(EPOCH FROM time) AS ts, power_w
FROM appliance_power
WHERE appliance = %(app)s AND time >= %(start)s AND time < %(end)s
ORDER BY time;
"""


def _fetch(conn: psycopg.Connection, sql: str, params: dict) -> list[Sample]:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return [(float(ts), float(v)) for ts, v in cur.fetchall() if v is not None]


def load_aligned(
    start: datetime,
    end: datetime,
    appliances: Sequence[str] | None = None,
    step: int | None = None,
) -> AlignedDataset:
    """Load + align the aggregate and per-appliance truth over ``[start, end)``."""
    apps = list(appliances) if appliances is not None else list(settings.nilm_appliances)
    grid_step = step if step is not None else settings.ingest_grid_seconds
    params = {"start": start, "end": end}

    with psycopg.connect(settings.database_url) as conn:
        aggregate = _fetch(conn, _AGG_SQL, params)
        appliance_power = {app: _fetch(conn, _APP_SQL, {**params, "app": app}) for app in apps}

    dataset = build_aligned(aggregate, appliance_power, step=grid_step)
    logger.info(
        "Loaded aligned dataset: {} ticks, {} appliances, step={}s, {:.1f}h",
        len(dataset),
        len(apps),
        grid_step,
        len(dataset) * grid_step / 3600.0,
    )
    return dataset
