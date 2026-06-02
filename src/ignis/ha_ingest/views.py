"""Compatibility SQL views over ``ha_samples``.

These views adapt the narrow ``ha_samples`` table to the shapes other modules
expect, so neither the harvested engine nor ``eval`` has to know about the
ingestion schema:

- ``linky_realtime(time, papp)`` -- the aggregate feed the engine reads verbatim
  via ``get_consumption_data`` (``time_bucket(interval, time), AVG(papp)``).
  ``papp`` is apparent power (VA), exactly the Linky aggregate.
- ``appliance_power(time, appliance, power_w)`` -- per-appliance ground truth for
  ``eval`` and ``training``: the Meross ``power_w`` rows. ON/OFF is derived from
  this power, not the switch entity (the plugs stay powered, used only to meter).

All idempotent (``CREATE OR REPLACE VIEW``); created at ingest startup.
"""

from __future__ import annotations

import asyncpg
from loguru import logger

# Engine aggregate feed. Matches get_consumption_data()'s linky_realtime(time, papp).
_LINKY_REALTIME = """
CREATE OR REPLACE VIEW linky_realtime AS
SELECT ts AS time, value AS papp
FROM ha_samples
WHERE kind = 'aggregate' AND value IS NOT NULL;
"""

# Per-appliance power truth (Meross power_w rows).
_APPLIANCE_POWER = """
CREATE OR REPLACE VIEW appliance_power AS
SELECT ts AS time, appliance, value AS power_w
FROM ha_samples
WHERE kind = 'power_w' AND appliance IS NOT NULL AND value IS NOT NULL;
"""

_VIEWS = (_LINKY_REALTIME, _APPLIANCE_POWER)

# Views retired in the switch-truth removal; dropped if a prior deploy created them.
_DROPPED_VIEWS = ("appliance_onoff",)


async def ensure_views(conn: asyncpg.Connection) -> None:
    """Create / replace all compatibility views (idempotent)."""
    for view in _DROPPED_VIEWS:
        await conn.execute(f"DROP VIEW IF EXISTS {view};")
    for ddl in _VIEWS:
        await conn.execute(ddl)
    logger.info("Compatibility views ensured: linky_realtime, appliance_power")
