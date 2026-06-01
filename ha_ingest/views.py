"""Compatibility SQL views over ``ha_samples``.

These views adapt the narrow ``ha_samples`` table to the shapes other modules
expect, so neither the harvested engine nor ``eval`` has to know about the
ingestion schema:

- ``linky_realtime(time, papp)`` -- the aggregate feed the engine reads verbatim
  via ``get_consumption_data`` (``time_bucket(interval, time), AVG(papp)``).
  ``papp`` is apparent power (VA), exactly the Linky aggregate.
- ``appliance_truth(time, appliance, power_w, on_off)`` -- per-appliance ground
  truth for ``eval``: Meross ``power_w`` plus the switch state (1.0/0.0). The
  switch arrives on its own rows, so ``on_off`` is a separate LEFT-JOIN-able
  view rather than a column on the power rows.

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

# Per-appliance ON/OFF truth from the Meross switch entity (1.0/0.0).
_APPLIANCE_ONOFF = """
CREATE OR REPLACE VIEW appliance_onoff AS
SELECT ts AS time, appliance, (value > 0.5) AS is_on
FROM ha_samples
WHERE kind = 'switch' AND appliance IS NOT NULL AND value IS NOT NULL;
"""

_VIEWS = (_LINKY_REALTIME, _APPLIANCE_POWER, _APPLIANCE_ONOFF)


async def ensure_views(conn: asyncpg.Connection) -> None:
    """Create / replace all compatibility views (idempotent)."""
    for ddl in _VIEWS:
        await conn.execute(ddl)
    logger.info("Compatibility views ensured: linky_realtime, appliance_power, appliance_onoff")
