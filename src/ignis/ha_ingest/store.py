"""TimescaleDB sink for raw HA samples.

Narrow/long schema: one row per (entity, arrival). Full resolution is kept
here; ``eval`` buckets to the common grid via ``time_bucket`` at read time.
"""

from __future__ import annotations

import asyncpg
from loguru import logger

from ignis.nilm.config import settings

from .views import ensure_views

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS ha_samples (
    ts          TIMESTAMPTZ      NOT NULL DEFAULT now(),
    entity_id   TEXT             NOT NULL,
    appliance   TEXT,
    kind        TEXT             NOT NULL,
    value       DOUBLE PRECISION,
    raw         TEXT
);
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS ha_samples_entity_ts
    ON ha_samples (entity_id, ts DESC);
"""

# Best-effort hypertable; no-op / ignored if TimescaleDB is absent.
_CREATE_HYPERTABLE = "SELECT create_hypertable('ha_samples', 'ts', if_not_exists => TRUE);"

_INSERT = """
INSERT INTO ha_samples (entity_id, appliance, kind, value, raw)
VALUES ($1, $2, $3, $4, $5);
"""


class Store:
    """Thin asyncpg wrapper around the ha_samples hypertable."""

    def __init__(self) -> None:
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        dsn = settings.database_url  # plain libpq DSN works for asyncpg
        self._pool = await asyncpg.create_pool(dsn, min_size=1, max_size=4)
        await self._ensure_schema()
        logger.info("Store connected: {}:{}/{}", settings.local_db_host, settings.local_db_port, settings.local_db_name)

    async def _ensure_schema(self) -> None:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute(_CREATE_TABLE)
            await conn.execute(_CREATE_INDEX)
            try:
                await conn.execute(_CREATE_HYPERTABLE)
            except asyncpg.PostgresError as exc:
                logger.warning("Hypertable not created (TimescaleDB missing?): {}", exc)
            await ensure_views(conn)

    async def insert(self, entity_id: str, appliance: str | None, kind: str, value: float | None, raw: str) -> None:
        assert self._pool is not None
        await self._pool.execute(_INSERT, entity_id, appliance, kind, value, raw)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
