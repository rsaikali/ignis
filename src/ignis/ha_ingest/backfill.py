"""Backfill the HA recorder history into ha_samples.

ha_ingest only captures live messages from startup. Training needs a longer
span, so this one-shot replays the HA recorder's raw history (kept ~30d) for
our whitelisted entities into ha_samples.

Mechanism: a small extractor runs INSIDE the HA container (sqlite3 access to
home-assistant_v2.db, same path the probes use), streaming NDJSON rows
``[entity_id, ts_epoch, raw_state]`` over SSH stdout. The local side maps each
to its EntitySpec, parses the value, and bulk-inserts with explicit ts.

Idempotent: for each entity we delete the covered [start, end) window before
inserting, so re-running replaces rather than duplicates.

Run native (engine extra for psycopg):
    .venv/bin/python -m ha_ingest.backfill --days 14
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime, timedelta

import psycopg
from loguru import logger

from ignis.nilm.config import settings

from .entities import build_specs, entity_index, parse_value

# Extractor executed inside the HA container. Reads the recorder DB and emits
# NDJSON [entity_id, ts_epoch, raw_state] for the given entity_ids since N days.
_REMOTE_EXTRACT = r"""
import sqlite3, sys, json, time
db = "/config/home-assistant_v2.db"
days = float(sys.argv[1])
entity_ids = sys.argv[2:]
since = time.time() - days * 86400
c = sqlite3.connect(db)
q = c.cursor()
for eid in entity_ids:
    row = q.execute("SELECT metadata_id FROM states_meta WHERE entity_id=?", (eid,)).fetchone()
    if not row:
        continue
    mid = row[0]
    for st, ts in q.execute(
        "SELECT state, last_updated_ts FROM states "
        "WHERE metadata_id=? AND last_updated_ts>=? ORDER BY last_updated_ts",
        (mid, since),
    ):
        if st is None:
            continue
        sys.stdout.write(json.dumps([eid, ts, st]) + "\n")
"""

_DELETE_WINDOW = "DELETE FROM ha_samples WHERE entity_id = %(eid)s AND ts >= %(start)s AND ts < %(end)s;"
_INSERT = "INSERT INTO ha_samples (ts, entity_id, appliance, kind, value, raw) VALUES (%s, %s, %s, %s, %s, %s);"


def _extract_rows(days: float) -> list[tuple[str, float, str]]:
    """Run the remote extractor over SSH; return [(entity_id, ts, raw)]."""
    entity_ids = [spec.entity_id for spec in build_specs()]
    target = f"{settings.ha_ssh_user}@{settings.ha_ssh_host}"  # the HA box
    cmd = [
        "ssh",
        "-o",
        "BatchMode=yes",
        target,
        "docker",
        "exec",
        "-i",
        "homeassistant",
        "python3",
        "-",
        str(days),
        *entity_ids,
    ]
    logger.info("Extracting {} entities, {} days from HA recorder on {}", len(entity_ids), days, target)
    proc = subprocess.run(cmd, input=_REMOTE_EXTRACT, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(f"Remote extract failed: {proc.stderr.strip()}")
    rows = [tuple(json.loads(line)) for line in proc.stdout.splitlines() if line.strip()]
    logger.info("Extracted {} raw state rows", len(rows))
    return rows  # type: ignore[return-value]


def backfill(days: float) -> dict[str, int]:
    """Replay HA recorder history into ha_samples. Returns rows per entity."""
    index = entity_index()
    rows = _extract_rows(days)

    end = datetime.now(UTC)
    start = end - timedelta(days=days)

    inserted: dict[str, int] = {}
    to_insert: list[tuple] = []
    for eid, ts, raw in rows:
        spec = index.get(eid)
        if spec is None:
            continue
        when = datetime.fromtimestamp(ts, tz=UTC)
        value = parse_value(spec, str(raw))
        to_insert.append((when, spec.entity_id, spec.appliance, spec.kind, value, str(raw)))
        inserted[eid] = inserted.get(eid, 0) + 1

    with psycopg.connect(settings.database_url) as conn:
        with conn.cursor() as cur:
            for spec in build_specs():
                cur.execute(_DELETE_WINDOW, {"eid": spec.entity_id, "start": start, "end": end})
            cur.executemany(_INSERT, to_insert)
        conn.commit()

    logger.info("Backfill inserted {} rows across {} entities", len(to_insert), len(inserted))
    return inserted


def main() -> None:
    ap = argparse.ArgumentParser(prog="ha_ingest.backfill")
    ap.add_argument("--days", type=float, default=14)
    args = ap.parse_args()
    counts = backfill(args.days)
    for eid, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        logger.info("  {:50s} {}", eid, n)


if __name__ == "__main__":
    main()
