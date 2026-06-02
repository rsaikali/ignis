"""DB-backed reads for the portfolio API (history + live truth).

psycopg is imported lazily so the rest of backend stays importable without the
engine extra. Query strings are module-level; the row-shaping is small and
pure-ish (takes fetched rows, returns JSON-ready dicts).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from ignis.nilm.config import settings

_HISTORY_SQL = """
SELECT logged_at, model, trained_at, train_days, mean_f1, gate_passes, promoted, appliances
FROM model_runs
ORDER BY logged_at ASC
LIMIT %(limit)s;
"""

# Latest value per appliance within the window, from the power truth view.
_TRUTH_POWER_SQL = """
SELECT DISTINCT ON (appliance) appliance, power_w
FROM appliance_power
WHERE time >= %(since)s
ORDER BY appliance, time DESC;
"""


def models_history(limit: int = 500) -> list[dict]:
    """Return one row per logged retrain (oldest -> newest) for the curve."""
    import psycopg

    with psycopg.connect(settings.database_url) as conn:
        with conn.cursor() as cur:
            # Tolerate a missing table (no retrain has run yet).
            cur.execute("SELECT to_regclass('public.model_runs');")
            if cur.fetchone()[0] is None:
                return []
            cur.execute(_HISTORY_SQL, {"limit": limit})
            rows = cur.fetchall()
    out: list[dict] = []
    for logged_at, model, trained_at, train_days, mean_f1, gate_passes, promoted, appliances in rows:
        out.append(
            {
                "logged_at": logged_at.isoformat() if logged_at else None,
                "model": model,
                "trained_at": trained_at.isoformat() if trained_at else None,
                "train_days": train_days,
                "mean_f1": mean_f1,
                "gate_passes": gate_passes,
                "promoted": promoted,
                "appliances": appliances,  # JSONB -> already a dict
            }
        )
    return out


def latest_disaggregation() -> dict | None:
    """The most recent disaggregation snapshot + meta (HTTP surface A).

    publish upserts a single row; returns ``{updated_at, snapshot, meta}`` or
    None if no cycle has run / the table doesn't exist yet.
    """
    import psycopg

    with psycopg.connect(settings.database_url) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass('public.disaggregation_latest');")
            if cur.fetchone()[0] is None:
                return None
            cur.execute("SELECT updated_at, snapshot, meta FROM disaggregation_latest WHERE id = 1;")
            row = cur.fetchone()
    if row is None:
        return None
    updated_at, snapshot, meta = row
    return {
        "updated_at": updated_at.isoformat() if updated_at else None,
        "snapshot": snapshot,  # JSONB -> dict
        "meta": meta,
    }


def truth_recent(window_seconds: int) -> dict:
    """Latest per-appliance HA truth (power + on/off) within the window.

    ON/OFF is derived from measured power vs the NILM threshold, not the Meross
    switch entity (the plugs stay powered, used only to meter).
    """
    import psycopg

    since = datetime.now(UTC) - timedelta(seconds=window_seconds)
    threshold = settings.nilm_min_power_threshold
    power: dict[str, float] = {}
    with psycopg.connect(settings.database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(_TRUTH_POWER_SQL, {"since": since})
            power = {app: float(w) for app, w in cur.fetchall()}
    appliances = {app: {"power_w": w, "on": w > threshold} for app, w in power.items()}
    return {"window_seconds": window_seconds, "appliances": appliances}
