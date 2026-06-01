"""Persist each retrain run to a model_runs table (evolution history).

The dated .keras / .comparison.json sidecars are pruned (keep 5), but the
portfolio's evolution curve needs the full series of scores over time. This
logs one compact row per retrain -- no model weights, just the numbers -- so
history survives pruning.

Pure SQL strings + a thin psycopg writer (imported lazily). The row-building
(``row_from_comparison``) is pure and tested.
"""

from __future__ import annotations

from datetime import UTC, datetime

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS model_runs (
    logged_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    model        TEXT        NOT NULL,
    trained_at   TIMESTAMPTZ,
    train_days   INTEGER,
    mean_f1      DOUBLE PRECISION,
    gate_passes  INTEGER,
    promoted     BOOLEAN,
    appliances   JSONB
);
"""

_CREATE_INDEX = "CREATE INDEX IF NOT EXISTS model_runs_logged_at ON model_runs (logged_at DESC);"

_INSERT = """
INSERT INTO model_runs (model, trained_at, train_days, mean_f1, gate_passes, promoted, appliances)
VALUES (%(model)s, %(trained_at)s, %(train_days)s, %(mean_f1)s, %(gate_passes)s, %(promoted)s, %(appliances)s);
"""


def row_from_comparison(comparison: dict, train_days: int, promoted: bool) -> dict:
    """Build a compact model_runs row from a ComparisonReport dict (pure).

    Keeps per-appliance state_f1 / energy_error / passes_gate; drops sample
    counts. mean_f1 / gate_passes are derived here so the API needn't recompute.
    """
    apps = comparison.get("appliances", {})
    per = {
        name: {
            "state_f1": float(m.get("state_f1", 0.0)),
            "energy_error": (None if m.get("energy_error") is None else float(m["energy_error"])),
            "passes_gate": bool(m.get("passes_gate")),
        }
        for name, m in apps.items()
    }
    f1s = [v["state_f1"] for v in per.values()]
    return {
        "model": comparison.get("model_name", "unknown"),
        "trained_at": comparison.get("period_end"),
        "train_days": train_days,
        "mean_f1": (sum(f1s) / len(f1s) if f1s else 0.0),
        "gate_passes": sum(1 for v in per.values() if v["passes_gate"]),
        "promoted": promoted,
        "appliances": per,
    }


def log_run(comparison: dict, train_days: int, promoted: bool) -> None:
    """Write one retrain run to model_runs (creates the table if needed)."""
    import json

    import psycopg
    from loguru import logger

    from ignis.nilm.config import settings

    row = row_from_comparison(comparison, train_days, promoted)
    row["appliances"] = json.dumps(row["appliances"])
    row.setdefault("logged_at", datetime.now(UTC))
    with psycopg.connect(settings.database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(_CREATE_TABLE)
            cur.execute(_CREATE_INDEX)
            cur.execute(_INSERT, row)
        conn.commit()
    logger.info("Logged run {} (mean_f1={:.3f}, promoted={})", row["model"], row["mean_f1"], promoted)
