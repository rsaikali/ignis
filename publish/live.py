"""Live publisher: champion inference -> MQTT (spec 6.4) + scores.

Loads the champion model, pulls the most recent aggregate window from the lab
DB, disaggregates the latest point, and publishes:
- ``nilm/disaggregation`` -- the grouped per-appliance snapshot (portfolio + subs)
- ``nilm/_meta/model``     -- model version + the honest F1-vs-HA scores read
                              from the champion's .comparison.json

numpy / the engine model are imported lazily (engine extra). The honest-scores
assembly (``meta_from_artifacts``) is pure and tested; inference is a thin shell.
"""

from __future__ import annotations

import json
import pickle
from datetime import UTC, datetime, timedelta
from pathlib import Path

from loguru import logger

from nilm.config import settings

from .contract import DisaggregationSnapshot, ModelMeta


def meta_from_artifacts(report: dict, comparison: dict | None) -> ModelMeta:
    """Build the model-meta payload from a model's report + comparison (pure).

    Metrics carry the honest per-appliance state F1 and energy error vs HA, so
    the portfolio can show exactly where each appliance stands.
    """
    metrics: dict[str, dict[str, float]] = {}
    if comparison:
        for app, m in comparison.get("appliances", {}).items():
            metrics[app] = {
                "state_f1": float(m.get("state_f1", 0.0)),
                "energy_error": float(m.get("energy_error") or 0.0),
                "passes_gate": 1.0 if m.get("passes_gate") else 0.0,
            }
    trained = report.get("period_end") or datetime.now(UTC).isoformat()
    return ModelMeta(
        version=report.get("model_name", "champion"),
        model_type=report.get("model_type", "gru"),
        trained_at=datetime.fromisoformat(trained),
        appliances=list(report.get("appliances", {})),
        metrics=metrics,
    )


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _latest_snapshot(model_path: Path, window_minutes: int) -> DisaggregationSnapshot:
    """Disaggregate the most recent window; snapshot its centre point."""
    import numpy as np

    from nilm.nilm.models import Seq2PointMultiOutputModel
    from training.source import load_aligned

    end = datetime.now(UTC)
    start = end - timedelta(minutes=window_minutes)
    ds = load_aligned(start, end, step=settings.ingest_grid_seconds)
    if len(ds) == 0:
        raise SystemExit("No recent aggregate data to disaggregate.")

    model = Seq2PointMultiOutputModel(
        appliance_ids=[], appliance_names=[], sequence_length=settings.effective_sequence_length
    )
    model.load(str(model_path))
    sc = pickle.load(model_path.with_suffix(".scalers.pkl").open("rb"))
    model.preprocessor.input_scaler = sc["input_scaler"]
    model.preprocessor.target_scaler = sc["target_scaler"]
    model.preprocessor.fitted = True

    agg = np.asarray(ds.aggregate, dtype=np.float32)
    preds = model.predict(agg, stride=1)

    last = len(agg) - 1
    appliances = {app: float(preds[model.appliance_ids[i]][last]) for i, app in enumerate(model.appliance_names)}
    return DisaggregationSnapshot(ts=end, total_w=float(agg[last]), appliances=appliances)


async def publish_once(window_minutes: int = 60) -> None:
    """Single inference + publish cycle (snapshot + meta)."""
    import aiomqtt

    from .mqtt_publisher import MqttPublisher, make_client

    model_path = Path(settings.nilm_model_path) / "champion.keras"
    if not model_path.exists():
        raise SystemExit(f"No champion model at {model_path}. Run make eval to promote one.")

    snapshot = _latest_snapshot(model_path, window_minutes)
    report = _load_json(model_path.with_suffix(".report.json")) or {}
    comparison = _load_json(model_path.with_suffix(".comparison.json"))
    meta = meta_from_artifacts(report, comparison)

    client: aiomqtt.Client = make_client()
    async with client:
        pub = MqttPublisher(client)
        await pub.publish_snapshot(snapshot)
        await pub.publish_meta(meta)
    logger.info("Published snapshot ({} appliances) + meta {}", len(snapshot.appliances), meta.version)
