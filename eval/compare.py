"""HA-vs-NILM comparison: disaggregate a holdout, score against HA truth.

Loads a trained model (+ its persisted scalers), predicts per-appliance power
over a holdout period, aligns predictions with the Meross truth on the common
grid, and scores each appliance with eval.metrics (state F1 + energy error)
against the acceptance gate.

numpy / the engine model are imported lazily (engine extra), so the rest of
``eval`` stays importable in the dev venv. The aggregation of metrics into a
report is pure (``build_comparison``) and unit-tested.
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from loguru import logger

from nilm.config import settings

from .metrics import ApplianceMetrics, evaluate_appliance


@dataclass
class ComparisonReport:
    """Per-appliance HA-vs-NILM scores for one holdout window."""

    model_name: str
    period_start: str
    period_end: str
    grid_seconds: int
    n_ticks: int
    threshold: float
    appliances: dict[str, dict] = field(default_factory=dict)
    passed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def build_comparison(
    model_name: str,
    start: datetime,
    end: datetime,
    grid_seconds: int,
    threshold: float,
    per_appliance: Mapping[str, ApplianceMetrics],
) -> ComparisonReport:
    """Assemble a comparison report from per-appliance metrics (pure)."""
    appliances: dict[str, dict] = {}
    passed: list[str] = []
    failed: list[str] = []
    n_ticks = 0
    for app, m in per_appliance.items():
        n_ticks = max(n_ticks, m.n_samples)
        appliances[app] = {
            "state_f1": round(m.state_f1, 4),
            "energy_error": round(m.energy_error, 4) if m.energy_error != float("inf") else None,
            "n_samples": m.n_samples,
            "passes_gate": m.passes_gate,
        }
        (passed if m.passes_gate else failed).append(app)
    return ComparisonReport(
        model_name=model_name,
        period_start=start.isoformat(),
        period_end=end.isoformat(),
        grid_seconds=grid_seconds,
        n_ticks=n_ticks,
        threshold=threshold,
        appliances=appliances,
        passed=sorted(passed),
        failed=sorted(failed),
    )


def _load_scalers(model_path: Path):
    sidecar = model_path.with_suffix(".scalers.pkl")
    if not sidecar.exists():
        raise FileNotFoundError(f"Scalers sidecar missing: {sidecar}")
    with sidecar.open("rb") as f:
        return pickle.load(f)


def compare(model_path: Path, start: datetime, end: datetime, stride: int = 1) -> ComparisonReport:
    """Disaggregate [start, end) and score predictions vs HA truth."""
    import numpy as np

    from nilm.nilm.models import Seq2PointMultiOutputModel
    from training.source import load_aligned

    from .truth import load_truth_onoff

    threshold = settings.nilm_min_power_threshold
    grid = settings.ingest_grid_seconds

    dataset = load_aligned(start, end, step=grid)
    if len(dataset) == 0:
        raise SystemExit("No aligned data in holdout window.")

    model = Seq2PointMultiOutputModel(
        appliance_ids=[], appliance_names=[], sequence_length=settings.effective_sequence_length
    )
    model.load(str(model_path))
    scalers = _load_scalers(model_path)
    model.preprocessor.input_scaler = scalers["input_scaler"]
    model.preprocessor.target_scaler = scalers["target_scaler"]
    model.preprocessor.fitted = True

    agg = np.asarray(dataset.aggregate, dtype=np.float32)
    preds = model.predict(agg, stride=stride)  # {appliance_id: signal aligned to agg}

    onoff = load_truth_onoff(start, end, grid)  # {app: [bool, ...]} aligned to grid

    per_appliance: dict[str, ApplianceMetrics] = {}
    for idx, app in enumerate(model.appliance_names):
        pred_w = list(preds[model.appliance_ids[idx]])
        truth_w = dataset.appliances.get(app, [])
        truth_on = onoff.get(app)
        per_appliance[app] = evaluate_appliance(app, truth_w, pred_w, threshold, truth_on=truth_on)

    report = build_comparison(model_path.stem, start, end, grid, threshold, per_appliance)
    _write(model_path, report)
    logger.info("Comparison: {} passed, {} failed gate", len(report.passed), len(report.failed))
    return report


def _write(model_path: Path, report: ComparisonReport) -> None:
    out = model_path.with_suffix(".comparison.json")
    out.write_text(json.dumps(report.to_dict(), indent=2))


def gate_summary(report: ComparisonReport) -> str:
    """One-line human summary of the acceptance gate outcome."""
    total = len(report.appliances)
    return f"{len(report.passed)}/{total} appliances pass gate (F1>=0.8, energy_err<=15%)"
