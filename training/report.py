"""Train report: what data went into a model, per appliance (pure, no numpy).

Serialisable summary persisted next to the ``.keras`` artifact and surfaced by
the admin UI: period covered, grid, total/active labeled hours per appliance,
window counts, and (after fit) epochs + per-appliance MAE.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime

from .dataset import AlignedDataset


@dataclass
class TrainReport:
    """Everything we want to show about one training run."""

    model_name: str
    model_type: str
    period_start: str
    period_end: str
    grid_seconds: int
    sequence_length: int
    total_ticks: int
    labeled_hours: float
    appliances: dict[str, dict] = field(default_factory=dict)
    n_windows: int = 0
    epochs_trained: int | None = None
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def build_report(
    model_name: str,
    model_type: str,
    start: datetime,
    end: datetime,
    dataset: AlignedDataset,
    sequence_length: int,
    n_windows: int,
    threshold: float,
) -> TrainReport:
    """Assemble the pre-fit report from the aligned dataset."""
    cov = dataset.coverage(threshold)
    appliances = {
        app: {
            "labeled_hours": round(c.labeled_hours, 2),
            "active_hours": round(c.active_hours, 2),
            "active_ticks": c.active_ticks,
            "active_fraction": round(c.active_fraction, 4),
        }
        for app, c in cov.items()
    }
    return TrainReport(
        model_name=model_name,
        model_type=model_type,
        period_start=start.isoformat(),
        period_end=end.isoformat(),
        grid_seconds=dataset.step,
        sequence_length=sequence_length,
        total_ticks=len(dataset),
        labeled_hours=round(len(dataset) * dataset.step / 3600.0, 2),
        appliances=appliances,
        n_windows=n_windows,
    )


def attach_fit_metrics(report: TrainReport, epochs: int, metrics: Mapping[str, Mapping[str, float]]) -> TrainReport:
    """Fill in post-fit numbers (epochs, per-appliance MAE)."""
    report.epochs_trained = epochs
    report.metrics = {app: {k: round(v, 4) for k, v in m.items()} for app, m in metrics.items()}
    return report
