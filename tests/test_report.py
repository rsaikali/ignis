"""Tests for training.report (pure)."""

from __future__ import annotations

from datetime import UTC, datetime

from ignis.training.dataset import AlignedDataset
from ignis.training.report import attach_fit_metrics, build_report

START = datetime(2026, 5, 20, tzinfo=UTC)
END = datetime(2026, 6, 1, tzinfo=UTC)


def _dataset() -> AlignedDataset:
    return AlignedDataset(
        step=30,
        grid=[0.0, 30.0, 60.0, 90.0],
        aggregate=[100.0, 100.0, 100.0, 100.0],
        appliances={"four": [0.0, 50.0, 60.0, 0.0], "pc": [0.0, 0.0, 0.0, 0.0]},
    )


def test_report_captures_period_and_coverage():
    r = build_report("ignis_gru_x", "gru", START, END, _dataset(), sequence_length=599, n_windows=12, threshold=15)
    assert r.model_name == "ignis_gru_x"
    assert r.period_start == START.isoformat()
    assert r.total_ticks == 4
    assert r.n_windows == 12
    assert r.appliances["four"]["active_ticks"] == 2
    assert r.appliances["pc"]["active_ticks"] == 0


def test_attach_fit_metrics_rounds():
    r = build_report("m", "gru", START, END, _dataset(), sequence_length=599, n_windows=1, threshold=15)
    r = attach_fit_metrics(r, epochs=7, metrics={"four": {"train_mae": 0.12345, "val_mae": 0.2}})
    assert r.epochs_trained == 7
    assert r.metrics["four"]["train_mae"] == 0.1235


def test_to_dict_is_json_serialisable():
    import json

    r = build_report("m", "gru", START, END, _dataset(), sequence_length=599, n_windows=1, threshold=15)
    json.dumps(r.to_dict())  # must not raise
