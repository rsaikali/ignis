"""Tests for backend.registry (pure filesystem, temp dir)."""

from __future__ import annotations

import json
from pathlib import Path

from ignis.backend.registry import discover, get


def _write_model(d: Path, name: str, report=None, comparison=None, scalers=False):
    (d / f"{name}.keras").write_text("fake-keras")
    if report is not None:
        (d / f"{name}.report.json").write_text(json.dumps(report))
    if comparison is not None:
        (d / f"{name}.comparison.json").write_text(json.dumps(comparison))
    if scalers:
        (d / f"{name}.scalers.pkl").write_bytes(b"fake")


def test_empty_dir(tmp_path):
    assert discover(tmp_path) == []


def test_missing_dir():
    assert discover(Path("/nonexistent/xyz")) == []


def test_discovers_models_newest_first(tmp_path):
    _write_model(tmp_path, "ignis_gru_20260101T000000")
    _write_model(tmp_path, "ignis_gru_20260601T000000")
    names = [e.name for e in discover(tmp_path)]
    assert names == ["ignis_gru_20260601T000000", "ignis_gru_20260101T000000"]


def test_pairs_artifacts(tmp_path):
    _write_model(
        tmp_path,
        "m1",
        report={"period_end": "2026-06-01T00:00:00+00:00", "appliances": {}},
        comparison={"passed": ["four"], "failed": ["pc"]},
        scalers=True,
    )
    e = get("m1", tmp_path)
    assert e is not None
    assert e.trained_at == "2026-06-01T00:00:00+00:00"
    assert e.gate_passed == ["four"]
    assert e.gate_failed == ["pc"]
    assert e.predictable is True


def test_predictable_false_without_scalers(tmp_path):
    _write_model(tmp_path, "m2", report={"appliances": {}})
    e = get("m2", tmp_path)
    assert e is not None
    assert e.predictable is False
    assert e.comparison is None


def test_corrupt_json_tolerated(tmp_path):
    (tmp_path / "m3.keras").write_text("k")
    (tmp_path / "m3.report.json").write_text("{not json")
    e = get("m3", tmp_path)
    assert e is not None
    assert e.report is None
