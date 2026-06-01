"""Tests for eval.compare pure aggregation (no model/DB)."""

from __future__ import annotations

from datetime import UTC, datetime

from eval.compare import ComparisonReport, build_comparison, gate_summary
from eval.metrics import ApplianceMetrics

START = datetime(2026, 5, 30, tzinfo=UTC)
END = datetime(2026, 6, 1, tzinfo=UTC)


def _m(app, f1, err, n=100):
    return ApplianceMetrics(appliance=app, state_f1=f1, energy_error=err, n_samples=n)


def test_partitions_pass_and_fail():
    per = {
        "four": _m("four", 0.9, 0.10),  # pass
        "pc": _m("pc", 0.7, 0.10),  # fail F1
        "tv": _m("tv", 0.85, 0.30),  # fail energy
    }
    r = build_comparison("m", START, END, 30, 15.0, per)
    assert r.passed == ["four"]
    assert r.failed == ["pc", "tv"]
    assert r.appliances["four"]["passes_gate"] is True
    assert r.n_ticks == 100


def test_inf_energy_error_serialised_as_none():
    per = {"x": _m("x", 0.9, float("inf"))}
    r = build_comparison("m", START, END, 30, 15.0, per)
    assert r.appliances["x"]["energy_error"] is None


def test_to_dict_json_serialisable():
    import json

    per = {"four": _m("four", 0.9, 0.1)}
    r = build_comparison("m", START, END, 30, 15.0, per)
    json.dumps(r.to_dict())


def test_gate_summary_counts():
    per = {"a": _m("a", 0.9, 0.1), "b": _m("b", 0.5, 0.1)}
    r = build_comparison("m", START, END, 30, 15.0, per)
    assert gate_summary(r) == "1/2 appliances pass gate (F1>=0.8, energy_err<=15%)"


def test_empty_report_is_valid():
    r = ComparisonReport(
        model_name="m",
        period_start=START.isoformat(),
        period_end=END.isoformat(),
        grid_seconds=30,
        n_ticks=0,
        threshold=15.0,
    )
    assert r.passed == []
    assert r.failed == []
