"""Tests for eval.runlog.row_from_comparison (pure; no DB)."""

from __future__ import annotations

from ignis.eval.runlog import row_from_comparison

COMPARISON = {
    "model_name": "ignis_gru_20260601T131335",
    "period_end": "2026-06-01T13:13:35+00:00",
    "appliances": {
        "television": {"state_f1": 0.701, "energy_error": 0.82, "passes_gate": False},
        "pc": {"state_f1": 0.416, "energy_error": 4.16, "passes_gate": False},
        "four": {"state_f1": 0.0, "energy_error": 0.94, "passes_gate": False},
    },
}


def test_row_keeps_per_appliance_scores():
    row = row_from_comparison(COMPARISON, train_days=30, promoted=False)
    assert row["model"] == "ignis_gru_20260601T131335"
    assert row["train_days"] == 30
    assert row["promoted"] is False
    assert row["appliances"]["television"]["state_f1"] == 0.701
    assert row["appliances"]["four"]["passes_gate"] is False


def test_mean_f1_and_gate_passes_derived():
    row = row_from_comparison(COMPARISON, train_days=30, promoted=True)
    # mean of 0.701, 0.416, 0.0
    assert abs(row["mean_f1"] - (0.701 + 0.416) / 3) < 1e-9
    assert row["gate_passes"] == 0


def test_gate_passes_counts_true():
    cmp = {
        "model_name": "m",
        "period_end": "2026-06-01T00:00:00+00:00",
        "appliances": {
            "a": {"state_f1": 0.9, "energy_error": 0.1, "passes_gate": True},
            "b": {"state_f1": 0.5, "energy_error": 0.2, "passes_gate": False},
        },
    }
    row = row_from_comparison(cmp, train_days=14, promoted=True)
    assert row["gate_passes"] == 1


def test_none_energy_error_preserved():
    cmp = {
        "model_name": "m",
        "period_end": "2026-06-01T00:00:00+00:00",
        "appliances": {"lave_linge": {"state_f1": 0.0, "energy_error": None, "passes_gate": False}},
    }
    row = row_from_comparison(cmp, train_days=30, promoted=False)
    assert row["appliances"]["lave_linge"]["energy_error"] is None


def test_empty_appliances_safe():
    row = row_from_comparison({"model_name": "m", "appliances": {}}, train_days=30, promoted=False)
    assert row["mean_f1"] == 0.0
    assert row["gate_passes"] == 0
