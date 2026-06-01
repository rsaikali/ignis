"""Tests for publish.live.meta_from_artifacts (pure; no model/DB/MQTT)."""

from __future__ import annotations

from publish.live import meta_from_artifacts

REPORT = {
    "model_name": "ignis_gru_20260601T131335",
    "model_type": "gru",
    "period_end": "2026-06-01T13:13:35+00:00",
    "appliances": {"four": {}, "television": {}, "pc": {}},
}


def test_meta_carries_honest_scores():
    comparison = {
        "appliances": {
            "television": {"state_f1": 0.701, "energy_error": 0.82, "passes_gate": False},
            "four": {"state_f1": 0.0, "energy_error": 0.94, "passes_gate": False},
        }
    }
    meta = meta_from_artifacts(REPORT, comparison)
    assert meta.version == "ignis_gru_20260601T131335"
    assert meta.model_type == "gru"
    assert meta.metrics["television"]["state_f1"] == 0.701
    assert meta.metrics["four"]["passes_gate"] == 0.0
    # Payload is JSON-serialisable and rounds metrics.
    payload = meta.payload()
    assert payload["metrics"]["television"]["state_f1"] == 0.701


def test_meta_without_comparison_has_empty_metrics():
    meta = meta_from_artifacts(REPORT, None)
    assert meta.metrics == {}
    assert meta.appliances == ["four", "television", "pc"]


def test_meta_handles_none_energy_error():
    comparison = {"appliances": {"lave_linge": {"state_f1": 0.0, "energy_error": None, "passes_gate": False}}}
    meta = meta_from_artifacts(REPORT, comparison)
    # None energy error coerced to 0.0 (JSON-safe).
    assert meta.metrics["lave_linge"]["energy_error"] == 0.0


def test_meta_payload_json_serialisable():
    import json

    meta = meta_from_artifacts(
        REPORT, {"appliances": {"pc": {"state_f1": 0.4, "energy_error": 4.1, "passes_gate": False}}}
    )
    json.dumps(meta.payload())
