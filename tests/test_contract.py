"""Tests pinning the portfolio MQTT contract (spec 6.4, Option B) shape."""

from __future__ import annotations

from datetime import UTC, datetime

from publish.contract import (
    DISAGGREGATION_TOPIC,
    META_TOPIC,
    DisaggregationSnapshot,
    ModelMeta,
    iso_utc,
)

TS = datetime(2026, 5, 31, 18, 0, 0, tzinfo=UTC)


def test_topics_match_spec():
    assert DISAGGREGATION_TOPIC == "nilm/disaggregation"
    assert META_TOPIC == "nilm/_meta/model"


def test_iso_utc_has_z_suffix():
    assert iso_utc(TS) == "2026-05-31T18:00:00Z"


def test_iso_utc_assumes_naive_is_utc():
    naive = datetime(2026, 5, 31, 18, 0, 0)
    assert iso_utc(naive) == "2026-05-31T18:00:00Z"


def test_snapshot_payload_matches_6_4_exactly():
    snap = DisaggregationSnapshot(
        ts=TS,
        total_w=2310.4,
        appliances={"four": 1245.0, "pc": 0.0},
    )
    assert snap.payload() == {
        "ts": "2026-05-31T18:00:00Z",
        "total_w": 2310.4,
        "appliances": {"four": 1245.0, "pc": 0.0},
    }


def test_snapshot_rounds_powers():
    snap = DisaggregationSnapshot(ts=TS, total_w=2310.456, appliances={"four": 1245.678})
    payload = snap.payload()
    assert payload["total_w"] == 2310.5
    assert payload["appliances"]["four"] == 1245.7


def test_meta_payload_rounds_metrics():
    m = ModelMeta(
        version="2026-05-31T12:00:00Z",
        model_type="gru",
        trained_at=datetime(2026, 5, 31, 12, 0, 0, tzinfo=UTC),
        appliances=["four", "pc"],
        metrics={"four": {"state_f1": 0.8765, "energy_error": 0.0912}},
    )
    payload = m.payload()
    assert payload["appliances"] == ["four", "pc"]
    assert payload["trained_at"] == "2026-05-31T12:00:00Z"
    assert payload["metrics"]["four"] == {"state_f1": 0.876, "energy_error": 0.091}
