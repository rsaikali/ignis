"""Tests for the history/truth API routes (TestClient, dbapi monkeypatched)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ignis.backend.app import _parse_window, app


@pytest.mark.parametrize(
    "window,seconds",
    [("15m", 900), ("30s", 30), ("2h", 7200), ("bad", 900), ("", 900), ("10x", 900)],
)
def test_parse_window(window, seconds):
    assert _parse_window(window) == seconds


def test_history_route(monkeypatch):
    from ignis.backend import dbapi

    fake = [{"model": "ignis_gru_x", "mean_f1": 0.18, "gate_passes": 0, "promoted": True}]
    monkeypatch.setattr(dbapi, "models_history", lambda limit=500: fake)
    r = TestClient(app).get("/api/models/history")
    assert r.status_code == 200
    assert r.json()[0]["model"] == "ignis_gru_x"


def test_truth_route(monkeypatch):
    from ignis.backend import dbapi

    fake = {"window_seconds": 900, "appliances": {"television": {"power_w": 45.0, "on": True}}}
    monkeypatch.setattr(dbapi, "truth_recent", lambda secs: fake)
    r = TestClient(app).get("/api/truth/recent?window=15m")
    assert r.status_code == 200
    assert r.json()["appliances"]["television"]["on"] is True


def test_disaggregation_route(monkeypatch):
    from ignis.backend import dbapi

    latest = {
        "updated_at": "2026-06-01T17:00:00+00:00",
        "snapshot": {"ts": "2026-06-01T17:00:00Z", "total_w": 320.0, "appliances": {"television": 0.0}},
        "meta": {"version": "ignis_gru_x", "metrics": {"television": {"state_f1": 0.701}}},
    }
    monkeypatch.setattr(dbapi, "latest_disaggregation", lambda: latest)
    r = TestClient(app).get("/api/disaggregation")
    assert r.status_code == 200
    assert r.json()["snapshot"]["total_w"] == 320.0
    assert r.json()["meta"]["version"] == "ignis_gru_x"


def test_disaggregation_route_404_when_empty(monkeypatch):
    from ignis.backend import dbapi

    monkeypatch.setattr(dbapi, "latest_disaggregation", lambda: None)
    r = TestClient(app).get("/api/disaggregation")
    assert r.status_code == 404


def test_truth_route_default_window(monkeypatch):
    from ignis.backend import dbapi

    captured = {}

    def _fake(secs):
        captured["secs"] = secs
        return {"window_seconds": secs, "appliances": {}}

    monkeypatch.setattr(dbapi, "truth_recent", _fake)
    TestClient(app).get("/api/truth/recent")
    assert captured["secs"] == 900
