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


def test_truth_route_default_window(monkeypatch):
    from ignis.backend import dbapi

    captured = {}

    def _fake(secs):
        captured["secs"] = secs
        return {"window_seconds": secs, "appliances": {}}

    monkeypatch.setattr(dbapi, "truth_recent", _fake)
    TestClient(app).get("/api/truth/recent")
    assert captured["secs"] == 900
