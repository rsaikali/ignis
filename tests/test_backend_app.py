"""Tests for the admin app (FastAPI TestClient + temp models dir)."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from ignis.backend import app as app_module
from ignis.backend.app import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Point discover() at a temp dir by patching settings.nilm_model_path.
    from ignis.nilm.config import settings

    monkeypatch.setattr(settings, "nilm_model_path", str(tmp_path))
    # registry.discover reads settings at call time, so this is enough.
    return TestClient(app), tmp_path


def _model(d, name, report=None, comparison=None, scalers=False):
    (d / f"{name}.keras").write_text("k")
    if report is not None:
        (d / f"{name}.report.json").write_text(json.dumps(report))
    if comparison is not None:
        (d / f"{name}.comparison.json").write_text(json.dumps(comparison))
    if scalers:
        (d / f"{name}.scalers.pkl").write_bytes(b"x")


def test_health(client):
    c, _ = client
    r = c.get("/api/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_index_empty(client):
    c, _ = client
    r = c.get("/")
    assert r.status_code == 200
    assert "Aucun modele" in r.text


def test_index_lists_model(client):
    c, d = client
    _model(d, "ignis_gru_x", report={"period_end": "2026-06-01", "appliances": {}}, scalers=True)
    r = c.get("/")
    assert "ignis_gru_x" in r.text


def test_api_models_json(client):
    c, d = client
    _model(
        d,
        "m1",
        report={"period_end": "2026-06-01", "appliances": {}},
        comparison={"passed": ["four"], "failed": []},
        scalers=True,
    )
    r = c.get("/api/models")
    body = r.json()
    assert body[0]["name"] == "m1"
    assert body[0]["gate_passed"] == ["four"]
    assert body[0]["predictable"] is True


def test_api_models_carries_metrics(client):
    c, d = client
    _model(
        d,
        "m1",
        report={"period_end": "2026-06-01", "appliances": {}},
        comparison={
            "appliances": {"television": {"state_f1": 0.701, "energy_error": 0.82, "passes_gate": False}},
            "passed": [],
            "failed": ["television"],
        },
        scalers=True,
    )
    body = c.get("/api/models").json()
    assert body[0]["metrics"]["television"]["state_f1"] == 0.701
    assert body[0]["metrics"]["television"]["passes_gate"] is False


def test_model_detail_with_comparison(client):
    c, d = client
    _model(
        d,
        "m1",
        report={
            "period_start": "2026-05-20",
            "period_end": "2026-06-01",
            "grid_seconds": 30,
            "n_windows": 100,
            "labeled_hours": 12.0,
            "appliances": {"four": {"labeled_hours": 12.0, "active_hours": 1.0, "active_ticks": 120}},
        },
        comparison={
            "appliances": {"four": {"state_f1": 0.9, "energy_error": 0.1, "passes_gate": True}},
            "passed": ["four"],
            "failed": [],
        },
    )
    r = c.get("/model/m1")
    assert r.status_code == 200
    assert "four" in r.text
    assert "PASS" in r.text
    assert "Comparaison HA vs NILM" in r.text


def test_model_detail_not_found(client):
    c, _ = client
    r = c.get("/model/nope")
    assert "introuvable" in r.text


def test_app_module_importable():
    assert app_module.app is not None
