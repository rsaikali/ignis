"""Tests for eval.promote (score logic + champion/challenger file ops)."""

from __future__ import annotations

import json

import pytest

from ignis.eval import promote
from ignis.eval.promote import (
    Score,
    latest_challenger,
    maybe_promote,
    prune_challengers,
    score_from_comparison,
)


@pytest.fixture
def models(tmp_path, monkeypatch):
    from ignis.nilm.config import settings

    monkeypatch.setattr(settings, "nilm_model_path", str(tmp_path))
    return tmp_path


def _challenger(d, name, *, gate_passes=0, mean_f1=0.0, with_sidecars=True):
    (d / f"{name}.keras").write_text("k")
    if with_sidecars:
        # Build a comparison.json whose score matches the request.
        apps = {}
        for i in range(6):
            passes = i < gate_passes
            apps[f"a{i}"] = {"state_f1": mean_f1, "passes_gate": passes}
        (d / f"{name}.comparison.json").write_text(json.dumps({"appliances": apps}))
        (d / f"{name}.scalers.pkl").write_bytes(b"x")
    return d / f"{name}.keras"


def test_score_ordering():
    assert Score(3, 0.5) > Score(2, 0.9)  # gate passes dominate
    assert Score(2, 0.9) > Score(2, 0.8)  # mean_f1 breaks ties
    assert not (Score(1, 0.5) > Score(1, 0.5))


def test_score_from_comparison():
    cmp = {"appliances": {"a": {"state_f1": 0.8, "passes_gate": True}, "b": {"state_f1": 0.4, "passes_gate": False}}}
    s = score_from_comparison(cmp)
    assert s.gate_passes == 1
    assert abs(s.mean_f1 - 0.6) < 1e-9


def test_latest_challenger_picks_newest(models):
    _challenger(models, "ignis_gru_20260101T000000")
    _challenger(models, "ignis_gru_20260601T000000")
    assert latest_challenger().stem == "ignis_gru_20260601T000000"


def test_latest_challenger_excludes_champion(models):
    (models / "champion.keras").write_text("k")
    assert latest_challenger() is None


def test_first_model_promoted_unconditionally(models):
    ch = _challenger(models, "ignis_gru_x", gate_passes=0, mean_f1=0.1)
    promoted, champ_score = maybe_promote(ch, Score(0, 0.1))
    assert promoted is True
    assert champ_score is None
    assert (models / "champion.keras").exists()
    assert (models / "champion.comparison.json").exists()  # sidecar copied


def test_better_challenger_promoted(models):
    weak = _challenger(models, "ignis_gru_weak", gate_passes=1, mean_f1=0.3)
    maybe_promote(weak, score_from_comparison(json.loads((models / "ignis_gru_weak.comparison.json").read_text())))
    strong = _challenger(models, "ignis_gru_strong", gate_passes=4, mean_f1=0.7)
    promoted, champ_score = maybe_promote(strong, Score(4, 0.7))
    assert promoted is True
    assert champ_score == Score(1, 0.3)


def test_worse_challenger_not_promoted(models):
    strong = _challenger(models, "ignis_gru_strong", gate_passes=4, mean_f1=0.7)
    maybe_promote(strong, Score(4, 0.7))
    weak = _challenger(models, "ignis_gru_weak", gate_passes=1, mean_f1=0.3)
    promoted, _ = maybe_promote(weak, Score(1, 0.3))
    assert promoted is False


def test_prune_keeps_newest(models):
    for i in range(7):
        _challenger(models, f"ignis_gru_2026010{i}T000000", with_sidecars=False)
    removed = prune_challengers(keep=3)
    assert len(removed) == 4
    remaining = sorted(p.stem for p in models.glob("*.keras"))
    assert len(remaining) == 3
    assert "ignis_gru_20260106T000000" in remaining  # newest kept


def test_prune_ignores_champion(models):
    (models / "champion.keras").write_text("k")
    for i in range(3):
        _challenger(models, f"ignis_gru_2026010{i}T000000", with_sidecars=False)
    prune_challengers(keep=1)
    assert (models / "champion.keras").exists()
    assert promote.champion_path().exists()
