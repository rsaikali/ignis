"""Tests for eval.drift (pure retrain-trigger logic)."""

from __future__ import annotations

from ignis.eval.drift import DriftPolicy, evaluate_drift


def _errors(*vals: float) -> list[float]:
    return list(vals)


def test_retrain_when_all_conditions_met():
    d = evaluate_drift(
        "four",
        daily_energy_errors=_errors(0.05, 0.20, 0.21, 0.22),
        days_since_last_train=10,
        new_labeled_hours=30,
    )
    assert d.should_retrain
    assert "drift" in d.reason


def test_no_retrain_during_cooldown():
    d = evaluate_drift(
        "four",
        daily_energy_errors=_errors(0.21, 0.22, 0.23),
        days_since_last_train=3,
        new_labeled_hours=100,
    )
    assert not d.should_retrain
    assert "cooldown" in d.reason


def test_no_retrain_when_not_consecutive():
    # last 3 days: 0.10 breaks the streak.
    d = evaluate_drift(
        "four",
        daily_energy_errors=_errors(0.21, 0.22, 0.10),
        days_since_last_train=10,
        new_labeled_hours=100,
    )
    assert not d.should_retrain
    assert "energy error" in d.reason


def test_no_retrain_when_insufficient_labels():
    d = evaluate_drift(
        "four",
        daily_energy_errors=_errors(0.20, 0.21, 0.22),
        days_since_last_train=10,
        new_labeled_hours=5,
    )
    assert not d.should_retrain
    assert "new labels" in d.reason


def test_not_enough_history_does_not_trigger():
    # Only 2 days but policy needs 3 consecutive.
    d = evaluate_drift(
        "four",
        daily_energy_errors=_errors(0.30, 0.30),
        days_since_last_train=10,
        new_labeled_hours=100,
    )
    assert not d.should_retrain


def test_custom_policy_thresholds():
    policy = DriftPolicy(energy_error_threshold=0.10, consecutive_days=2, cooldown_days=1, min_new_labeled_hours=1)
    d = evaluate_drift(
        "pc",
        daily_energy_errors=_errors(0.12, 0.13),
        days_since_last_train=2,
        new_labeled_hours=2,
        policy=policy,
    )
    assert d.should_retrain
