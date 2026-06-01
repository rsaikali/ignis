"""Tests for eval.metrics (pure, no DB/model)."""

from __future__ import annotations

import math

from eval.metrics import (
    ENERGY_ERROR_MAX,
    STATE_F1_MIN,
    ApplianceMetrics,
    energy_error,
    evaluate_appliance,
    on_off,
    state_f1,
)


def test_on_off_threshold():
    assert on_off([0, 10, 16, 15], 15) == [False, False, True, False]


def test_state_f1_perfect():
    truth = [True, True, False, False]
    assert state_f1(truth, truth) == 1.0


def test_state_f1_all_off_is_one():
    assert state_f1([False, False], [False, False]) == 1.0


def test_state_f1_missed_all_on():
    assert state_f1([True, True], [False, False]) == 0.0


def test_state_f1_partial():
    # tp=1, fp=1, fn=1 -> precision=recall=0.5 -> F1=0.5
    truth = [True, True, False]
    pred = [True, False, True]
    assert state_f1(truth, pred) == 0.5


def test_energy_error_basic():
    # truth sum=100, pred sum=110 -> 0.10
    assert math.isclose(energy_error([50, 50], [55, 55]), 0.10)


def test_energy_error_zero_truth_zero_pred():
    assert energy_error([0, 0], [0, 0]) == 0.0


def test_energy_error_zero_truth_nonzero_pred_is_inf():
    assert energy_error([0, 0], [1, 0]) == float("inf")


def test_evaluate_appliance_uses_switch_truth():
    # pred matches power but switch says OFF the whole time -> some FP.
    m = evaluate_appliance(
        "four",
        truth_w=[100, 100],
        pred_w=[100, 100],
        threshold=15,
        truth_on=[False, False],
    )
    assert m.appliance == "four"
    assert m.n_samples == 2
    assert m.state_f1 == 0.0  # pred ON, truth (switch) OFF


def test_gate_pass_and_fail():
    good = ApplianceMetrics("x", state_f1=0.9, energy_error=0.10, n_samples=10)
    assert good.passes_gate
    bad_f1 = ApplianceMetrics("x", state_f1=0.7, energy_error=0.10, n_samples=10)
    assert not bad_f1.passes_gate
    bad_energy = ApplianceMetrics("x", state_f1=0.9, energy_error=0.20, n_samples=10)
    assert not bad_energy.passes_gate


def test_gate_constants():
    assert STATE_F1_MIN == 0.8
    assert ENERGY_ERROR_MAX == 0.15
