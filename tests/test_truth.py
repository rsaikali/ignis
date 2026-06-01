"""Tests for eval.truth forward-fill (pure helper)."""

from __future__ import annotations

from ignis.eval.truth import _forward_fill_bool


def test_default_off_before_first_change():
    # No change at/under the first two ticks -> OFF.
    out = _forward_fill_bool([(100.0, True)], [0.0, 50.0, 100.0, 150.0])
    assert out == [False, False, True, True]


def test_holds_last_state():
    changes = [(0.0, True), (90.0, False)]
    out = _forward_fill_bool(changes, [0.0, 30.0, 60.0, 90.0, 120.0])
    assert out == [True, True, True, False, False]


def test_empty_changes_all_off():
    out = _forward_fill_bool([], [0.0, 30.0, 60.0])
    assert out == [False, False, False]


def test_multiple_changes_within_one_tick_takes_last():
    # Two flips before tick 100 -> the later (False) wins.
    changes = [(10.0, True), (20.0, False)]
    out = _forward_fill_bool(changes, [100.0])
    assert out == [False]
