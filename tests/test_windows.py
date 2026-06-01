"""Tests for training.windows (pure, no numpy)."""

from __future__ import annotations

import pytest

from ignis.training.windows import activation_stats, balance_windows, make_windows


def test_sequence_length_forced_odd():
    agg = list(range(10))
    w = make_windows(agg, {}, sequence_length=4)
    assert w.sequence_length == 3


def test_windows_centred_with_concurrent_targets():
    # 7 ticks, seq_len 3 (half=1), stride 1 -> centres at i=1..5.
    agg = [10, 11, 12, 13, 14, 15, 16]
    apps = {"a": [0, 1, 2, 3, 4, 5, 6], "b": [6, 5, 4, 3, 2, 1, 0]}
    w = make_windows(agg, apps, sequence_length=3, stride=1)
    assert len(w) == 5
    # First window centred on i=1: [10,11,12]; targets a=1, b=5.
    assert w.x[0] == [10, 11, 12]
    assert w.targets["a"][0] == 1
    assert w.targets["b"][0] == 5
    # Last centre i=5: window [14,15,16]; a=5, b=1.
    assert w.x[-1] == [14, 15, 16]
    assert w.targets["a"][-1] == 5
    assert w.targets["b"][-1] == 1


def test_stride_reduces_window_count():
    agg = list(range(21))
    apps = {"a": list(range(21))}
    w1 = make_windows(agg, apps, sequence_length=3, stride=1)
    w5 = make_windows(agg, apps, sequence_length=3, stride=5)
    assert len(w5) < len(w1)
    # All targets line up with their window centre value (a == index here).
    assert w5.targets["a"][0] == w5.x[0][1]


def test_too_short_yields_empty():
    w = make_windows([1, 2], {"a": [1, 2]}, sequence_length=5)
    assert len(w) == 0
    assert w.targets["a"] == []


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        make_windows([1, 2, 3], {"a": [1, 2]}, sequence_length=3)


def test_all_targets_index_aligned_with_x():
    agg = list(range(30))
    apps = {"a": list(range(30)), "b": list(range(30))}
    w = make_windows(agg, apps, sequence_length=5, stride=3)
    assert len(w.targets["a"]) == len(w.x)
    assert len(w.targets["b"]) == len(w.x)


def test_balance_keeps_all_active_and_matches_dead():
    # 10 windows: 2 active (four ON), 8 dead. dead_ratio=1 -> keep 2 dead.
    agg = list(range(30))
    four = [0.0] * 30
    four[5] = 100.0  # centre of an early window
    four[20] = 100.0
    w = make_windows(agg, {"four": four}, sequence_length=3, stride=1)
    active_before = sum(1 for v in w.targets["four"] if v > 15)
    b = balance_windows(w, threshold=15, dead_ratio=1.0, seed=0)
    active_after = sum(1 for v in b.targets["four"] if v > 15)
    # All active windows kept.
    assert active_after == active_before
    # Total = active + min(dead, active*1).
    assert len(b) == active_after * 2


def test_balance_is_deterministic():
    agg = list(range(60))
    series = [0.0] * 60
    series[10] = 100.0
    w = make_windows(agg, {"a": series}, sequence_length=3, stride=1)
    b1 = balance_windows(w, threshold=15, seed=42)
    b2 = balance_windows(w, threshold=15, seed=42)
    assert b1.x == b2.x


def test_balance_keeps_all_when_fewer_dead_than_ratio():
    # Mostly active -> few dead -> keep them all (no oversampling of dead).
    agg = list(range(12))
    series = [100.0] * 12
    series[5] = 0.0
    w = make_windows(agg, {"a": series}, sequence_length=3, stride=1)
    b = balance_windows(w, threshold=15, dead_ratio=1.0)
    # Every window kept (dead count <= active count).
    assert len(b) == len(w)


def test_activation_stats():
    agg = list(range(30))
    series = [0.0] * 30
    series[5] = 100.0
    series[6] = 100.0
    w = make_windows(agg, {"a": series}, sequence_length=3, stride=1)
    stats = activation_stats(w, threshold=15)
    assert 0.0 < stats["a"] < 1.0
