"""Tests for training.dataset alignment (pure, no DB)."""

from __future__ import annotations

from training.dataset import (
    AlignedDataset,
    build_aligned,
    floor_bucket,
)


def test_floor_bucket():
    assert floor_bucket(0, 30) == 0
    assert floor_bucket(29, 30) == 0
    assert floor_bucket(30, 30) == 30
    assert floor_bucket(91, 30) == 90


def test_aggregate_averaged_per_bucket():
    # Two aggregate samples in bucket 0, one in bucket 30.
    agg = [(0.0, 100.0), (10.0, 200.0), (30.0, 50.0)]
    ds = build_aligned(agg, {}, step=30)
    assert ds.grid == [0.0, 30.0]
    assert ds.aggregate == [150.0, 50.0]  # mean of bucket 0, then 50


def test_forward_fill_holds_last_value():
    agg = [(0.0, 10.0), (30.0, 10.0), (60.0, 10.0)]
    # appliance reports 5 at t=0, changes to 80 at t=35; bucket 30 sees 5,
    # bucket 60 sees 80.
    app = {"four": [(0.0, 5.0), (35.0, 80.0)]}
    ds = build_aligned(agg, app, step=30)
    assert ds.grid == [0.0, 30.0, 60.0]
    assert ds.appliances["four"] == [5.0, 5.0, 80.0]


def test_zero_fill_before_first_sample():
    agg = [(0.0, 10.0), (30.0, 10.0)]
    # appliance first reports only at t=40 -> bucket 0 and 30 are before it.
    app = {"pc": [(40.0, 12.0)]}
    ds = build_aligned(agg, app, step=30)
    assert ds.appliances["pc"] == [0.0, 0.0]  # zero-filled before first


def test_drop_before_first_when_not_zero_fill():
    agg = [(0.0, 10.0), (30.0, 10.0), (60.0, 10.0)]
    app = {"pc": [(50.0, 12.0)]}
    ds = build_aligned(agg, app, step=30, zero_fill_before_first=False)
    # Buckets 0 and 30 dropped (pc unknown); only bucket 60 kept.
    assert ds.grid == [60.0]
    assert ds.appliances["pc"] == [12.0]


def test_empty_aggregate_yields_empty():
    ds = build_aligned([], {"four": [(0.0, 5.0)]}, step=30)
    assert len(ds) == 0
    assert ds.appliances == {"four": []}


def test_lengths_are_consistent():
    agg = [(0.0, 10.0), (30.0, 20.0), (60.0, 30.0)]
    app = {"a": [(0.0, 1.0)], "b": [(0.0, 2.0)]}
    ds = build_aligned(agg, app, step=30)
    n = len(ds)
    assert len(ds.aggregate) == n
    assert all(len(s) == n for s in ds.appliances.values())


def test_coverage_counts_active_ticks():
    ds = AlignedDataset(
        step=30,
        grid=[0.0, 30.0, 60.0, 90.0],
        aggregate=[100.0, 100.0, 100.0, 100.0],
        appliances={"four": [0.0, 50.0, 60.0, 0.0]},
    )
    cov = ds.coverage(threshold=15)["four"]
    assert cov.active_ticks == 2
    assert cov.total_ticks == 4
    assert cov.active_fraction == 0.5
    # 4 ticks * 30s = 120s = 1/30 h labeled; 2 active ticks = 60s active.
    assert abs(cov.labeled_hours - 4 * 30 / 3600) < 1e-9
    assert abs(cov.active_hours - 2 * 30 / 3600) < 1e-9
