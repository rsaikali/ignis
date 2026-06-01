"""Self-supervised dataset alignment (pure logic, no DB / no numpy).

Turns raw on-arrival samples into an aligned grid so the engine sees, at each
grid tick, the aggregate VA and every appliance's truth W for the same instant.

Design notes
------------
- **Grid**: fixed step (default 30s, ``settings.ingest_grid_seconds``), floored
  bucket boundaries from the first to the last covered tick.
- **Aggregate** (Linky, ~7s, frequent): averaged within each bucket
  (downsample). A bucket with no aggregate sample is a gap -> dropped.
- **Appliance power** (Meross, on-change, sparse when idle): the last value
  before/within a bucket holds until the next change (forward-fill). Before the
  first ever sample the value is unknown -> ``None`` (caller drops or zero-fills).
- A grid tick is *complete* when the aggregate AND every appliance have a value;
  training uses complete ticks only, so targets are never invented.

All inputs are ``(epoch_seconds: float, value: float)`` lists sorted ascending.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

Sample = tuple[float, float]


def floor_bucket(ts: float, step: int) -> float:
    """Floor an epoch timestamp to the grid step."""
    return (int(ts) // step) * step


def _bucket_mean_aggregate(samples: Sequence[Sample], step: int) -> dict[float, float]:
    """Average aggregate samples within each grid bucket."""
    sums: dict[float, float] = {}
    counts: dict[float, int] = {}
    for ts, val in samples:
        b = floor_bucket(ts, step)
        sums[b] = sums.get(b, 0.0) + val
        counts[b] = counts.get(b, 0) + 1
    return {b: sums[b] / counts[b] for b in sums}


def _forward_fill(samples: Sequence[Sample], grid: Sequence[float]) -> list[float | None]:
    """Hold the last value at/under each grid tick (on-change semantics)."""
    out: list[float | None] = []
    i = 0
    n = len(samples)
    last: float | None = None
    for tick in grid:
        while i < n and samples[i][0] <= tick:
            last = samples[i][1]
            i += 1
        out.append(last)
    return out


@dataclass(frozen=True)
class AlignedDataset:
    """Aligned, gap-free dataset over a single time grid.

    ``grid`` and every series in ``appliances`` and ``aggregate`` have equal
    length; only *complete* ticks (aggregate + all appliances known) are kept.
    """

    step: int
    grid: list[float]
    aggregate: list[float]
    appliances: dict[str, list[float]]

    def __len__(self) -> int:
        return len(self.grid)

    def coverage(self, threshold: float) -> dict[str, ApplianceCoverage]:
        """Per-appliance label coverage over the aligned grid."""
        secs = self.step
        out: dict[str, ApplianceCoverage] = {}
        for app, series in self.appliances.items():
            active = sum(1 for v in series if v > threshold)
            out[app] = ApplianceCoverage(
                appliance=app,
                labeled_hours=len(series) * secs / 3600.0,
                active_hours=active * secs / 3600.0,
                active_ticks=active,
                total_ticks=len(series),
            )
        return out


@dataclass(frozen=True)
class ApplianceCoverage:
    """How much usable label an appliance contributes."""

    appliance: str
    labeled_hours: float
    active_hours: float
    active_ticks: int
    total_ticks: int

    @property
    def active_fraction(self) -> float:
        return self.active_ticks / self.total_ticks if self.total_ticks else 0.0


def build_aligned(
    aggregate: Sequence[Sample],
    appliance_power: Mapping[str, Sequence[Sample]],
    step: int,
    zero_fill_before_first: bool = True,
) -> AlignedDataset:
    """Align aggregate + per-appliance power onto a common grid.

    The grid spans the buckets that actually have an aggregate value (the model
    input). At each tick an appliance value is forward-filled; ``None`` before
    its first sample becomes ``0.0`` when ``zero_fill_before_first`` (a plug
    that has never reported is reasonably assumed off), else the tick is dropped.
    """
    if not aggregate:
        return AlignedDataset(step=step, grid=[], aggregate=[], appliances={app: [] for app in appliance_power})

    agg_by_bucket = _bucket_mean_aggregate(aggregate, step)
    full_grid = sorted(agg_by_bucket)

    filled: dict[str, list[float | None]] = {
        app: _forward_fill(sorted(samples), full_grid) for app, samples in appliance_power.items()
    }

    grid: list[float] = []
    agg_series: list[float] = []
    app_series: dict[str, list[float]] = {app: [] for app in appliance_power}

    for idx, tick in enumerate(full_grid):
        row: dict[str, float] = {}
        complete = True
        for app in appliance_power:
            v = filled[app][idx]
            if v is None:
                if zero_fill_before_first:
                    v = 0.0
                else:
                    complete = False
                    break
            row[app] = v
        if not complete:
            continue
        grid.append(tick)
        agg_series.append(agg_by_bucket[tick])
        for app, v in row.items():
            app_series[app].append(v)

    return AlignedDataset(step=step, grid=grid, aggregate=agg_series, appliances=app_series)
