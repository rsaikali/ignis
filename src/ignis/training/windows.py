"""Sliding-window assembly for multi-output Seq2Point (pure, no numpy).

The model takes a window of aggregate power and predicts every appliance's
power at the window's centre. Unlike Linkya's signature path (one appliance
active, the rest forced to zero), here each centre carries the REAL power of
all appliances at that instant -- true concurrent self-supervised targets.

Kept numpy-free so it is unit-testable in the dev venv; the trainer converts
to arrays at the ``model.fit`` boundary.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class WindowedData:
    """Windows + per-appliance centre targets, all index-aligned.

    ``x[i]`` is a window of length ``sequence_length``; ``targets[app][i]`` is
    that appliance's power at the window centre.
    """

    sequence_length: int
    x: list[list[float]]
    targets: dict[str, list[float]]

    def __len__(self) -> int:
        return len(self.x)


def make_windows(
    aggregate: Sequence[float],
    appliances: Mapping[str, Sequence[float]],
    sequence_length: int,
    stride: int = 10,
) -> WindowedData:
    """Build centred windows with concurrent multi-output targets.

    ``aggregate`` and every appliance series must share length (the aligned
    grid). ``sequence_length`` is forced odd so a centre point exists.
    """
    if sequence_length % 2 == 0:
        sequence_length -= 1
    half = sequence_length // 2

    n = len(aggregate)
    for app, series in appliances.items():
        if len(series) != n:
            raise ValueError(f"appliance '{app}' length {len(series)} != aggregate {n}")

    x: list[list[float]] = []
    targets: dict[str, list[float]] = {app: [] for app in appliances}

    if n < sequence_length:
        return WindowedData(sequence_length=sequence_length, x=x, targets=targets)

    for i in range(half, n - half, stride):
        x.append(list(aggregate[i - half : i + half + 1]))
        for app, series in appliances.items():
            targets[app].append(series[i])

    return WindowedData(sequence_length=sequence_length, x=x, targets=targets)


def balance_windows(
    data: WindowedData,
    threshold: float,
    dead_ratio: float = 1.0,
    seed: int = 0,
) -> WindowedData:
    """Subsample "dead" windows (no appliance ON at the centre).

    Per-appliance activation is ~2-3% of ticks, so most windows have every
    target at zero; a model minimises MAE by always predicting ~0 (F1 = 0).
    Keep every *active* window (>=1 appliance above ``threshold`` at the centre)
    and keep dead windows up to ``dead_ratio`` x the active count, sampled
    deterministically (``seed``). ``dead_ratio=1.0`` => ~50/50 active/dead.
    """
    apps = list(data.targets)
    active_idx: list[int] = []
    dead_idx: list[int] = []
    for i in range(len(data)):
        if any(data.targets[app][i] > threshold for app in apps):
            active_idx.append(i)
        else:
            dead_idx.append(i)

    keep_dead = min(len(dead_idx), int(len(active_idx) * dead_ratio))
    rng = random.Random(seed)
    sampled_dead = rng.sample(dead_idx, keep_dead) if keep_dead < len(dead_idx) else dead_idx

    keep = sorted(active_idx + sampled_dead)
    return WindowedData(
        sequence_length=data.sequence_length,
        x=[data.x[i] for i in keep],
        targets={app: [data.targets[app][i] for i in keep] for app in apps},
    )


def activation_stats(data: WindowedData, threshold: float) -> dict[str, float]:
    """Fraction of windows where each appliance is ON at the centre."""
    total = len(data) or 1
    return {app: sum(1 for v in series if v > threshold) / total for app, series in data.targets.items()}
