"""Drift-triggered retrain decision (pure logic).

Trigger (CLAUDE.md): rolling 7-day energy-error vs HA per appliance above a
threshold for 3 consecutive days AND >= X new labeled activation hours since
last train AND cooldown >= 7 days since the last train.

This module decides; it does not act. ``eval`` feeds it daily energy-error
values; ``publish``/training consume the resulting request.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

# Defaults mirror the CLAUDE.md retrain policy.
DEFAULT_ENERGY_ERROR_THRESHOLD = 0.15
DEFAULT_CONSECUTIVE_DAYS = 3
DEFAULT_COOLDOWN_DAYS = 7
DEFAULT_MIN_NEW_LABELED_HOURS = 24.0


@dataclass(frozen=True)
class DriftPolicy:
    """Tunable thresholds for the retrain trigger."""

    energy_error_threshold: float = DEFAULT_ENERGY_ERROR_THRESHOLD
    consecutive_days: int = DEFAULT_CONSECUTIVE_DAYS
    cooldown_days: int = DEFAULT_COOLDOWN_DAYS
    min_new_labeled_hours: float = DEFAULT_MIN_NEW_LABELED_HOURS


@dataclass(frozen=True)
class RetrainDecision:
    """Outcome of a drift evaluation for one appliance."""

    appliance: str
    should_retrain: bool
    reason: str


def _tail_exceeds(daily_errors: Sequence[float], threshold: float, n: int) -> bool:
    """True if the last ``n`` daily errors all strictly exceed ``threshold``."""
    if len(daily_errors) < n:
        return False
    return all(e > threshold for e in daily_errors[-n:])


def evaluate_drift(
    appliance: str,
    daily_energy_errors: Sequence[float],
    days_since_last_train: int,
    new_labeled_hours: float,
    policy: DriftPolicy | None = None,
) -> RetrainDecision:
    """Decide whether ``appliance`` should be retrained.

    ``daily_energy_errors`` is ordered oldest -> newest, one value per day
    (relative energy error vs HA). All three conditions must hold.
    """
    p = policy or DriftPolicy()

    if days_since_last_train < p.cooldown_days:
        return RetrainDecision(appliance, False, f"cooldown ({days_since_last_train}/{p.cooldown_days}d)")

    if not _tail_exceeds(daily_energy_errors, p.energy_error_threshold, p.consecutive_days):
        return RetrainDecision(
            appliance,
            False,
            f"energy error under {p.energy_error_threshold:.0%} within last {p.consecutive_days}d",
        )

    if new_labeled_hours < p.min_new_labeled_hours:
        return RetrainDecision(
            appliance,
            False,
            f"insufficient new labels ({new_labeled_hours:.1f}/{p.min_new_labeled_hours:.1f}h)",
        )

    return RetrainDecision(
        appliance,
        True,
        f"drift {p.consecutive_days}d > {p.energy_error_threshold:.0%}, "
        f"{new_labeled_hours:.1f}h new labels, cooldown met",
    )
