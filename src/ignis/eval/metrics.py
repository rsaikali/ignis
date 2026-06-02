"""Per-appliance NILM-vs-HA metrics: state F1 + energy error.

Pure functions over aligned power series (same time grid, same length). No
numpy/pandas dependency so this runs anywhere, including the Pi.

Definitions
-----------
- ON/OFF state: power above a threshold (W), for both truth and prediction.
  The Meross plugs stay powered (used only to meter), so their switch entity is
  not a usable activation signal -- ON/OFF is derived from measured power.
- State F1: F1 of the ON class (prediction ON vs truth ON).
- Energy error: ``|sum(pred) - sum(truth)| / sum(truth)``. On a uniform grid
  the per-sample dt cancels, so summing power is proportional to energy.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

# Acceptance gate (CLAUDE.md). Energy error uses the stricter 0.15 bound.
STATE_F1_MIN = 0.8
ENERGY_ERROR_MAX = 0.15


def on_off(power: Sequence[float], threshold: float) -> list[bool]:
    """Boolean ON series: power strictly above ``threshold`` (W)."""
    return [p > threshold for p in power]


def state_f1(truth_on: Sequence[bool], pred_on: Sequence[bool]) -> float:
    """F1 of the ON class.

    Returns 1.0 when both series are all-OFF (nothing to detect, no false
    positives), 0.0 when one side has ON and the other none.
    """
    if len(truth_on) != len(pred_on):
        raise ValueError("truth_on and pred_on must have equal length")
    tp = sum(1 for t, p in zip(truth_on, pred_on, strict=True) if t and p)
    fp = sum(1 for t, p in zip(truth_on, pred_on, strict=True) if p and not t)
    fn = sum(1 for t, p in zip(truth_on, pred_on, strict=True) if t and not p)
    if tp == 0:
        # No true positives: F1 is 1.0 only if there were no positives at all.
        return 1.0 if fp == 0 and fn == 0 else 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def energy_error(truth_w: Sequence[float], pred_w: Sequence[float]) -> float:
    """Relative energy error ``|Σpred - Σtruth| / Σtruth``.

    Returns 0.0 when both totals are ~0, and ``inf`` when truth is ~0 but the
    prediction is not (no baseline to normalise against).
    """
    if len(truth_w) != len(pred_w):
        raise ValueError("truth_w and pred_w must have equal length")
    truth_sum = sum(truth_w)
    pred_sum = sum(pred_w)
    if truth_sum == 0:
        return 0.0 if pred_sum == 0 else float("inf")
    return abs(pred_sum - truth_sum) / truth_sum


@dataclass(frozen=True)
class ApplianceMetrics:
    """Evaluation result for one appliance over one window."""

    appliance: str
    state_f1: float
    energy_error: float
    n_samples: int

    @property
    def passes_gate(self) -> bool:
        """True when both acceptance-gate conditions hold."""
        return self.state_f1 >= STATE_F1_MIN and self.energy_error <= ENERGY_ERROR_MAX


def evaluate_appliance(
    appliance: str,
    truth_w: Sequence[float],
    pred_w: Sequence[float],
    threshold: float,
) -> ApplianceMetrics:
    """Compute metrics for one appliance.

    Truth and prediction ON/OFF are both derived from power via ``threshold``.
    """
    truth_state = on_off(truth_w, threshold)
    pred_state = on_off(pred_w, threshold)
    return ApplianceMetrics(
        appliance=appliance,
        state_f1=state_f1(truth_state, pred_state),
        energy_error=energy_error(truth_w, pred_w),
        n_samples=len(truth_w),
    )
