"""Portfolio NILM MQTT contract (spec section 6.4).

Frozen topic + payload shapes so Ignis and the portfolio stay decoupled.
Pure builders -- no MQTT here -- so the contract is unit-testable.

**Option B (grouped payload) -- the decision recorded in spec 6.4.** One
message per inference cycle = one coherent snapshot, so the portfolio consumer
never has to re-stitch time-shifted per-appliance messages::

    topic:   nilm/disaggregation
    payload: { "ts": "<ISO8601>", "total_w": <float>,
               "appliances": { "<appliance_key>": <power_w>, ... } }

The portfolio derives ``share`` itself (spec 6.1) -- the contract carries raw
power only. ``nilm/_meta/model`` is an Ignis-side extension (not part of the
frozen 6.4 contract) carrying the active-model version + metrics.

Rules: ``ts`` is ISO-8601 UTC; powers are watts (float).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime

TOPIC_PREFIX = "nilm"
# Frozen 6.4 topic: grouped snapshot.
DISAGGREGATION_TOPIC = f"{TOPIC_PREFIX}/disaggregation"
# Ignis extension (outside 6.4): active-model metadata.
META_TOPIC = f"{TOPIC_PREFIX}/_meta/model"


def iso_utc(ts: datetime) -> str:
    """Format a datetime as ISO-8601 UTC with a trailing ``Z``."""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class DisaggregationSnapshot:
    """One coherent inference cycle -> nilm/disaggregation (spec 6.4).

    ``appliances`` maps appliance_key -> predicted power (W). ``total_w`` is
    the measured aggregate (model input) for the same instant.
    """

    ts: datetime
    total_w: float
    appliances: Mapping[str, float]

    def payload(self) -> dict:
        return {
            "ts": iso_utc(self.ts),
            "total_w": round(self.total_w, 1),
            "appliances": {k: round(v, 1) for k, v in self.appliances.items()},
        }


@dataclass(frozen=True)
class ModelMeta:
    """Active-model metadata -> nilm/_meta/model (Ignis extension, not 6.4)."""

    version: str
    model_type: str
    trained_at: datetime
    appliances: list[str]
    metrics: Mapping[str, Mapping[str, float]] = field(default_factory=dict)

    def payload(self) -> dict:
        return {
            "version": self.version,
            "model_type": self.model_type,
            "trained_at": iso_utc(self.trained_at),
            "appliances": list(self.appliances),
            "metrics": {app: {k: round(v, 3) for k, v in m.items()} for app, m in self.metrics.items()},
        }
