"""Entity-id <-> MQTT-statestream-topic mapping for the HA setup.

HA ``mqtt_statestream`` republishes every entity state under
``<base_topic>/<domain>/<object_id>/state`` with the raw state as payload.

The Meross mss315 plugs expose, per appliance ``<app>``::

    sensor.<app>_<tag>_power_w_main_channel      # instant W  -> NILM label
    sensor.<app>_<tag>_energy_kwh_main_channel   # cumulative -> energy gate
    sensor.<app>_<tag>_current_a_main_channel    # A
    sensor.<app>_<tag>_voltage_v_main_channel    # V

The ``switch.*`` entity is intentionally ignored: the plugs stay powered (used
only to meter), so the switch state carries no activation signal. ON/OFF truth
is derived from measured ``power_w`` instead.

The aggregate NILM input is a single sensor (Linky apparent power, VA).
"""

from __future__ import annotations

from dataclasses import dataclass

from ignis.nilm.config import settings

# Measurement kinds we keep per Meross plug. Maps a logical kind to the entity
# suffix used by the Meross integration.
MEROSS_SENSOR_KINDS: dict[str, str] = {
    "power_w": "power_w",
    "energy_kwh": "energy_kwh",
    "current_a": "current_a",
    "voltage_v": "voltage_v",
}


@dataclass(frozen=True)
class EntitySpec:
    """One HA entity we ingest."""

    entity_id: str  # e.g. "sensor.four_mss315_power_w_main_channel"
    appliance: str | None  # "four" ... ; None for the aggregate
    kind: str  # "aggregate" | "power_w" | "energy_kwh" | "current_a" | "voltage_v"

    @property
    def domain(self) -> str:
        return self.entity_id.split(".", 1)[0]

    @property
    def object_id(self) -> str:
        return self.entity_id.split(".", 1)[1]

    def topic(self, prefix: str) -> str:
        return f"{prefix}/{self.domain}/{self.object_id}/state"


def _meross_sensor(app: str, suffix: str, tag: str) -> str:
    return f"sensor.{app}_{tag}_{suffix}_main_channel"


def build_specs() -> list[EntitySpec]:
    """All entities to ingest, derived from settings."""
    tag = settings.meross_device_tag
    specs: list[EntitySpec] = [
        EntitySpec(settings.ha_aggregate_entity, None, "aggregate"),
    ]
    for app in settings.nilm_appliances:
        for kind, suffix in MEROSS_SENSOR_KINDS.items():
            specs.append(EntitySpec(_meross_sensor(app, suffix, tag), app, kind))
    return specs


def topic_index(prefix: str) -> dict[str, EntitySpec]:
    """Map statestream topic -> spec, for O(1) lookup on message arrival."""
    return {spec.topic(prefix): spec for spec in build_specs()}


def subscribe_filters(prefix: str) -> list[str]:
    """Wildcard topic filters covering all ingested domains."""
    domains = {spec.domain for spec in build_specs()}
    return [f"{prefix}/{domain}/+/state" for domain in sorted(domains)]


def entity_index() -> dict[str, EntitySpec]:
    """Map entity_id -> spec (for history backfill keyed by entity_id)."""
    return {spec.entity_id: spec for spec in build_specs()}


# State strings that carry no numeric value.
NON_NUMERIC = {"unavailable", "unknown", "none", ""}


def parse_value(spec: EntitySpec, payload: str) -> float | None:
    """Convert a raw statestream payload to a float, or None if not storable."""
    s = payload.strip().lower()
    if s in NON_NUMERIC:
        return None
    try:
        return float(s)
    except ValueError:
        return None
