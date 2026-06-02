"""Tests for backfill mapping logic (pure; no SSH/DB)."""

from __future__ import annotations

from ignis.ha_ingest.entities import entity_index, parse_value


def test_entity_index_keyed_by_entity_id():
    idx = entity_index()
    # Aggregate + Meross entities are reachable by their entity_id.
    assert "sensor.puissance_generale" in idx
    assert idx["sensor.puissance_generale"].kind == "aggregate"
    assert "sensor.four_mss315_power_w_main_channel" in idx
    assert idx["sensor.four_mss315_power_w_main_channel"].appliance == "four"


def test_recorder_raw_states_parse_like_live():
    idx = entity_index()
    agg = idx["sensor.puissance_generale"]
    power = idx["sensor.four_mss315_power_w_main_channel"]
    # Recorder stores the same string states MQTT carries.
    assert parse_value(agg, "316") == 316.0
    assert parse_value(agg, "unavailable") is None
    assert parse_value(power, "42.5") == 42.5
    assert parse_value(power, "unknown") is None


def test_switch_entity_not_indexed():
    # Switch entities are intentionally not ingested.
    assert entity_index().get("switch.four_mss315_main_channel") is None


def test_unknown_entity_filtered():
    idx = entity_index()
    assert idx.get("sensor.something_else") is None
