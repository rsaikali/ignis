"""Pure-logic tests for ha_ingest.entities (no broker/DB needed)."""

from __future__ import annotations

from ignis.ha_ingest.entities import (
    build_specs,
    parse_value,
    subscribe_filters,
    topic_index,
)
from ignis.nilm.config import settings


def test_specs_cover_aggregate_and_all_plugs():
    specs = build_specs()
    # 1 aggregate + per appliance (4 sensors; switch entity intentionally ignored).
    expected = 1 + len(settings.nilm_appliances) * 4
    assert len(specs) == expected
    assert any(s.kind == "aggregate" for s in specs)
    assert not any(s.kind == "switch" for s in specs)
    assert specs[0].entity_id == settings.ha_aggregate_entity


def test_topic_shape_matches_statestream():
    prefix = "statestream"
    idx = topic_index(prefix)
    topic = f"{prefix}/sensor/{settings.ha_aggregate_entity.split('.')[1]}/state"
    assert topic in idx
    assert idx[topic].kind == "aggregate"


def test_meross_power_entity_built_correctly():
    idx = topic_index("statestream")
    tag = settings.meross_device_tag
    topic = f"statestream/sensor/four_{tag}_power_w_main_channel/state"
    assert topic in idx
    assert idx[topic].appliance == "four"
    assert idx[topic].kind == "power_w"


def test_subscribe_filters_cover_sensor_only():
    filters = subscribe_filters("statestream")
    assert "statestream/sensor/+/state" in filters
    # Switch entities are not ingested -> no switch filter.
    assert "statestream/switch/+/state" not in filters


def test_parse_value_sensor():
    agg = build_specs()[0]
    assert parse_value(agg, "316") == 316.0
    assert parse_value(agg, "3.14") == 3.14
    assert parse_value(agg, "unknown") is None
    assert parse_value(agg, "off") is None
