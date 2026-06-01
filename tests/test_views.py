"""Tests for the compatibility view DDL.

No live DB: assert the DDL strings are well-formed and idempotent, and that
they map ha_samples columns to the shapes the engine / eval expect. The
end-to-end behaviour (engine reading linky_realtime) is verified manually
against the running stack.
"""

from __future__ import annotations

from ha_ingest.views import _VIEWS, ensure_views


def test_all_views_are_create_or_replace():
    # Idempotent: every view uses CREATE OR REPLACE.
    for ddl in _VIEWS:
        assert "CREATE OR REPLACE VIEW" in ddl


def test_linky_realtime_maps_aggregate_to_time_papp():
    ddl = next(d for d in _VIEWS if "linky_realtime" in d)
    # Engine reads get_consumption_data -> time_bucket(interval, time), AVG(papp).
    assert "ts AS time" in ddl
    assert "value AS papp" in ddl
    assert "kind = 'aggregate'" in ddl


def test_appliance_power_filters_power_w():
    ddl = next(d for d in _VIEWS if "appliance_power" in d)
    assert "value AS power_w" in ddl
    assert "kind = 'power_w'" in ddl
    assert "appliance IS NOT NULL" in ddl


def test_appliance_onoff_maps_switch_to_bool():
    ddl = next(d for d in _VIEWS if "appliance_onoff" in d)
    assert "value > 0.5" in ddl
    assert "kind = 'switch'" in ddl


async def test_ensure_views_executes_every_ddl():
    executed: list[str] = []

    class FakeConn:
        async def execute(self, ddl: str) -> None:
            executed.append(ddl)

    await ensure_views(FakeConn())
    assert executed == list(_VIEWS)
    assert len(executed) == 3
