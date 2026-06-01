"""Tests for the MQTT publisher + HA discovery using a fake client."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime

from publish.contract import DisaggregationSnapshot
from publish.ha_discovery import publish_discovery
from publish.mqtt_publisher import MqttPublisher

TS = datetime(2026, 5, 31, 18, 0, 0, tzinfo=UTC)


@dataclass
class _Sent:
    topic: str
    payload: str
    qos: int
    retain: bool


@dataclass
class FakeClient:
    """Captures publish() calls without touching a broker."""

    sent: list[_Sent] = field(default_factory=list)

    async def publish(self, topic, payload, qos=0, retain=False):
        self.sent.append(_Sent(topic, payload, qos, retain))


async def test_publish_snapshot_single_grouped_message():
    client = FakeClient()
    pub = MqttPublisher(client)
    await pub.publish_snapshot(DisaggregationSnapshot(ts=TS, total_w=2310.0, appliances={"four": 1245.0, "pc": 0.0}))
    # Option B: exactly one message, grouped.
    assert len(client.sent) == 1
    sent = client.sent[0]
    assert sent.topic == "nilm/disaggregation"
    assert sent.qos == 1 and sent.retain


async def test_published_payload_is_valid_json_grouped():
    client = FakeClient()
    pub = MqttPublisher(client)
    await pub.publish_snapshot(DisaggregationSnapshot(ts=TS, total_w=2310.0, appliances={"four": 1245.0}))
    body = json.loads(client.sent[0].payload)
    assert body["total_w"] == 2310.0
    assert body["appliances"]["four"] == 1245.0
    assert body["ts"] == "2026-05-31T18:00:00Z"


async def test_ha_discovery_reads_grouped_topic():
    client = FakeClient()
    await publish_discovery(client, appliances=["four", "pc"])
    assert [s.topic for s in client.sent] == [
        "homeassistant/sensor/ignis_four_power/config",
        "homeassistant/sensor/ignis_pc_power/config",
    ]
    cfg = json.loads(client.sent[0].payload)
    # All HA sensors read the same grouped snapshot, extracting their key.
    assert cfg["state_topic"] == "nilm/disaggregation"
    assert cfg["value_template"] == "{{ value_json.appliances.four }}"
    assert cfg["device_class"] == "power"
    assert all(s.retain for s in client.sent)
