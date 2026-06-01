"""MQTT publisher for the portfolio NILM contract (spec 6.4, Option B).

Thin aiomqtt wrapper: serialise contract payloads and publish them retained
at QoS 1. One inference cycle = one grouped ``nilm/disaggregation`` message;
``nilm/_meta/model`` is published when the active model changes.
"""

from __future__ import annotations

import json

import aiomqtt
from loguru import logger

from ignis.nilm.config import settings

from .contract import (
    DISAGGREGATION_TOPIC,
    META_TOPIC,
    DisaggregationSnapshot,
    ModelMeta,
)

_QOS = 1
_RETAIN = True


def _dumps(payload: dict) -> str:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


class MqttPublisher:
    """Publishes contract payloads to the broker (portfolio + external subs)."""

    def __init__(self, client: aiomqtt.Client) -> None:
        self._client = client

    async def publish_snapshot(self, snapshot: DisaggregationSnapshot) -> None:
        """Publish one grouped inference cycle (spec 6.4)."""
        await self._publish(DISAGGREGATION_TOPIC, snapshot.payload())

    async def publish_meta(self, meta: ModelMeta) -> None:
        await self._publish(META_TOPIC, meta.payload())

    async def _publish(self, topic: str, payload: dict) -> None:
        await self._client.publish(topic, _dumps(payload), qos=_QOS, retain=_RETAIN)
        logger.debug("published {} {}", topic, payload)


def make_client() -> aiomqtt.Client:
    """Broker client using the shared MQTT settings."""
    return aiomqtt.Client(
        hostname=settings.mqtt_host,
        port=settings.mqtt_port,
        username=settings.mqtt_username,
        password=settings.mqtt_password,
    )
