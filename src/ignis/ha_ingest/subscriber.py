"""MQTT statestream subscriber -> TimescaleDB.

Subscribes to the HA ``mqtt_statestream`` topic tree, filters to the entities
we care about (aggregate + Meross plugs), parses the raw payload and stores it.
"""

from __future__ import annotations

import asyncio

import aiomqtt
from loguru import logger

from ignis.nilm.config import settings

from .entities import parse_value, subscribe_filters, topic_index
from .store import Store

_RECONNECT_SECONDS = 5


async def run() -> None:
    """Run the subscriber loop forever, reconnecting on failure."""
    prefix = settings.mqtt_statestream_prefix
    index = topic_index(prefix)
    filters = subscribe_filters(prefix)
    logger.info(
        "ha_ingest: {} entities, {} topic filters, broker {}:{}",
        len(index),
        len(filters),
        settings.mqtt_host,
        settings.mqtt_port,
    )

    store = Store()
    await store.connect()
    try:
        while True:
            try:
                await _session(store, index, filters)
            except aiomqtt.MqttError as exc:
                logger.warning("MQTT error: {} -- reconnecting in {}s", exc, _RECONNECT_SECONDS)
                await asyncio.sleep(_RECONNECT_SECONDS)
    finally:
        await store.close()


async def _session(store: Store, index, filters) -> None:
    async with aiomqtt.Client(
        hostname=settings.mqtt_host,
        port=settings.mqtt_port,
        username=settings.mqtt_username,
        password=settings.mqtt_password,
    ) as client:
        for f in filters:
            await client.subscribe(f)
        logger.info("Subscribed; ingesting")
        async for message in client.messages:
            topic = str(message.topic)
            spec = index.get(topic)
            if spec is None:
                continue  # entity outside our whitelist
            raw = (
                message.payload.decode(errors="replace") if isinstance(message.payload, bytes) else str(message.payload)
            )
            value = parse_value(spec, raw)
            await store.insert(spec.entity_id, spec.appliance, spec.kind, value, raw)
