"""Home Assistant MQTT discovery bridge.

Publishes discovery configs so HA auto-creates an Ignis device with one power
sensor per appliance. All sensors read the SAME grouped ``nilm/disaggregation``
topic (spec 6.4); each extracts its value via
``value_template: {{ value_json.appliances.<key> }}``. This avoids a second
output surface -- one published snapshot feeds both portfolio and HA.

This is the lightweight HA surface; the full custom_components/ignis
integration + LitElement cards remain a separate TODO (git 457cc1b).

Discovery topic (default HA prefix)::

    homeassistant/sensor/ignis_<appliance>_power/config   retained
"""

from __future__ import annotations

import json

import aiomqtt
from loguru import logger

from ignis.nilm.config import settings

from .contract import DISAGGREGATION_TOPIC

# HA's MQTT discovery prefix. Distinct from the statestream prefix used for
# ingestion -- this is the discovery tree, conventionally "homeassistant".
DISCOVERY_PREFIX = "homeassistant"
DEVICE_ID = "ignis"
DEVICE = {
    "identifiers": [DEVICE_ID],
    "name": "Ignis NILM",
    "manufacturer": "Ignis",
    "model": "Seq2Point",
}


def _discovery_topic(appliance: str) -> str:
    return f"{DISCOVERY_PREFIX}/sensor/{DEVICE_ID}_{appliance}_power/config"


def _discovery_config(appliance: str) -> dict:
    return {
        "name": f"{appliance} (NILM)",
        "unique_id": f"{DEVICE_ID}_{appliance}_power",
        "state_topic": DISAGGREGATION_TOPIC,
        "value_template": f"{{{{ value_json.appliances.{appliance} }}}}",
        "unit_of_measurement": "W",
        "device_class": "power",
        "state_class": "measurement",
        "device": DEVICE,
    }


async def publish_discovery(client: aiomqtt.Client, appliances: list[str] | None = None) -> None:
    """Publish retained discovery configs for each appliance."""
    apps = appliances if appliances is not None else settings.nilm_appliances
    for app in apps:
        topic = _discovery_topic(app)
        await client.publish(topic, json.dumps(_discovery_config(app)), qos=1, retain=True)
        logger.info("HA discovery published: {}", topic)
