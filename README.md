# Ignis

NILM/HA lab: per-appliance electrical disaggregation (Seq2Point, TensorFlow/Keras)
from a Linky aggregate, with Home Assistant as ground-truth source and output surface.

See `CLAUDE.md` for architecture, conventions and the current roadmap.

## Modules

- `nilm/` — disaggregation engine (harvested from Linky).
- `ha_ingest/` — HA → TimescaleDB ingestion (MQTT-push via `mqtt_statestream`).
- `eval/` — NILM vs HA diff/drift → retrain requests. *(TODO)*
- `publish/` — MQTT + HA entity outputs. *(TODO)*

## Quickstart (dev)

```bash
uv venv --python 3.12
uv pip install -e ".[dev,engine]"
cp .env.example .env        # adjust to your setup
make test
```

Training runs natively on macOS (Metal); the Raspberry Pi runs inference only.
See `CLAUDE.md` → "Training is OFF-device".
