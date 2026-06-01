# Ignis

NILM/HA lab: per-appliance electrical disaggregation (Seq2Point, TensorFlow/Keras)
from a Linky smart-meter aggregate, with Home Assistant as both ground-truth
source and output surface.

Pipeline: HA → `ha_ingest` (MQTT statestream) → TimescaleDB → native training
(self-supervised from Meross per-appliance truth) → champion/challenger eval vs
HA (state F1 + energy error) → `publish` (MQTT + honest scores) → portfolio.

## Layout

All code lives under `src/ignis/`:

- `ignis.nilm` — Seq2Point multi-output engine (GRU/LSTM + attention; harvested
  from Linkya). Nested `nilm/nilm/` is the original sub-package.
- `ignis.ha_ingest` — HA → TimescaleDB ingestion, history backfill, compat views.
- `ignis.training` — self-supervised dataset builder + native trainer.
- `ignis.eval` — HA-vs-NILM metrics, drift, champion/challenger promotion.
- `ignis.publish` — MQTT contract (spec 6.4), live inference, HA discovery.
- `ignis.backend` — minimal admin console (FastAPI).

Docs: `docs/nilm-imbalance.md` (the core ML challenge), `docs/deploy.md`.

## Quickstart (dev)

```bash
uv venv --python 3.12
uv pip install -e ".[dev,engine]"
cp .env.example .env        # adjust to your HA / broker / DB
make test
```

Common targets: `make backfill` (replay HA history), `make train`, `make eval`
(auto-promotes champion), `make ship` (rsync champion to the Pi), `make admin`
(console at :8001). See `make help`.

Training runs natively (CPU/Metal); the Raspberry Pi runs inference only.
