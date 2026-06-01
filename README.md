# Ignis

[![CI](https://github.com/rsaikali/ignis/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/rsaikali/ignis/actions/workflows/ci.yml)
[![CD](https://github.com/rsaikali/ignis/actions/workflows/cd.yml/badge.svg)](https://github.com/rsaikali/ignis/actions/workflows/cd.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](pyproject.toml)

**NILM / Home Assistant lab.** Ignis guesses **per-appliance power** from a
single whole-house meter (a Linky smart meter), and checks itself against the
**real** per-appliance consumption that smart plugs already report to Home
Assistant. HA is both the input (aggregate) and the ground truth — so the
labels are free and the model can be trained *self-supervised* and graded
honestly (per-appliance F1 + energy error).

Deep-learning engine: Seq2Point multi-output (GRU/LSTM + attention,
TensorFlow/Keras). The full loop runs unattended:

```
HA ─► ingest ─► TimescaleDB ─┬─► publish  (MQTT: per-appliance + scores)
                             ├─► backend  (HTTP API: history + truth)
                             └─► nightly retrain ─► eval vs HA ─► promote champion
```

> Honest status: with a few weeks of data, high-duty appliances are detected
> (TV ~0.70 F1, PC ~0.42) while rare ones (oven, dishwasher) are still
> data-limited and score ~0. The why — and how it's tackled — is documented in
> [`docs/nilm-imbalance.md`](docs/nilm-imbalance.md). The nightly retrain keeps
> accruing rare-class labels over time.

## What you need

- **Home Assistant** with:
  - a **whole-house aggregate** sensor — apparent power in VA (e.g. a Linky via
    ZLinky/Zigbee, or any `sensor.*` reporting live VA).
  - **per-appliance smart plugs** that report power in W (the ground truth).
    Out of the box Ignis expects **Meross mss315** entity naming
    (`sensor.<name>_mss315_power_w_main_channel` + a `switch.<name>_mss315_main_channel`).
    Other plugs work too — you adjust the entity mapping (see *Other plugs*).
  - the **`mqtt_statestream`** integration enabled (HA republishes states to MQTT):
    ```yaml
    # HA configuration.yaml
    mqtt_statestream:
      base_topic: statestream
      publish_attributes: false
    ```
  - an **MQTT broker** (e.g. Mosquitto) reachable from where Ignis runs.
- **Docker + Docker Compose** (for the prod stack).
- For training: **Python 3.12 + uv**, native (CPU; macOS Metal optional). Training
  is off-device by design but the Pi can also retrain itself overnight.

## Configure (`.env`)

Copy `.env.example` to `.env` and set, at minimum:

| Key | What it is |
|-----|------------|
| `MQTT_HOST` / `MQTT_PORT` | your broker (the HA box, usually) |
| `MQTT_USERNAME` / `MQTT_PASSWORD` | broker creds (if `allow_anonymous false`) |
| `MQTT_STATESTREAM_PREFIX` | must match HA's `base_topic` (default `statestream`) |
| `HA_AGGREGATE_ENTITY` | your aggregate VA sensor, e.g. `sensor.puissance_generale` |
| `NILM_APPLIANCES` | comma-separated appliance keys you have plugs for, e.g. `four,lave_linge,television` |
| `MEROSS_DEVICE_TAG` | the device tag in your plug entity ids (default `mss315`) |
| `LOCAL_DB_*` | TimescaleDB credentials |
| `HA_SSH_USER` / `HA_SSH_HOST` | SSH to the HA box, only for history backfill |

**How the appliance entities are derived.** For each key in `NILM_APPLIANCES`,
Ignis subscribes to (and the truth comes from):

```
sensor.<key>_<MEROSS_DEVICE_TAG>_power_w_main_channel    # W  -> the label
sensor.<key>_<MEROSS_DEVICE_TAG>_energy_kwh_main_channel # cumulative kWh
switch.<key>_<MEROSS_DEVICE_TAG>_main_channel            # ON/OFF truth
```

So `NILM_APPLIANCES=four` + `MEROSS_DEVICE_TAG=mss315` →
`sensor.four_mss315_power_w_main_channel`. Name your appliances so the keys line
up with your HA entity ids.

### Other plugs (not Meross)

If your smart plugs use a different entity-id pattern, edit the builders in
`src/ignis/ha_ingest/entities.py` (`_meross_sensor` / `_meross_switch`) to match
your `sensor.*` / `switch.*` names. Everything downstream (aggregate VA + a W
truth + an ON/OFF truth per appliance) stays the same.

## Run

### Dev (your machine)

```bash
uv venv --python 3.12
uv pip install -e ".[dev,engine]"
cp .env.example .env        # edit it
make test
make backfill               # replay HA history into TimescaleDB (needs SSH to HA)
make train                  # train a model (CPU/Metal)
make eval                   # score vs HA; auto-promotes the best to champion.keras
make admin                  # admin console at http://localhost:8001
```

### Prod (a Raspberry Pi, alongside HA)

`make deploy` brings up four services (TimescaleDB, ingest, publish, backend).
A nightly cron retrains on-device. Full guide: [`docs/deploy.md`](docs/deploy.md).

## Outputs

- **MQTT** (retained): `nilm/disaggregation` (per-appliance W snapshot) and
  `nilm/_meta/model` (active model + per-appliance F1/energy-error vs HA).
- **HTTP API**: `/api/models/history` (accuracy over time), `/api/truth/recent`
  (live per-plug truth), `/api/models`, `/api/health`.
- See [`docs/portfolio-contract.md`](docs/portfolio-contract.md) for the consumer
  contract.

*(A Home Assistant custom component + Lovelace cards are planned but not shipped
yet.)*

## Docs

- [`docs/nilm-imbalance.md`](docs/nilm-imbalance.md) — the core ML challenge
  (why MAE lies, the predict-zero collapse, the fixes). Written to teach.
- [`docs/deploy.md`](docs/deploy.md) — prod architecture, Pi provisioning, CI/CD,
  on-device retrain.
- [`docs/portfolio-contract.md`](docs/portfolio-contract.md) — what Ignis exposes.

## License

MIT — see [LICENSE](LICENSE).
