# Deploy Ignis

Two environments, one database.

- **Prod = the Raspberry Pi** (also runs Home Assistant + mosquitto + the GitHub
  runner). Runs: TimescaleDB + `ha_ingest` + `publish` (champion inference).
- **Dev = your Mac**. Training runs natively here (CPU/Metal). Dev reads the
  **prod** TimescaleDB over the LAN, so it always has fresh HA data.

Training is never containerised. The Pi does inference only.

## Architecture

```
HA + mosquitto (Pi) ──► ha_ingest (Pi) ──► TimescaleDB (Pi, LAN-exposed)
                                                │
            dev Mac (training, reads prod DB) ──┘
                                                │
   champion.keras ──(make ship)──► publish (Pi) ──► MQTT nilm/* ──► portfolio
```

## One-time Pi provisioning

```bash
# On the Pi
git clone <repo-url> ~/ignis
cd ~/ignis
cp .env.example .env        # then edit: DB password, MQTT creds, DB_BIND_ADDR
mkdir -p models             # champion.keras lands here via `make ship`
```

Key `.env` values on the Pi:

```
LOCAL_DB_HOST=timescaledb            # service name inside compose
LOCAL_DB_PASSWORD=<strong-password>  # required: DB is LAN-exposed
DB_BIND_ADDR=<pi-lan-ip>             # your Pi's LAN IP -- LAN only, NOT 0.0.0.0
MQTT_HOST=<pi-lan-ip>                # the broker
MQTT_USERNAME=... / MQTT_PASSWORD=...
```

The GitHub **self-hosted runner** must already be registered on the Pi
(labels `self-hosted, linux, arm64`).

## Dev (Mac) pointing at the prod DB

In the Mac `.env`:

```
LOCAL_DB_HOST=<pi-lan-ip>            # read prod data directly
LOCAL_DB_PASSWORD=<same-as-pi>
```

Then `make train` / `make eval` use fresh prod HA data; no local backfill.

## CI/CD

- **CI** (`.github/workflows/ci.yml`) on `develop` + PRs: ruff lint+format,
  pytest, docker build (both images). Runs on `ubuntu-latest`.
- **CD** (`.github/workflows/cd.yml`) on `main`: the Pi runner pulls `main` into
  `~/ignis` and runs `make deploy` (`docker compose --profile prod up -d --build`).

Branch flow: work on `develop` → PR → merge to `main` → auto-deploy.

## Shipping a model

Training produces a champion on the Mac. Push it to the Pi:

```bash
make eval        # promotes the best challenger to champion.keras
make ship        # rsync champion.* to the Pi models volume
```

`publish` reloads it on its next cycle (it mounts the models volume read-only).
