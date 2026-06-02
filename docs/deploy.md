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

- **CI** (`.github/workflows/ci.yml`) on push to `develop` + `main`, and on PRs
  to either: ruff lint+format, pytest, docker build (both images). Runs on
  `ubuntu-latest`.
- **CD** (`.github/workflows/cd.yml`) is **gated on CI**: it triggers via
  `workflow_run` only when CI completes **successfully on `main`**, then the Pi
  runner pulls `main` into `~/ignis` and runs `make deploy`
  (`docker compose --profile prod up -d --build`). A red CI never deploys.
  Manual `workflow_dispatch` bypasses the gate on purpose (emergency deploy).

`main` is **protected**: PR required before merge (no direct push), both CI
checks (`Ruff + pytest`, `Docker build (ingest + worker)`) required and strict
(branch up to date). Admins are **not** enforced — the owner can force-merge in
an emergency.

Branch flow: work on `develop` → push → PR to `main` → CI green → merge → CI on
`main` → auto-deploy. `make ship` (model `.keras` → Pi) is out of band (rsync,
no git), so shipping a model does not go through this flow.

## Shipping a model (from the Mac)

Fast dev iteration trains on the Mac, then ships:

```bash
make eval        # promotes the best challenger to champion.keras
make ship        # rsync champion.* to the Pi models volume
```

`publish` reloads it on its next cycle (it mounts the models volume read-only).

## On-device nightly retrain (autonomous, no Mac)

The Pi can retrain itself overnight — benchmarked at ~68s/epoch (Pi 4, RAM
stable, no OOM), so a full ~40min pass is fine off-peak. One pass = train a
challenger → eval vs HA → promote only if better → prune.

```bash
make retrain     # = docker compose --profile train run --rm train
```

The `train` service mounts the models volume READ-WRITE (publish stays
read-only) and exits after one pass. `publish` picks up a new champion on its
next cycle, no restart.

Schedule it with cron on the Pi (a flock guard prevents overlap):

```cron
17 3 * * *  /home/<user>/ignis/scripts/cron-retrain.sh >> /home/<user>/ignis/retrain.log 2>&1
```

This is the semi-open loop closed on-device: detect drift / retrain / ship the
winner all on the Pi. The Mac stays optional (fast experimentation only).
