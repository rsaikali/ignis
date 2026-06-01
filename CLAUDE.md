# CLAUDE.md — Ignis

Guidance for Claude Code working in this repo. Solo project, owner: Roland Saïkali.

## What Ignis is

**The NILM/HA lab.** A single consolidated repo that:
- runs a **NILM engine** (Non-Intrusive Load Monitoring): per-appliance electrical disaggregation from a Linky smart-meter aggregate, via a deep-learning Seq2Point multi-output model (TensorFlow/Keras, LSTM/GRU + attention);
- pulls **ground truth from Home Assistant** (per-appliance smart-plug sensors) + the aggregate, computes the **HA-vs-NILM diff** and **drift** per appliance;
- runs a **drift-triggered retrain** loop;
- exposes results through **two output surfaces** (see below).

This is a reboot: the old Ignis (local-LLM-on-Raspberry-Pi + Lovelace card) was dropped. Its `.git` history is kept — the old **HA-integration pattern is reusable** (see "Recovering the HA pattern").

## History / how we got here (2026-05-31)

- Was going to be 3 repos (linkya engine / ignis lab / portfolio). **Consolidated to one** — premature separation for a solo project. The NILM engine was **harvested from `../linkya/nilm-service/src`** into `nilm/`.
- **`../linkya` is now a FROZEN reference** (NILM knowledge source). Do NOT edit or deploy it. Read it for context only.
- Old LLM-on-RPi worktree deleted; git kept at commit `457cc1b`.

## Architecture

All code lives under `src/ignis/` (src layout, `ignis` namespace). Import as
`from ignis.nilm.config import settings`; run modules as `python -m ignis.<pkg>`.

```
src/ignis/
├── nilm/         # harvested engine (verbatim). NOTE nested nilm/nilm/ sub-package.
├── ha_ingest/    # HA -> store: aggregate + per-appliance truth + backfill + compat views
├── training/     # self-supervised dataset builder + native (CPU/Metal) trainer
├── eval/         # NILM vs HA metrics (F1/energy) + drift + champion/challenger promote
├── publish/      # MQTT contract (spec 6.4) + live inference + HA discovery
└── backend/      # minimal admin console (FastAPI)
```

Still TODO (not yet built): `custom_components/ignis/` HA integration + LitElement
cards (skeleton kept locally in `.ha-pattern-ref/`, gitignored; recover/rewrite
for NILM). Engine internal imports stay relative (`from .nilm.models import ...`).

### Two output surfaces (Ignis has multiple consumers, not one)

1. **MQTT** -> the portfolio site (`../portfolio.saikali.fr`) **and any external subscriber**.
2. **HA integration** (`custom_components/ignis/` + LitElement cards) -> view the lab inside Home Assistant (diff, drift, per-appliance accuracy) and other HA views.

Symmetric: Ignis **reads** HA (ground truth) and **writes** HA (cards/entities).

### Data source = Home Assistant

No MySQL, no own MQTT broker, no TIC/TeleInfo client (all of that died with Linkya's old ingestion). `ha_ingest` reads from HA. Open question: **MQTT-push vs HA history API-pull** (pull is simpler; daily aggregates are likely enough). Truth is stored as a table in the lab's own TimescaleDB (single store — don't invent a second one).

## Engine (`nilm/`)

Harvested verbatim from Linkya, ~3825 LOC, TensorFlow/Keras. Key files:
- `nilm/seq2point_nilm.py` (721) — manager: `train_all_appliances`, `disaggregate`, signature filtering
- `nilm/tasks.py` (600) — Celery train/detect tasks (`import tensorflow as tf` at call time)
- `nilm/database.py` (409) — NILM TimescaleDB tables
- `nilm/config.py` (76) — plain `class Settings` reading `os.getenv` (NOT pydantic). Defaults include `local_db_name="linkya_db"`, `nilm_model_path="/app/models"`, Celery redis URLs, NILM window/seq/epochs/lr, `use_gpu`, `nilm_model_type="gru"`. `database_url` property builds the postgres DSN.
- `nilm/nilm/models/multioutput_model.py` (589) — the Keras Seq2Point multi-output model
- `nilm/nilm/detectors/` — change_point_detector (427), state_detector (128, ON/OFF)
- `nilm/nilm/` — morphology, preprocessing, callbacks, losses, layers, utils

**Nesting note**: harvest produced `nilm/nilm/` (a sub-package). That's intentional in the original code (imports are `from .nilm.models import ...`), so it works as-is. Flatten later only if it bothers you.

Models are saved as `.keras` + `.metadata.json`.

### Adaptation still needed on the harvested engine
- `config.py`: no MySQL leftovers (the MySQL/sync lived in the dropped sync-service, not here). But it's a plain `os.getenv` class → consider porting to `pydantic-settings` for workspace consistency, and repoint the `local_db_name="linkya_db"` default to the lab's.
- Engine uses stdlib `logging` → migrate to **loguru** as you touch files.
- `tasks.py` assumes GPU/`.keras` paths under `/app/models` — fine for the Pi inference container; training stays native (see below).
- No git remote configured yet.

## Training is OFF-device

The Seq2Point model is real deep learning → wants a GPU. Owner is on **Mac (Metal)**; **Metal is not exposed to Docker containers**. So:
- **Training runs NATIVE macOS** (`tensorflow-metal`, a `uv` venv — NOT a Docker service), or on a real CUDA box if available.
- **The Raspberry Pi runs inference only** (CPU, loads the `.keras` artifact).
- Workflow: train native here → export `.keras` → `rsync`/`scp` to the Pi's models volume.

**Retrain loop is semi-open** (not fully auto on the Pi): Pi `eval` detects drift → emits a retrain request → training runs here → champion/challenger compare here → ship only the winner → Pi reloads → `eval` re-validates.

Retrain trigger (drift-triggered): rolling 7-day energy-error vs HA per appliance > threshold for 3 consecutive days AND >= X new labeled activation hours since last train AND cooldown >= 7 days.

**Acceptance gate** (model good enough): per-appliance state F1 >= 0.8 **and** energy error <= 10-15% vs HA. This is the target — NOT a signature count. Signatures are the means; the validated diff vs HA is the goal.

**The core ML challenge — class imbalance — is documented in `docs/nilm-imbalance.md`** (why MAE lies, the predict-zero collapse, the three levers: balance / loss / scaling, and a results journal). Read it before touching training.

## Recovering the old HA-integration pattern

The reboot deleted the worktree but kept git. The old HA integration + LitElement card are a good skeleton (config_flow, coordinator, sensor, services, card) — the LLM bits go, the plumbing stays:

```bash
git show 457cc1b:custom_components/ignis/config_flow.py
git show 457cc1b:custom_components/ignis/coordinator.py
git show 457cc1b:custom_components/ignis/sensor.py
git show 457cc1b:frontend/ignis-card/src/ignis-card.js
git ls-tree -r --name-only 457cc1b   # full file list
```

## Conventions (inherited workspace rules)

- Code/comments/docstrings/logs: **English**. Frontend UI text: **French** (single-lang HA cards).
- Logs: **loguru** only — no `print()`, no stdlib `logging`. (Harvested engine uses stdlib `logging` → migrate to loguru when touched.)
- Python 3.12+, **uv** for deps/venv, **ruff** for lint+format.
- pytest for tests; new non-trivial code ships with tests.
- `.env.example` = source of truth for params; keep `.env` synced.
- Scripts/migrations idempotent.
- Docker: `docker compose`, services split, dev override. Remember: training is NOT a container (Metal).
- No emojis in code or logs.
- Commits: Conventional Commits. Propose a commit after a meaningful chunk.

## Scope discipline

Owner expands features before finishing the previous one — **flag scope creep**. Ignis already holds a lot (engine + ingest + eval + 2 output surfaces + UI). Keep tight module boundaries. Do not let Ignis absorb the portfolio widget or re-grow into the dropped LLM project.

## Where we are (2026-06-01)

**LIVE in production on the Pi.** The full loop runs autonomously:
`HA → ha_ingest → TimescaleDB → publish (MQTT) + nightly on-device retrain → champion → portfolio API`.

- Public repo `github.com/rsaikali/ignis` (MIT, src layout `src/ignis/`), CI/CD green.
- Prod = 4 compose services on the Pi (timescaledb, ha_ingest, publish, backend), profile `prod`.
- On-device retrain: cron `17 3 * * *` (train→eval→promote→log model_runs). Pi 4 ~40min/pass. Mac optional (fast dev).
- Portfolio interface ready (`docs/portfolio-contract.md`): MQTT live+scores + history/truth API.
- Deploy + ops details: `docs/deploy.md`. The core ML story: `docs/nilm-imbalance.md`.

**Open / next:**
1. **Accuracy** — only tv (~0.70) and pc (~0.42) clear-ish the gate; four/lave_vaisselle/smart_plug are data-limited (~2% ON over 30d). The nightly retrain accrues rare-class labels over time; revisit per-cycle val split / longer windows if needed.
2. **Portfolio** — build the 3 surfaces from the contract (separate repo, `../portfolio.saikali.fr`).
3. **Dette**: TF ARM build ~11min/deploy (→ ghcr pre-build); `make ship` Pi defaults (pass overrides); ROTATE the MQTT password (it transited a chat).
4. Later: recover HA integration + cards (skeleton in local `.ha-pattern-ref/`, gitignored; from old git `457cc1b`).

## Related repos in the workspace

- `../linkya` — frozen NILM reference (where the engine came from; has the Makefile MLOps targets to mirror: train/detect/model-compare/model-rollback/signatures-stats).
- `../portfolio.saikali.fr` — public showcase, consumes Ignis MQTT; case study "HA vs NILM diff". Has a `DisaggregationSource` Protocol (`app/domain/source.py`) — an MQTT/Ignis adapter fits that seam. Note: its CLAUDE.md still says source = HA, but specs section 6.4 target = MQTT from the NILM model. Drift to resolve.

<!-- rtk-instructions auto-appended by hook below; leave it -->
