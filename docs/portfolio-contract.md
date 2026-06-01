# Ignis → Portfolio contract

What Ignis exposes for `portfolio.saikali.fr` to consume. The portfolio's job
is **not** to show a finished product — it is to prove a rigorous ML research
approach (self-supervised NILM, HA as free ground truth, an honest accuracy
gate, and a model that improves over time).

Three surfaces, two transports.

## Transports

### 1. MQTT (live) — already published, retained

The Pi's `publish` service emits every ~60s, **retained** (a new subscriber
gets the latest immediately):

- `nilm/disaggregation`
  ```json
  { "ts": "<ISO8601>", "total_w": <float>,
    "appliances": { "<key>": <power_w>, ... } }
  ```
- `nilm/_meta/model`  (active champion + honest scores vs HA)
  ```json
  { "version": "ignis_gru_<ts>", "model_type": "gru",
    "trained_at": "<ISO8601>", "appliances": ["four", ...],
    "metrics": { "<key>": { "state_f1": <0..1>,
                            "energy_error": <float>,
                            "passes_gate": 0|1 }, ... } }
  ```

Broker: the Pi's mosquitto (LAN), auth required (`allow_anonymous false`).

### 2. HTTP API (history + truth) — Ignis `backend` (FastAPI)

For data MQTT-retained can't carry (history is a series, not a last value):

- `GET /api/health` → `{ ok, models, ... }`
- `GET /api/models` → current registry (name, trained_at, gate pass/fail).
- `GET /api/models/history` → **one point per nightly retrain** (the evolution
  curve):
  ```json
  [ { "model": "ignis_gru_<ts>", "trained_at": "<ISO8601>",
      "train_days": 30, "mean_f1": <float>, "gate_passes": <int>,
      "appliances": { "<key>": { "state_f1": <0..1>,
                                 "energy_error": <float>,
                                 "labeled_hours": <float> }, ... } }, ... ]
  ```
- `GET /api/truth/recent?window=15m` → per-appliance HA truth (smart-plug
  power), to draw NILM-vs-real side by side:
  ```json
  { "window_seconds": 900,
    "appliances": { "<key>": { "power_w": <float>, "on": bool } } }
  ```

## The three surfaces

### A. Hero — "NILM vs HA" (live diff)
Sources: MQTT `nilm/disaggregation` + API `/api/truth/recent`.
Show: Linky aggregate → NILM disaggregation **vs** smart-plug ground truth, per
appliance, with the gap. This is the project's central idea: guess per-appliance
power from one meter, verified against truth.

### B. Scorecard — honest + pedagogical
Source: MQTT `nilm/_meta/model`.
Per-appliance table: state F1, energy error, gate pass (>=0.8 / <=15%).
Show **every** appliance including the 0.0s (tv ~0.70, pc ~0.42,
four/lave_vaisselle/smart_plug 0.0). Add a "why" panel: rare appliances are
data-limited (~2% ON), window-length and class-imbalance effects — link
`docs/nilm-imbalance.md`. Showing failures *with the analysis* is the rigorous
signal, not a weakness.

### C. Evolution — research alive
Source: API `/api/models/history`.
A curve of mean F1 (and per-appliance) across nightly retrains. Narrative:
self-supervised — the model improves on its own as appliances get used (more
rare-class labels accrue); the cron retrains every night. Starts flat (one
point), grows each night. This is the surface that **proves the loop**.

## Honesty policy

Total + pedagogical. Display the real scores including zeros, each with the
reason. A recruiter tells "hiding failures" from "I understand why a 11h-active
oven is hard and here's the plan". The latter signals maturity.

## What Ignis must add for this contract

- **Persist each retrain run** as a small row (version, date, scores) so the
  evolution curve has long history — NOT just the 5 retained `.keras`/JSON
  sidecars (pruned). A `model_runs` table in TimescaleDB is the right home.
- **Backend API** endpoints above (`backend/` FastAPI already exists with the
  registry; add `/api/models/history` from `model_runs` and `/api/truth/recent`
  from the `appliance_power`/`appliance_onoff` views).

Until then, the portfolio can ship A + B from MQTT alone (already live) and add
C once the history API is up.
