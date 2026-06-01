# Ignis -- NILM/HA lab. Keep this simple (workspace rule).
.PHONY: help install lint test fmt up down logs ingest clean \
        train-deps train metal-deps backfill admin eval ship deploy retrain

UV := uv
COMPOSE := docker compose
MODELS ?= models
# Pi deploy target for `make ship` (override in your env / shell).
PI ?= pi@raspberrypi.local
PI_MODELS ?= /home/pi/ignis/models

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "} {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Create venv and install dev + engine deps
	$(UV) venv --python 3.12
	$(UV) pip install -e ".[dev,engine]"

lint: ## Ruff lint + format check
	$(UV) run ruff check .
	$(UV) run ruff format --check .

fmt: ## Ruff format (write)
	$(UV) run ruff format .

test: ## Run the test suite
	$(UV) run pytest -q

up: ## Start the prod-ish stack (timescaledb + ha_ingest)
	$(COMPOSE) up -d --build

down: ## Stop the stack
	$(COMPOSE) down

logs: ## Tail ha_ingest logs
	$(COMPOSE) logs -f ha_ingest

ingest: ## Run ha_ingest locally (no container; needs .env + reachable broker/DB)
	$(UV) run python -m ignis.ha_ingest

# --- training runs NATIVE, never in a container ---------------------------
DAYS ?= 30
EVAL_DAYS ?= 3

train-deps: ## Install engine + training extras (TensorFlow CPU) natively
	$(UV) pip install -e ".[engine,training]"

train: train-deps ## Native training (CPU) over the last $(DAYS) days (default 30)
	$(UV) run python -m ignis.training --days $(DAYS)

eval: ## Eval the latest model vs HA, auto-promote to champion if better
	$(UV) run python -m ignis.eval --days $(EVAL_DAYS)

ship: ## rsync champion.keras (+ sidecars) to the Pi models volume
	@test -f $(MODELS)/champion.keras || { echo "No champion yet. Run make eval."; exit 1; }
	rsync -av $(MODELS)/champion.* $(PI):$(PI_MODELS)/

deploy: ## Prod deploy on the Pi (run by CD): build + up db/ingest/publish
	$(COMPOSE) --profile prod up -d --build
	$(COMPOSE) ps

retrain: ## On-device retrain pass (train -> eval -> auto-promote). For cron.
	$(COMPOSE) --profile train run --rm train

metal-deps: ## OPT-IN: add the macOS Metal GPU plugin (only if TF-compatible)
	@echo "WARN: tensorflow-metal lags TF releases and may break import."
	@echo "Pin a compatible tensorflow first if import fails, then:"
	$(UV) pip install -e ".[engine,training,metal]"

backfill: ## Replay HA recorder history into ha_samples ($(DAYS) days)
	$(UV) run python -m ignis.ha_ingest.backfill --days $(DAYS)

admin: ## Run the admin console (FastAPI) at http://localhost:8001
	$(UV) pip install -e ".[backend]"
	$(UV) run uvicorn ignis.backend.app:app --host 0.0.0.0 --port 8001 --reload

clean: ## Remove caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
