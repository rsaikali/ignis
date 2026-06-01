"""Ignis NILM/lab configuration.

Ported from the harvested Linky engine (plain ``os.getenv`` class) to
``pydantic-settings`` for workspace consistency. The engine imports
``settings`` from here; all engine-used fields are preserved verbatim.
New blocks (MQTT ingestion, NILM targets) are additive.

Values come from the environment / a root ``.env`` (see ``.env.example``).
"""

from functools import lru_cache
from typing import Annotated

from pydantic import computed_field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """Single source of config for the engine, ingestion and eval."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- TimescaleDB (single lab store) --------------------------------------
    local_db_host: str = "localhost"
    local_db_port: int = 5432
    local_db_name: str = "ignis"
    local_db_user: str = "ignis"
    local_db_password: str = "ignis"

    # --- Redis / Celery (engine train/detect tasks) --------------------------
    # broker/result fall back to redis_url when unset (resolved post-init).
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str | None = None
    celery_result_backend: str | None = None

    # --- NILM scheduling / detection (engine) --------------------------------
    nilm_training_interval_hours: int = 24
    nilm_detection_interval_minutes: int = 5
    # Analysed window: None = all history, else a number of hours.
    nilm_detection_period_hours: int | None = None
    nilm_window_size_minutes: int = 10
    nilm_min_power_threshold: int = 15
    nilm_min_duration_seconds: int = 30

    # --- NILM model (engine) -------------------------------------------------
    # Default is repo-relative (native training writes here; gitignored via
    # /models/). The Pi inference container overrides to /app/models in compose.
    nilm_model_path: str = "models"
    # Window length in grid points. 599 was Linkya's 1 Hz value (= 10 min). On
    # our 30 s grid that is 5 h -- far longer than an appliance cycle, so short
    # cycles are drowned (see docs/nilm-imbalance.md). 99 pts x 30 s = ~50 min,
    # matched to a cycle. Must be odd (centre point).
    nilm_sequence_length: int | None = 99
    nilm_batch_size: int = 32
    nilm_epochs: int = 50
    nilm_learning_rate: float = 0.001
    nilm_validation_split: float = 0.2
    nilm_model_type: str = "gru"
    nilm_detect_states: bool = True
    # "true" / "false" / "auto" (None) -- consumed by training scripts.
    use_gpu: str | None = None

    # --- NILM targets (appliances with Meross W ground truth) ----------------
    # 6 Meross mss315 plugs verified on the prod HA (2026-05-31).
    # Comma-separated env (NILM_APPLIANCES) accepted thanks to env parsing.
    # NoDecode: stop pydantic-settings from JSON-parsing the env string so our
    # CSV validator runs instead (NILM_APPLIANCES=four,lave_linge,...).
    nilm_appliances: Annotated[list[str], NoDecode] = [
        "four",
        "lave_linge",
        "lave_vaisselle",
        "pc",
        "television",
        "smart_plug",
    ]

    # --- Home Assistant ingestion (MQTT-push via mqtt_statestream) -----------
    # Host of the HA box / MQTT broker. Set MQTT_HOST in .env to your own.
    mqtt_host: str = "localhost"
    mqtt_port: int = 1883
    mqtt_username: str | None = None
    mqtt_password: str | None = None
    # HA mqtt_statestream base_topic: HA republishes states under
    # ``<prefix>/<domain>/<object_id>/state``. Keep this DISTINCT from the
    # MQTT discovery prefix (also "homeassistant") to avoid topic-tree clash.
    mqtt_statestream_prefix: str = "statestream"

    # SSH access to the HA box, used by the history backfill (runs sqlite3
    # inside the HA container). Defaults derive from the MQTT host. Set in .env.
    ha_ssh_user: str = "pi"
    ha_ssh_host: str | None = None  # falls back to mqtt_host when unset

    # Aggregate NILM input: Linky apparent power (VA), ~7s native sampling.
    ha_aggregate_entity: str = "sensor.puissance_generale"
    # Meross device tag used to build per-appliance entity ids (see ha_ingest).
    meross_device_tag: str = "mss315"

    # Common resample grid (Meross report ~30s on-change; agg downsampled).
    ingest_grid_seconds: int = 30

    @field_validator("nilm_appliances", mode="before")
    @classmethod
    def _split_appliances(cls, v: object) -> object:
        """Accept a comma-separated env string as well as a real list."""
        if isinstance(v, str):
            return [a.strip() for a in v.split(",") if a.strip()]
        return v

    @model_validator(mode="after")
    def _resolve_defaults(self) -> "Settings":
        """Fill unset Celery/SSH-host values from their canonical source."""
        if self.celery_broker_url is None:
            self.celery_broker_url = self.redis_url
        if self.celery_result_backend is None:
            self.celery_result_backend = self.redis_url
        if self.ha_ssh_host is None:
            self.ha_ssh_host = self.mqtt_host
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def database_url(self) -> str:
        """Sync DSN (engine / SQLAlchemy)."""
        return (
            f"postgresql://{self.local_db_user}:{self.local_db_password}"
            f"@{self.local_db_host}:{self.local_db_port}/{self.local_db_name}"
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def database_url_async(self) -> str:
        """Async DSN (ha_ingest / asyncpg)."""
        return (
            f"postgresql+asyncpg://{self.local_db_user}:{self.local_db_password}"
            f"@{self.local_db_host}:{self.local_db_port}/{self.local_db_name}"
        )

    @property
    def effective_sequence_length(self) -> int:
        """Sequence length in points.

        Uses ``nilm_sequence_length`` if set, else derives it from
        ``nilm_window_size_minutes`` (1 Hz -> 60 points/min).
        """
        if self.nilm_sequence_length is not None:
            return self.nilm_sequence_length
        return self.nilm_window_size_minutes * 60


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
# Backwards-compatible alias (the harvested engine imports ``settings``;
# kept in case other modules reference ``config``).
config = settings
