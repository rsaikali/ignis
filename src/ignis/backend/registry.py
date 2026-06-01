"""Model registry: discover trained models + their artifacts (pure, no web).

A model is a ``<name>.keras`` in the models directory, optionally accompanied
by sidecars written by training/eval:
- ``<name>.report.json``      -- train report (period, per-appliance hours)
- ``<name>.comparison.json``  -- HA-vs-NILM scores vs the gate
- ``<name>.scalers.pkl``      -- fitted scalers (presence => predictable)

Pure filesystem + JSON; unit-tested against a temp dir.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ignis.nilm.config import settings


@dataclass(frozen=True)
class ModelEntry:
    """One discovered model and the artifacts beside it."""

    name: str
    keras_path: Path
    has_scalers: bool
    report: dict | None
    comparison: dict | None

    @property
    def trained_at(self) -> str | None:
        if self.report:
            return self.report.get("period_end")
        return None

    @property
    def gate_passed(self) -> list[str]:
        return self.comparison.get("passed", []) if self.comparison else []

    @property
    def gate_failed(self) -> list[str]:
        return self.comparison.get("failed", []) if self.comparison else []

    @property
    def predictable(self) -> bool:
        """A model can predict only if its scalers were persisted."""
        return self.has_scalers


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def discover(models_dir: Path | None = None) -> list[ModelEntry]:
    """List models in the directory, newest first (by name, which is timestamped)."""
    root = models_dir if models_dir is not None else Path(settings.nilm_model_path)
    if not root.exists():
        return []
    entries: list[ModelEntry] = []
    for keras in sorted(root.glob("*.keras"), reverse=True):
        name = keras.stem
        entries.append(
            ModelEntry(
                name=name,
                keras_path=keras,
                has_scalers=keras.with_suffix(".scalers.pkl").exists(),
                report=_load_json(keras.with_suffix(".report.json")),
                comparison=_load_json(keras.with_suffix(".comparison.json")),
            )
        )
    return entries


def get(name: str, models_dir: Path | None = None) -> ModelEntry | None:
    """Fetch a single model entry by name."""
    for entry in discover(models_dir):
        if entry.name == name:
            return entry
    return None
