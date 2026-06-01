"""Champion/challenger model management (the deploy seam).

A trained model is a timestamped *challenger* (``ignis_<type>_<ts>.keras``).
``champion.keras`` is the single production model (the one rsynced to the Pi).
A challenger is promoted (copied over champion) only if it scores better.

Scoring is a comparable scalar derived from a ComparisonReport:
``(appliances_passing_gate, mean_state_f1)`` -- more gate passes wins, mean F1
breaks ties. Pure functions here; file side-effects are explicit and logged.

Promotion is LOCAL only: it overwrites champion.keras but keeps the dated
challengers, so rollback is a copy. Shipping to the Pi (rsync) stays manual.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from ignis.nilm.config import settings

CHAMPION_NAME = "champion"
_SIDECARS = (".metadata.json", ".report.json", ".scalers.pkl", ".comparison.json")


@dataclass(frozen=True)
class Score:
    """Comparable model score. Higher is better, gate passes dominate."""

    gate_passes: int
    mean_f1: float

    def __gt__(self, other: Score) -> bool:
        return (self.gate_passes, self.mean_f1) > (other.gate_passes, other.mean_f1)

    def __ge__(self, other: Score) -> bool:
        return (self.gate_passes, self.mean_f1) >= (other.gate_passes, other.mean_f1)


def score_from_comparison(comparison: dict) -> Score:
    """Derive a Score from a .comparison.json dict."""
    apps = comparison.get("appliances", {})
    passes = sum(1 for m in apps.values() if m.get("passes_gate"))
    f1s = [m.get("state_f1", 0.0) for m in apps.values()]
    mean_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    return Score(gate_passes=passes, mean_f1=mean_f1)


def models_dir() -> Path:
    return Path(settings.nilm_model_path)


def champion_path() -> Path:
    return models_dir() / f"{CHAMPION_NAME}.keras"


def latest_challenger() -> Path | None:
    """Most recent dated challenger .keras (excludes the champion)."""
    root = models_dir()
    if not root.exists():
        return None
    cands = sorted((p for p in root.glob("*.keras") if p.stem != CHAMPION_NAME), reverse=True)
    return cands[0] if cands else None


def _copy_with_sidecars(src: Path, dst_stem: str) -> Path:
    """Copy src.keras + its sidecars to dst_stem.* in the same dir."""
    root = src.parent
    dst = root / f"{dst_stem}.keras"
    shutil.copy2(src, dst)
    for ext in _SIDECARS:
        s = src.with_suffix(ext)
        if s.exists():
            shutil.copy2(s, root / f"{dst_stem}{ext}")
    return dst


def promote(challenger: Path) -> Path:
    """Copy a challenger (and sidecars) over champion.*. Returns champion path."""
    champ = _copy_with_sidecars(challenger, CHAMPION_NAME)
    logger.info("Promoted {} -> {}", challenger.name, champ.name)
    return champ


def maybe_promote(challenger: Path, challenger_score: Score) -> tuple[bool, Score | None]:
    """Promote the challenger iff it beats the current champion.

    Returns ``(promoted, champion_score_before)``. With no champion yet, the
    challenger is promoted unconditionally.
    """
    champ = champion_path()
    if not champ.exists():
        promote(challenger)
        return True, None

    import json

    champ_cmp_path = champ.with_suffix(".comparison.json")
    champ_score = (
        score_from_comparison(json.loads(champ_cmp_path.read_text())) if champ_cmp_path.exists() else Score(0, 0.0)
    )

    if challenger_score > champ_score:
        promote(challenger)
        return True, champ_score
    logger.info("Not promoted: challenger {} <= champion {}", challenger_score, champ_score)
    return False, champ_score


def prune_challengers(keep: int = 5) -> list[str]:
    """Delete all but the newest ``keep`` dated challengers (+ their sidecars)."""
    root = models_dir()
    if not root.exists():
        return []
    dated = sorted((p for p in root.glob("*.keras") if p.stem != CHAMPION_NAME), reverse=True)
    removed: list[str] = []
    for old in dated[keep:]:
        for ext in (".keras", *_SIDECARS):
            f = old.with_suffix(ext)
            if f.exists():
                f.unlink()
        removed.append(old.stem)
    if removed:
        logger.info("Pruned {} old challengers (kept {})", len(removed), keep)
    return removed
