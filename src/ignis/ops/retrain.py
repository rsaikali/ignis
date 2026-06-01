"""Scheduled retrain: train -> eval -> auto-promote (one autonomous pass).

Run as an ephemeral job on the Pi (cron -> `docker compose run --rm train`),
with the models volume mounted read-write. Trains a fresh challenger on the
last N days, evaluates it vs HA on a holdout, and promotes it to champion only
if it beats the current one. publish reloads champion.keras on its next cycle,
so nothing else needs restarting.

This closes the loop on-device: no Mac required for production retrains.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta

from loguru import logger

from ignis.eval.compare import compare, gate_summary
from ignis.eval.promote import maybe_promote, prune_challengers, score_from_comparison
from ignis.training.trainer import train


def run(train_days: int, eval_days: float, epochs: int | None, keep: int) -> None:
    """Train a challenger, evaluate it, promote if better, prune old ones."""
    end = datetime.now(UTC)
    start = end - timedelta(days=train_days)

    logger.info("Retrain: training on last {} days", train_days)
    report = train(start, end, epochs=epochs)
    challenger = report.model_name
    logger.info("Trained challenger {} ({} windows, {} epochs)", challenger, report.n_windows, report.epochs_trained)

    # Evaluate the fresh challenger over a recent holdout.
    from pathlib import Path

    from ignis.nilm.config import settings

    model_path = Path(settings.nilm_model_path) / f"{challenger}.keras"
    eval_end = datetime.now(UTC)
    eval_start = eval_end - timedelta(days=eval_days)
    comparison = compare(model_path, eval_start, eval_end)
    logger.info("Eval: {}", gate_summary(comparison))

    score = score_from_comparison(comparison.to_dict())
    promoted, champ_score = maybe_promote(model_path, score)
    if promoted:
        logger.info("PROMOTED {} to champion (score {} > {})", challenger, score, champ_score)
    else:
        logger.info("Kept current champion (challenger {} <= {})", score, champ_score)
    prune_challengers(keep=keep)


def main() -> None:
    ap = argparse.ArgumentParser(prog="ignis.ops.retrain")
    ap.add_argument("--train-days", type=int, default=30)
    ap.add_argument("--eval-days", type=float, default=3)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--keep", type=int, default=5)
    args = ap.parse_args()
    run(args.train_days, args.eval_days, args.epochs, args.keep)


if __name__ == "__main__":
    main()
