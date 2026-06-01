"""Eval entrypoint: ``python -m eval [--model PATH] [--days N]``.

With no --model, evaluates the latest dated challenger, then auto-promotes it
to champion.keras if it beats the current champion (unless --no-promote).

Run native (engine extra):
    .venv/bin/python -m eval               # latest model, auto-promote
    .venv/bin/python -m eval --model models/foo.keras --days 3
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
from pathlib import Path

from loguru import logger

from .compare import compare, gate_summary
from .promote import latest_challenger, maybe_promote, prune_challengers, score_from_comparison


def main() -> None:
    ap = argparse.ArgumentParser(prog="eval")
    ap.add_argument("--model", type=Path, default=None, help="Model .keras (default: latest challenger)")
    ap.add_argument("--days", type=float, default=3, help="Holdout length (most recent N days)")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--no-promote", action="store_true", help="Do not auto-promote to champion")
    ap.add_argument("--keep", type=int, default=5, help="Dated challengers to retain on prune")
    args = ap.parse_args()

    model = args.model or latest_challenger()
    if model is None:
        raise SystemExit("No model found. Train one first (make train).")

    end = datetime.now(UTC)
    start = end - timedelta(days=args.days)
    report = compare(model, start, end, stride=args.stride)
    logger.info("Model: {}", model.name)
    logger.info(gate_summary(report))
    for app, m in report.appliances.items():
        logger.info("  {:16s} F1={:.3f} energy_err={} gate={}", app, m["state_f1"], m["energy_error"], m["passes_gate"])

    if args.no_promote:
        return
    score = score_from_comparison(report.to_dict())
    promoted, champ_score = maybe_promote(model, score)
    if promoted:
        logger.info("PROMOTED to champion (score {} > {})", score, champ_score)
    prune_challengers(keep=args.keep)


if __name__ == "__main__":
    main()
