"""Native training entrypoint: ``python -m training [--days N]``.

Run on macOS with the training extra:
    uv pip install -e ".[engine,training]"
    python -m training --days 14
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta

from loguru import logger

from .trainer import train


def main() -> None:
    parser = argparse.ArgumentParser(prog="training")
    parser.add_argument("--days", type=int, default=14, help="Days of history to train on")
    parser.add_argument("--model-type", default=None, help="gru | lstm (default: settings)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--no-balance", action="store_true", help="Disable dead-window subsampling")
    parser.add_argument("--dead-ratio", type=float, default=1.0, help="Dead windows kept per active window")
    args = parser.parse_args()

    end = datetime.now(UTC)
    start = end - timedelta(days=args.days)
    logger.info("Training window: {} -> {} ({} days)", start, end, args.days)

    report = train(
        start,
        end,
        model_type=args.model_type,
        epochs=args.epochs,
        stride=args.stride,
        balance=not args.no_balance,
        dead_ratio=args.dead_ratio,
    )
    logger.info("Done. {} windows, {} epochs", report.n_windows, report.epochs_trained)


if __name__ == "__main__":
    main()
