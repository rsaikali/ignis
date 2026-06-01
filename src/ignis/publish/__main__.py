"""Publish entrypoint: ``python -m publish [--once] [--interval N]``.

Runs champion inference and publishes the grouped snapshot + model meta (with
honest F1-vs-HA scores). Default loops every --interval seconds; --once does a
single cycle. Run native or on the Pi (engine extra):
    python -m publish --once
    python -m publish --interval 60
"""

from __future__ import annotations

import argparse
import asyncio

from loguru import logger

from .live import publish_once


async def _loop(window_minutes: int, interval: int) -> None:
    while True:
        try:
            await publish_once(window_minutes)
        except SystemExit as exc:
            logger.warning("Publish skipped: {}", exc)
        await asyncio.sleep(interval)


def main() -> None:
    ap = argparse.ArgumentParser(prog="publish")
    ap.add_argument("--once", action="store_true", help="Single cycle then exit")
    ap.add_argument("--interval", type=int, default=60, help="Seconds between cycles")
    # Needs >= sequence_length aligned ticks (99 x 30s = 50min minimum); use a
    # wider window so live gaps still leave enough points.
    ap.add_argument("--window-minutes", type=int, default=180, help="Aggregate window to disaggregate")
    args = ap.parse_args()

    if args.once:
        asyncio.run(publish_once(args.window_minutes))
    else:
        try:
            asyncio.run(_loop(args.window_minutes, args.interval))
        except KeyboardInterrupt:
            logger.info("publish stopped")


if __name__ == "__main__":
    main()
