"""Entrypoint: ``python -m ha_ingest``."""

from __future__ import annotations

import asyncio

from loguru import logger

from .subscriber import run


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("ha_ingest stopped")


if __name__ == "__main__":
    main()
