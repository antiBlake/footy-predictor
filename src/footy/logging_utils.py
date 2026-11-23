"""Logging setup shared by the CLI and library code."""

from __future__ import annotations

import logging


def setup_logging(level: str | int = "INFO") -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # requests/urllib3 are noisy at DEBUG
    logging.getLogger("urllib3").setLevel(logging.WARNING)
