"""Project-wide logger factory with a consistent format."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def get_logger(name: str = "rfe", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger; idempotent across calls."""
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root = logging.getLogger("rfe")
        root.addHandler(handler)
        root.setLevel(level)
        root.propagate = False
        _CONFIGURED = True
    return logging.getLogger(name if name.startswith("rfe") else f"rfe.{name}")
