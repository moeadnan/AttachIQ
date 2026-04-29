"""Loguru-based structured logger for AttachIQ. No print() statements anywhere."""

from __future__ import annotations

import sys

from loguru import logger

_CONFIGURED = False


def setup_logger(level: str = "INFO") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        ),
        colorize=True,
    )
    _CONFIGURED = True


def get_logger(name: str | None = None):
    setup_logger()
    if name:
        return logger.bind(scope=name)
    return logger


setup_logger()
