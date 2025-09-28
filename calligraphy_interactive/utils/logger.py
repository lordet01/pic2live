from __future__ import annotations

from loguru import logger as _logger


def get_logger(level: str = "INFO"):
    _logger.remove()
    _logger.add(lambda msg: print(msg, end=""), level=level.upper())
    return _logger


