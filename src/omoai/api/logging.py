"""Logging configuration for the API."""
import logging
import sys
from typing import Union

from litestar.logging.config import LoggingConfig


def configure_logging() -> LoggingConfig:
    """Configure logging for the application."""
    return LoggingConfig(
        root={"level": "INFO", "handlers": ["console"]},
        formatters={
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            }
        },
        handlers={
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "standard",
                "stream": sys.stdout
            }
        }
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name."""
    return logging.getLogger(name)