"""Compatibility logging facade for tests.

Exposes get_logger and get_performance_logger under src.omoai.logging
by delegating to logging_system.
"""
from __future__ import annotations

from ..logging_system.logger import get_logger  # noqa: F401
from ..logging_system.metrics import (  # noqa: F401
    get_performance_logger,
    get_metrics_collector,
)

__all__ = ["get_logger", "get_performance_logger", "get_metrics_collector"]