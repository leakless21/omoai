"""
OMOAI Structured Logging System.

This module provides comprehensive structured logging with JSON formatting,
performance metrics, error tracking, and request tracing capabilities.
"""

from .config import LoggingConfig, get_logging_config
from .logger import (
    LoggerAdapter,
    generate_request_id,
    get_component_logger,
    get_logger,
    get_performance_summary,
    log_error,
    log_performance,
    performance_context,
    setup_logging,
    timed,
    with_request_context,
)
from .metrics import (
    MetricsCollector,
    PerformanceLogger,
    get_metrics_collector,
    get_performance_logger,
)
from .middleware import LoggingMiddleware, RequestLoggingMiddleware
from .serializers import JSONFormatter, StructuredFormatter

__all__ = [
    "JSONFormatter",
    "LoggerAdapter",
    "LoggingConfig",
    "LoggingMiddleware",
    "MetricsCollector",
    "PerformanceLogger",
    "RequestLoggingMiddleware",
    "StructuredFormatter",
    "generate_request_id",
    "get_component_logger",
    "get_logger",
    "get_logging_config",
    "get_metrics_collector",
    "get_performance_logger",
    "get_performance_summary",
    "log_error",
    "log_performance",
    "performance_context",
    "setup_logging",
    "timed",
    "with_request_context",
]
