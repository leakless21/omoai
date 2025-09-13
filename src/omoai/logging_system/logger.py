"""
Core logging utilities for OMOAI.
"""

import functools
import logging
import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, TypeVar

from .config import LoggingConfig, configure_python_logging, get_logging_config

# Global logging configuration
_logging_configured = False
_logging_config: LoggingConfig | None = None

F = TypeVar("F", bound=Callable[..., Any])


def setup_logging(config: LoggingConfig | None = None) -> None:
    """Setup OMOAI structured logging system."""
    global _logging_configured, _logging_config

    if config is None:
        config = get_logging_config()

    _logging_config = config

    # Configure Python's logging system
    configure_python_logging(config)

    _logging_configured = True

    # Log that logging is configured
    logger = get_logger(__name__)
    logger.info(
        "Structured logging configured",
        extra={
            "format_type": config.format_type,
            "level": config.level,
            "performance_logging": config.enable_performance_logging,
            "error_tracking": config.enable_error_tracking,
            "metrics": config.enable_metrics,
        },
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with optional auto-setup."""
    global _logging_configured

    # Auto-configure if not already done
    if not _logging_configured:
        setup_logging()

    return logging.getLogger(name)


def log_performance(
    operation: str,
    duration_ms: float,
    logger: logging.Logger | None = None,
    **extra_metrics: Any,
) -> None:
    """Log performance metrics for an operation."""
    if logger is None:
        logger = get_logger("omoai.performance")

    config = _logging_config or get_logging_config()

    if not config.should_log_performance(duration_ms):
        return

    # Prepare extra data
    extra_data = {
        "operation": operation,
        "duration_ms": duration_ms,
        **extra_metrics,
    }

    # Determine log level based on performance
    if duration_ms > 5000:  # > 5 seconds
        level = logging.WARNING
        msg = f"SLOW operation: {operation} took {duration_ms:.2f}ms"
    elif duration_ms > 1000:  # > 1 second
        level = logging.INFO
        msg = f"Operation: {operation} took {duration_ms:.2f}ms"
    else:
        level = logging.DEBUG
        msg = f"Fast operation: {operation} completed in {duration_ms:.2f}ms"

    logger.log(level, msg, extra=extra_data)


def log_error(
    message: str,
    error: Exception | None = None,
    error_type: str = "GENERAL",
    error_code: str | None = None,
    remediation: str | None = None,
    logger: logging.Logger | None = None,
    **context: Any,
) -> None:
    """Log structured error information."""
    if logger is None:
        logger = get_logger("omoai.error")

    # Prepare extra data
    extra_data = {
        "error_type": error_type,
        **context,
    }

    if error_code:
        extra_data["error_code"] = error_code

    if remediation:
        extra_data["remediation"] = remediation

    # Log with exception info if available
    if error:
        logger.error(message, exc_info=error, extra=extra_data)
    else:
        logger.error(message, extra=extra_data)


@contextmanager
def performance_context(
    operation: str,
    logger: logging.Logger | None = None,
    log_start: bool = False,
    **extra_metrics: Any,
):
    """Context manager for performance measurement."""
    if logger is None:
        logger = get_logger("omoai.performance")

    start_time = time.time()

    if log_start:
        logger.debug(f"Starting operation: {operation}", extra={"operation": operation})

    try:
        yield
    finally:
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000

        log_performance(
            operation=operation,
            duration_ms=duration_ms,
            logger=logger,
            **extra_metrics,
        )


def timed(
    operation: str | None = None,
    logger: logging.Logger | None = None,
    log_args: bool = False,
    log_result: bool = False,
) -> Callable[[F], F]:
    """Decorator for automatic performance logging."""

    def decorator(func: F) -> F:
        op_name = operation or f"{func.__module__}.{func.__name__}"

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            perf_logger = logger or get_logger(f"omoai.performance.{func.__module__}")

            extra_data = {}
            if log_args and args:
                extra_data["args_count"] = len(args)
            if log_args and kwargs:
                extra_data["kwargs_keys"] = list(kwargs.keys())

            start_time = time.time()

            try:
                result = func(*args, **kwargs)

                end_time = time.time()
                duration_ms = (end_time - start_time) * 1000

                if log_result and result is not None:
                    if hasattr(result, "__len__"):
                        extra_data["result_size"] = len(result)
                    extra_data["result_type"] = type(result).__name__

                log_performance(
                    operation=op_name,
                    duration_ms=duration_ms,
                    logger=perf_logger,
                    **extra_data,
                )

                return result

            except Exception as e:
                end_time = time.time()
                duration_ms = (end_time - start_time) * 1000

                log_error(
                    message=f"Operation failed: {op_name}",
                    error=e,
                    error_type="OPERATION_FAILED",
                    logger=perf_logger,
                    operation=op_name,
                    duration_ms=duration_ms,
                    **extra_data,
                )
                raise

        return wrapper

    return decorator


def generate_request_id() -> str:
    """Generate a unique request ID for tracing."""
    return str(uuid.uuid4())


def with_request_context(
    request_id: str | None = None,
    **context: Any,
) -> Callable[[F], F]:
    """Decorator to add request context to all logs within a function."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate request ID if not provided
            rid = request_id or generate_request_id()

            # Add context to all loggers used within this function
            old_factory = logging.getLogRecordFactory()

            def record_factory(*args, **kwargs):
                record = old_factory(*args, **kwargs)
                record.request_id = rid
                for key, value in context.items():
                    setattr(record, key, value)
                return record

            logging.setLogRecordFactory(record_factory)

            try:
                return func(*args, **kwargs)
            finally:
                logging.setLogRecordFactory(old_factory)

        return wrapper

    return decorator


def get_performance_summary(
    operations: list,
    total_duration_ms: float,
) -> dict[str, Any]:
    """Generate a performance summary for multiple operations."""
    if not operations:
        return {"total_duration_ms": total_duration_ms, "operations": []}

    summary = {
        "total_duration_ms": total_duration_ms,
        "operation_count": len(operations),
        "operations": operations,
        "performance_breakdown": {},
    }

    # Calculate percentages
    for op in operations:
        op_duration = op.get("duration_ms", 0)
        percentage = (
            (op_duration / total_duration_ms * 100) if total_duration_ms > 0 else 0
        )
        op_name = op.get("operation", "unknown")
        summary["performance_breakdown"][op_name] = {
            "duration_ms": op_duration,
            "percentage": round(percentage, 1),
        }

    # Find slowest operation
    slowest = max(operations, key=lambda x: x.get("duration_ms", 0), default={})
    if slowest:
        summary["slowest_operation"] = {
            "name": slowest.get("operation", "unknown"),
            "duration_ms": slowest.get("duration_ms", 0),
        }

    return summary


class LoggerAdapter(logging.LoggerAdapter):
    """Custom logger adapter that adds consistent extra fields."""

    def __init__(self, logger: logging.Logger, extra: dict[str, Any]):
        super().__init__(logger, extra)

    def process(self, msg: Any, kwargs: dict[str, Any]) -> tuple:
        """Process log call to add consistent extra fields."""
        # Merge adapter's extra with call-specific extra
        extra = kwargs.get("extra", {})
        merged_extra = {**self.extra, **extra}
        kwargs["extra"] = merged_extra
        return msg, kwargs


def get_component_logger(component: str, **extra_context: Any) -> LoggerAdapter:
    """Get a logger adapter for a specific component with context."""
    base_logger = get_logger(f"omoai.{component}")
    return LoggerAdapter(base_logger, extra_context)
