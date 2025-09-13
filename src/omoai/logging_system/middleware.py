"""
Logging middleware for API requests and pipeline operations.
"""

import time
from collections.abc import Callable
from typing import Any

from .logger import generate_request_id, get_logger, log_error
from .metrics import get_performance_logger

try:
    # from litestar import Request, Response  # Unused imports removed
    # from litestar.middleware.base import AbstractMiddleware  # Unused import removed
    from litestar.types import ASGIApp, Receive, Scope, Send

    HAS_LITESTAR = True
except ImportError:
    HAS_LITESTAR = False


class RequestLoggingMiddleware:
    """Middleware for logging HTTP requests and responses."""

    def __init__(
        self,
        app: "ASGIApp",
        log_requests: bool = True,
        log_responses: bool = True,
        log_bodies: bool = False,
        exclude_paths: list | None = None,
        performance_threshold_ms: float = 100.0,
    ):
        self.app = app
        self.log_requests = log_requests
        self.log_responses = log_responses
        self.log_bodies = log_bodies
        self.exclude_paths = exclude_paths or ["/health", "/metrics", "/favicon.ico"]
        self.performance_threshold_ms = performance_threshold_ms
        self.logger = get_logger("omoai.api.requests")
        self.performance_logger = get_performance_logger()

    async def __call__(self, scope: "Scope", receive: "Receive", send: "Send") -> None:
        """ASGI middleware implementation."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Generate request ID
        request_id = generate_request_id()

        # Extract request info
        method = scope["method"]
        path = scope["path"]
        query_string = scope.get("query_string", b"").decode()

        # Skip excluded paths
        if path in self.exclude_paths:
            await self.app(scope, receive, send)
            return

        # Add request ID to scope for access in handlers
        scope["request_id"] = request_id

        start_time = time.time()

        # Log incoming request
        if self.log_requests:
            self.logger.info(
                f"Request started: {method} {path}",
                extra={
                    "request_id": request_id,
                    "method": method,
                    "path": path,
                    "query_string": query_string,
                    "user_agent": dict(scope.get("headers", {}))
                    .get(b"user-agent", b"")
                    .decode(),
                    "remote_addr": scope.get("client", ["unknown"])[0]
                    if scope.get("client")
                    else "unknown",
                },
            )

        # Track response info
        response_info = {"status_code": None, "content_length": None}

        async def send_wrapper(message: dict[str, Any]) -> None:
            """Wrapper to capture response information."""
            if message["type"] == "http.response.start":
                response_info["status_code"] = message["status"]
                headers = dict(message.get("headers", []))
                response_info["content_length"] = headers.get(b"content-length")

            await send(message)

        try:
            # Process request
            await self.app(scope, receive, send_wrapper)

            # Calculate duration
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000

            # Log response
            if self.log_responses:
                status_code = response_info["status_code"] or 500

                # Determine log level based on status code and duration
                if status_code >= 500:
                    level = "ERROR"
                elif status_code >= 400:
                    level = "WARNING"
                elif duration_ms > self.performance_threshold_ms:
                    level = "WARNING"
                else:
                    level = "INFO"

                log_method = getattr(self.logger, level.lower())
                log_method(
                    f"Request completed: {method} {path} - {status_code} in {duration_ms:.2f}ms",
                    extra={
                        "request_id": request_id,
                        "method": method,
                        "path": path,
                        "status_code": status_code,
                        "duration_ms": duration_ms,
                        "content_length": response_info["content_length"],
                    },
                )

            # Record performance metrics
            self.performance_logger.log_operation(
                operation=f"HTTP_{method}_{path}",
                duration_ms=duration_ms,
                success=response_info["status_code"] < 400
                if response_info["status_code"]
                else False,
                request_id=request_id,
                status_code=response_info["status_code"],
                path=path,
                method=method,
            )

        except Exception as e:
            # Calculate duration even for errors
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000

            # Log error
            log_error(
                message=f"Request failed: {method} {path}",
                error=e,
                error_type="REQUEST_PROCESSING_ERROR",
                logger=self.logger,
                request_id=request_id,
                method=method,
                path=path,
                duration_ms=duration_ms,
            )

            # Record failed operation
            self.performance_logger.log_operation(
                operation=f"HTTP_{method}_{path}",
                duration_ms=duration_ms,
                success=False,
                error_type="REQUEST_PROCESSING_ERROR",
                request_id=request_id,
                path=path,
                method=method,
            )

            raise


class LoggingMiddleware:
    """General purpose logging middleware for any callable."""

    def __init__(
        self,
        operation_name: str,
        logger_name: str | None = None,
        log_args: bool = False,
        log_result: bool = False,
        performance_threshold_ms: float = 100.0,
    ):
        self.operation_name = operation_name
        self.logger = get_logger(logger_name or "omoai.operations")
        self.performance_logger = get_performance_logger()
        self.log_args = log_args
        self.log_result = log_result
        self.performance_threshold_ms = performance_threshold_ms

    def __call__(self, func: Callable) -> Callable:
        """Decorator implementation."""

        async def async_wrapper(*args, **kwargs):
            return await self._execute_with_logging(func, args, kwargs, is_async=True)

        def sync_wrapper(*args, **kwargs):
            return self._execute_with_logging(func, args, kwargs, is_async=False)

        # Return appropriate wrapper based on function type
        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    async def _execute_async_with_logging(
        self, func: Callable, args: tuple, kwargs: dict
    ):
        """Execute async function with logging."""
        operation_id = generate_request_id()
        start_time = time.time()

        # Log start
        extra_data = {"operation_id": operation_id, "operation": self.operation_name}
        if self.log_args:
            extra_data.update(
                {
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys()),
                }
            )

        self.logger.debug(
            f"Starting operation: {self.operation_name}", extra=extra_data
        )

        try:
            result = await func(*args, **kwargs)

            # Calculate duration
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000

            # Log completion
            success_extra = extra_data.copy()
            success_extra["duration_ms"] = duration_ms

            if self.log_result and result is not None:
                if hasattr(result, "__len__"):
                    success_extra["result_size"] = len(result)
                success_extra["result_type"] = type(result).__name__

            if duration_ms > self.performance_threshold_ms:
                self.logger.warning(
                    f"Slow operation completed: {self.operation_name} took {duration_ms:.2f}ms",
                    extra=success_extra,
                )
            else:
                self.logger.info(
                    f"Operation completed: {self.operation_name} in {duration_ms:.2f}ms",
                    extra=success_extra,
                )

            # Record performance
            self.performance_logger.log_operation(
                operation=self.operation_name,
                duration_ms=duration_ms,
                success=True,
                operation_id=operation_id,
            )

            return result

        except Exception as e:
            # Calculate duration
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000

            # Log error
            log_error(
                message=f"Operation failed: {self.operation_name}",
                error=e,
                error_type="OPERATION_ERROR",
                logger=self.logger,
                operation_id=operation_id,
                operation=self.operation_name,
                duration_ms=duration_ms,
            )

            # Record failed operation
            self.performance_logger.log_operation(
                operation=self.operation_name,
                duration_ms=duration_ms,
                success=False,
                error_type="OPERATION_ERROR",
                operation_id=operation_id,
            )

            raise

    def _execute_with_logging(
        self, func: Callable, args: tuple, kwargs: dict, is_async: bool = False
    ):
        """Execute function with logging (sync version)."""
        if is_async:
            return self._execute_async_with_logging(func, args, kwargs)

        operation_id = generate_request_id()
        start_time = time.time()

        # Log start
        extra_data = {"operation_id": operation_id, "operation": self.operation_name}
        if self.log_args:
            extra_data.update(
                {
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys()),
                }
            )

        self.logger.debug(
            f"Starting operation: {self.operation_name}", extra=extra_data
        )

        try:
            result = func(*args, **kwargs)

            # Calculate duration
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000

            # Log completion
            success_extra = extra_data.copy()
            success_extra["duration_ms"] = duration_ms

            if self.log_result and result is not None:
                if hasattr(result, "__len__"):
                    success_extra["result_size"] = len(result)
                success_extra["result_type"] = type(result).__name__

            if duration_ms > self.performance_threshold_ms:
                self.logger.warning(
                    f"Slow operation completed: {self.operation_name} took {duration_ms:.2f}ms",
                    extra=success_extra,
                )
            else:
                self.logger.info(
                    f"Operation completed: {self.operation_name} in {duration_ms:.2f}ms",
                    extra=success_extra,
                )

            # Record performance
            self.performance_logger.log_operation(
                operation=self.operation_name,
                duration_ms=duration_ms,
                success=True,
                operation_id=operation_id,
            )

            return result

        except Exception as e:
            # Calculate duration
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000

            # Log error
            log_error(
                message=f"Operation failed: {self.operation_name}",
                error=e,
                error_type="OPERATION_ERROR",
                logger=self.logger,
                operation_id=operation_id,
                operation=self.operation_name,
                duration_ms=duration_ms,
            )

            # Record failed operation
            self.performance_logger.log_operation(
                operation=self.operation_name,
                duration_ms=duration_ms,
                success=False,
                error_type="OPERATION_ERROR",
                operation_id=operation_id,
            )

            raise


def logged_operation(
    operation_name: str,
    logger_name: str | None = None,
    log_args: bool = False,
    log_result: bool = False,
    performance_threshold_ms: float = 100.0,
) -> Callable:
    """Decorator for automatic operation logging."""
    return LoggingMiddleware(
        operation_name=operation_name,
        logger_name=logger_name,
        log_args=log_args,
        log_result=log_result,
        performance_threshold_ms=performance_threshold_ms,
    )
