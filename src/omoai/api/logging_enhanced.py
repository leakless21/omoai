"""
Enhanced API logging integration with structured logging system.

This module provides the integration layer between the OMOAI structured
logging system and the existing Litestar API framework.
"""
import time
from typing import Any, Dict, Optional

from litestar import Request, Response
from litestar.logging.config import LoggingConfig as LitestarLoggingConfig

from ..logging import get_logger, setup_logging, get_logging_config, JSONFormatter, StructuredFormatter
from ..logging.middleware import RequestLoggingMiddleware


def create_enhanced_logging_config() -> LitestarLoggingConfig:
    """Create enhanced Litestar logging configuration using OMOAI structured logging."""
    
    # Setup OMOAI logging first
    omoai_config = get_logging_config()
    setup_logging(omoai_config)
    
    # Create appropriate formatter based on config
    if omoai_config.format_type == "json":
        formatter = {
            "class": "src.omoai.logging.formatters.JSONFormatter",
            "include_extra": True,
        }
    elif omoai_config.format_type == "structured":
        formatter = {
            "class": "src.omoai.logging.formatters.StructuredFormatter", 
            "include_extra": True,
            "color": True,
        }
    else:
        formatter = {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        }
    
    return LitestarLoggingConfig(
        root={
            "level": omoai_config.level,
            "handlers": ["console", "file"] if omoai_config.enable_file else ["console"]
        },
        formatters={
            "standard": formatter,
            "json": {
                "class": "src.omoai.logging.formatters.JSONFormatter",
                "include_extra": True,
            },
            "structured": {
                "class": "src.omoai.logging.formatters.StructuredFormatter",
                "include_extra": True,
                "color": True,
            }
        },
        handlers={
            "console": {
                "class": "logging.StreamHandler",
                "level": omoai_config.level,
                "formatter": "standard",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": omoai_config.level,
                "formatter": "json",
                "filename": str(omoai_config.log_file) if omoai_config.log_file else "/tmp/omoai.log",
                "maxBytes": omoai_config.max_file_size,
                "backupCount": omoai_config.backup_count,
            } if omoai_config.enable_file else {}
        },
        configure_root_logger=True,
    )


def get_request_logger(request: Request) -> Any:
    """Get a request-specific logger with context."""
    from ..logging import LoggerAdapter
    
    base_logger = get_logger("omoai.api.request")
    
    # Extract request context
    context = {
        "request_id": getattr(request.scope, "request_id", "unknown"),
        "method": request.method,
        "path": request.url.path,
        "user_agent": request.headers.get("user-agent", "unknown"),
        "remote_addr": request.client.host if request.client else "unknown",
    }
    
    return LoggerAdapter(base_logger, context)


def log_request_start(request: Request) -> None:
    """Log the start of a request with context."""
    logger = get_request_logger(request)
    logger.info("Request started", extra={
        "query_params": dict(request.query_params),
        "content_type": request.headers.get("content-type"),
        "content_length": request.headers.get("content-length"),
    })


def log_request_end(
    request: Request,
    response: Response,
    duration_ms: float,
    error: Optional[Exception] = None
) -> None:
    """Log the end of a request with performance metrics."""
    logger = get_request_logger(request)
    
    if error:
        from ..logging import log_error
        log_error(
            message=f"Request failed: {request.method} {request.url.path}",
            error=error,
            error_type="API_REQUEST_ERROR",
            logger=logger,
            duration_ms=duration_ms,
            status_code=getattr(response, "status_code", 500),
        )
    else:
        # Determine log level based on performance and status
        status_code = response.status_code
        if status_code >= 500:
            level = "error"
        elif status_code >= 400:
            level = "warning"
        elif duration_ms > 1000:  # > 1 second
            level = "warning"
        else:
            level = "info"
        
        log_method = getattr(logger, level)
        log_method("Request completed", extra={
            "status_code": status_code,
            "duration_ms": duration_ms,
            "response_size": len(response.content) if hasattr(response, "content") else None,
        })


class APIRequestLogger:
    """High-level API request logging interface."""
    
    def __init__(self):
        self.logger = get_logger("omoai.api")
        from ..logging import get_performance_logger
        self.perf_logger = get_performance_logger()
    
    def log_endpoint_access(
        self,
        endpoint: str,
        method: str,
        request_size: Optional[int] = None,
        **context: Any
    ) -> None:
        """Log API endpoint access."""
        self.logger.info(f"Endpoint accessed: {method} {endpoint}", extra={
            "endpoint": endpoint,
            "method": method,
            "request_size_bytes": request_size,
            **context
        })
    
    def log_processing_start(
        self,
        operation: str,
        request_id: str,
        **context: Any
    ) -> None:
        """Log the start of processing operation."""
        self.logger.info(f"Processing started: {operation}", extra={
            "operation": operation,
            "request_id": request_id,
            **context
        })
    
    def log_processing_end(
        self,
        operation: str,
        request_id: str,
        duration_ms: float,
        success: bool = True,
        **metrics: Any
    ) -> None:
        """Log the end of processing operation."""
        # Log to standard logger
        if success:
            self.logger.info(f"Processing completed: {operation}", extra={
                "operation": operation,
                "request_id": request_id,
                "duration_ms": duration_ms,
                **metrics
            })
        else:
            self.logger.error(f"Processing failed: {operation}", extra={
                "operation": operation,
                "request_id": request_id,
                "duration_ms": duration_ms,
                **metrics
            })
        
        # Record performance metrics
        self.perf_logger.log_operation(
            operation=f"api_{operation}",
            duration_ms=duration_ms,
            success=success,
            request_id=request_id,
            **metrics
        )
    
    def log_model_operation(
        self,
        model_type: str,
        operation: str,
        duration_ms: float,
        success: bool = True,
        **context: Any
    ) -> None:
        """Log model-specific operations."""
        self.logger.info(f"Model operation: {model_type}.{operation}", extra={
            "model_type": model_type,
            "operation": operation,
            "duration_ms": duration_ms,
            "success": success,
            **context
        })
        
        # Record in performance logger
        self.perf_logger.log_operation(
            operation=f"model_{model_type}_{operation}",
            duration_ms=duration_ms,
            success=success,
            **context
        )
    
    def log_health_check(self, status: str, **metrics: Any) -> None:
        """Log health check results."""
        self.logger.info(f"Health check: {status}", extra={
            "health_status": status,
            **metrics
        })
    
    def log_configuration_load(self, config_path: str, success: bool = True) -> None:
        """Log configuration loading."""
        if success:
            self.logger.info(f"Configuration loaded: {config_path}", extra={
                "config_path": config_path,
                "success": success,
            })
        else:
            self.logger.error(f"Configuration failed to load: {config_path}", extra={
                "config_path": config_path,
                "success": success,
            })


# Global API logger instance
_api_logger: Optional[APIRequestLogger] = None


def get_api_logger() -> APIRequestLogger:
    """Get the global API logger instance."""
    global _api_logger
    if _api_logger is None:
        _api_logger = APIRequestLogger()
    return _api_logger
