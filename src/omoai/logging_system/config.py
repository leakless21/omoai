"""
Logging configuration for OMOAI structured logging.
"""
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

from ..config import get_config, OmoAIConfig


class LoggingConfig(BaseModel):
    """Structured logging configuration."""
    
    # Logging levels
    level: str = Field(default="INFO", description="Root logging level")
    
    # Output configuration
    format_type: str = Field(default="structured", description="Logging format: 'structured', 'json', or 'simple'")
    enable_console: bool = Field(default=True, description="Enable console output")
    enable_file: bool = Field(default=False, description="Enable file output")
    
    # File logging
    log_file: Optional[Path] = Field(default=None, description="Log file path")
    max_file_size: int = Field(default=10*1024*1024, description="Maximum log file size in bytes")
    backup_count: int = Field(default=5, description="Number of backup log files")
    
    # Performance logging
    enable_performance_logging: bool = Field(default=True, description="Enable performance metrics")
    performance_threshold_ms: float = Field(default=100.0, description="Log performance if above threshold (ms)")
    
    # Request tracing
    enable_request_tracing: bool = Field(default=True, description="Enable request ID tracing")
    trace_headers: bool = Field(default=False, description="Include request headers in traces")
    
    # Error tracking
    enable_error_tracking: bool = Field(default=True, description="Enable structured error tracking")
    include_stacktrace: bool = Field(default=True, description="Include stack traces in error logs")
    
    # Metrics
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    metrics_interval: int = Field(default=60, description="Metrics collection interval (seconds)")
    
    # Environment-based overrides
    debug_mode: bool = Field(default=False, description="Enable debug mode logging")
    quiet_mode: bool = Field(default=False, description="Suppress non-error output")
    
    @classmethod
    def from_environment(cls) -> "LoggingConfig":
        """Create logging config from environment variables."""
        return cls(
            level=os.environ.get("OMOAI_LOG_LEVEL", "INFO").upper(),
            format_type=os.environ.get("OMOAI_LOG_FORMAT", "structured"),
            enable_console=os.environ.get("OMOAI_LOG_CONSOLE", "true").lower() == "true",
            enable_file=os.environ.get("OMOAI_LOG_FILE_ENABLED", "false").lower() == "true",
            log_file=Path(os.environ["OMOAI_LOG_FILE"]) if "OMOAI_LOG_FILE" in os.environ else None,
            enable_performance_logging=os.environ.get("OMOAI_LOG_PERFORMANCE", "true").lower() == "true",
            performance_threshold_ms=float(os.environ.get("OMOAI_LOG_PERF_THRESHOLD", "100.0")),
            enable_request_tracing=os.environ.get("OMOAI_LOG_TRACING", "true").lower() == "true",
            trace_headers=os.environ.get("OMOAI_LOG_TRACE_HEADERS", "false").lower() == "true",
            enable_error_tracking=os.environ.get("OMOAI_LOG_ERRORS", "true").lower() == "true",
            include_stacktrace=os.environ.get("OMOAI_LOG_STACKTRACE", "true").lower() == "true",
            enable_metrics=os.environ.get("OMOAI_LOG_METRICS", "true").lower() == "true",
            metrics_interval=int(os.environ.get("OMOAI_LOG_METRICS_INTERVAL", "60")),
            debug_mode=os.environ.get("OMOAI_DEBUG", "false").lower() == "true",
            quiet_mode=os.environ.get("OMOAI_QUIET", "false").lower() == "true",
        )
    
    def get_log_level(self) -> int:
        """Get numeric log level."""
        if self.debug_mode:
            return logging.DEBUG
        if self.quiet_mode:
            return logging.ERROR
        
        return getattr(logging, self.level.upper(), logging.INFO)
    
    def should_log_performance(self, duration_ms: float) -> bool:
        """Check if performance should be logged based on threshold."""
        return self.enable_performance_logging and duration_ms >= self.performance_threshold_ms


def get_logging_config() -> LoggingConfig:
    """Get logging configuration with environment overrides."""
    # Start with environment-based config
    config = LoggingConfig.from_environment()
    
    # Try to get additional settings from main config
    try:
        main_config = get_config()
        if hasattr(main_config, 'logging'):
            # If logging config exists in main config, merge it
            logging_dict = main_config.logging.model_dump() if hasattr(main_config.logging, 'model_dump') else main_config.logging.__dict__
            for key, value in logging_dict.items():
                if hasattr(config, key) and value is not None:
                    setattr(config, key, value)
    except Exception:
        # If main config unavailable, use environment-only config
        pass
    
    return config


def configure_python_logging(config: LoggingConfig) -> None:
    """Configure Python's built-in logging system."""
    # Set root logger level
    root_logger = logging.getLogger()
    root_logger.setLevel(config.get_log_level())
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler if enabled
    if config.enable_console and not config.quiet_mode:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(config.get_log_level())
        
        if config.format_type == "json":
            from .formatters import JSONFormatter
            console_handler.setFormatter(JSONFormatter())
        elif config.format_type == "structured":
            from .formatters import StructuredFormatter
            console_handler.setFormatter(StructuredFormatter())
        else:
            formatter = logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            console_handler.setFormatter(formatter)
        
        root_logger.addHandler(console_handler)
    
    # Add file handler if enabled
    if config.enable_file and config.log_file:
        from logging.handlers import RotatingFileHandler
        
        # Ensure log directory exists
        config.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            config.log_file,
            maxBytes=config.max_file_size,
            backupCount=config.backup_count
        )
        file_handler.setLevel(config.get_log_level())
        
        # File output is always JSON for machine processing
        from .formatters import JSONFormatter
        file_handler.setFormatter(JSONFormatter())
        
        root_logger.addHandler(file_handler)
    
    # Configure third-party loggers
    _configure_third_party_loggers(config)


def _configure_third_party_loggers(config: LoggingConfig) -> None:
    """Configure logging levels for third-party libraries."""
    # Reduce noise from verbose libraries
    noisy_loggers = [
        "urllib3.connectionpool",
        "requests.packages.urllib3",
        "httpx",
        "httpcore",
        "torch",
        "transformers",
        "pydantic",
        "uvicorn.access",
        "litestar.middleware",
    ]
    
    for logger_name in noisy_loggers:
        logger = logging.getLogger(logger_name)
        if config.debug_mode:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.WARNING)
    
    # Special handling for specific loggers
    if not config.debug_mode:
        # Quiet down transformers tokenizer warnings
        logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
        
        # Reduce torch compilation noise
        logging.getLogger("torch._inductor").setLevel(logging.WARNING)
        
        # Reduce pydantic validation noise in production
        logging.getLogger("pydantic.main").setLevel(logging.WARNING)
