"""
Custom logging formatters for structured logging.
"""
import json
import logging
import sys
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def __init__(self, include_extra: bool = True):
        super().__init__()
        self.include_extra = include_extra
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Base log data
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread,
            "thread_name": record.threadName,
            "process": record.process,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info) if record.exc_info else None,
            }
        
        # Add extra fields if enabled
        if self.include_extra:
            extra_fields = {}
            for key, value in record.__dict__.items():
                if key not in {
                    'name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 'filename',
                    'module', 'exc_info', 'exc_text', 'stack_info', 'lineno', 'funcName',
                    'created', 'msecs', 'relativeCreated', 'thread', 'threadName',
                    'processName', 'process', 'message'
                }:
                    try:
                        # Attempt to serialize the value to ensure it's JSON-compatible
                        json.dumps(value)
                        extra_fields[key] = value
                    except (TypeError, ValueError):
                        # If not serializable, convert to string
                        extra_fields[key] = str(value)
            
            if extra_fields:
                log_data["extra"] = extra_fields
        
        return json.dumps(log_data, separators=(',', ':'))


class StructuredFormatter(logging.Formatter):
    """Human-readable structured formatter."""
    
    def __init__(self, include_extra: bool = True, color: bool = None):
        super().__init__()
        self.include_extra = include_extra
        # Auto-detect color support
        self.color = color if color is not None else (
            hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
        )
        
        # Color codes
        self.colors = {
            'DEBUG': '\033[36m',    # Cyan
            'INFO': '\033[32m',     # Green
            'WARNING': '\033[33m',  # Yellow
            'ERROR': '\033[31m',    # Red
            'CRITICAL': '\033[35m', # Magenta
            'RESET': '\033[0m',     # Reset
        }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with structure."""
        # Timestamp
        timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        
        # Level with color
        level = record.levelname
        if self.color and level in self.colors:
            level = f"{self.colors[level]}{level}{self.colors['RESET']}"
        
        # Logger name (shortened)
        logger_name = record.name
        if logger_name.startswith('src.omoai.'):
            logger_name = logger_name[10:]  # Remove 'src.omoai.' prefix
        
        # Location info
        location = f"{record.module}:{record.funcName}:{record.lineno}"
        
        # Main message
        message = record.getMessage()
        
        # Build base log line
        log_line = f"{timestamp} [{level:>7}] {logger_name:<20} {location:<30} {message}"
        
        # Add extra fields if enabled
        if self.include_extra:
            extra_parts = []
            for key, value in record.__dict__.items():
                if key not in {
                    'name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 'filename',
                    'module', 'exc_info', 'exc_text', 'stack_info', 'lineno', 'funcName',
                    'created', 'msecs', 'relativeCreated', 'thread', 'threadName',
                    'processName', 'process', 'message'
                }:
                    if key.startswith('_'):
                        continue
                    extra_parts.append(f"{key}={value}")
            
            if extra_parts:
                log_line += f" | {' '.join(extra_parts)}"
        
        # Add exception info if present
        if record.exc_info:
            log_line += "\n" + self.formatException(record.exc_info)
        
        return log_line


class PerformanceFormatter(logging.Formatter):
    """Specialized formatter for performance logs."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format performance-specific log records."""
        if hasattr(record, 'operation') and hasattr(record, 'duration_ms'):
            # Performance log
            timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime('%H:%M:%S.%f')[:-3]
            operation = getattr(record, 'operation', 'unknown')
            duration = getattr(record, 'duration_ms', 0)
            
            # Performance level indicator
            if duration > 1000:  # > 1 second
                perf_level = "SLOW"
            elif duration > 100:  # > 100ms
                perf_level = "NORMAL"
            else:
                perf_level = "FAST"
            
            base_msg = f"{timestamp} [PERF:{perf_level}] {operation} completed in {duration:.2f}ms"
            
            # Add optional fields
            extra_fields = []
            for field in ['real_time_factor', 'throughput', 'memory_usage', 'gpu_usage']:
                if hasattr(record, field):
                    value = getattr(record, field)
                    if field == 'real_time_factor':
                        extra_fields.append(f"rtf={value:.2f}")
                    elif field == 'throughput':
                        extra_fields.append(f"throughput={value:.1f}/s")
                    elif field == 'memory_usage':
                        extra_fields.append(f"mem={value:.1f}MB")
                    elif field == 'gpu_usage':
                        extra_fields.append(f"gpu={value:.1f}%")
            
            if extra_fields:
                base_msg += f" ({', '.join(extra_fields)})"
            
            return base_msg
        else:
            # Regular log, fall back to structured format
            return super().format(record)


class ErrorFormatter(logging.Formatter):
    """Specialized formatter for error logs."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format error-specific log records."""
        timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        
        # Error categorization
        error_type = getattr(record, 'error_type', 'UNKNOWN')
        error_code = getattr(record, 'error_code', None)
        
        base_msg = f"{timestamp} [ERROR:{error_type}]"
        if error_code:
            base_msg += f" [{error_code}]"
        
        base_msg += f" {record.getMessage()}"
        
        # Add context if available
        context_fields = []
        for field in ['request_id', 'user_id', 'operation', 'input_file', 'stage']:
            if hasattr(record, field):
                value = getattr(record, field)
                context_fields.append(f"{field}={value}")
        
        if context_fields:
            base_msg += f" | Context: {', '.join(context_fields)}"
        
        # Add remediation hint if available
        if hasattr(record, 'remediation'):
            base_msg += f" | Fix: {getattr(record, 'remediation')}"
        
        # Add exception info
        if record.exc_info:
            base_msg += "\n" + self.formatException(record.exc_info)
        
        return base_msg
