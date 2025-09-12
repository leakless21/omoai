"""
Custom serializers and formatters for OMOAI logging.
"""
from __future__ import annotations

import json
import logging
import traceback as _traceback
from datetime import datetime
from typing import Any, Dict


def _iso_timestamp(dt: datetime) -> str:
    try:
        # Prefer ISO with milliseconds
        return dt.isoformat(timespec="milliseconds")
    except Exception:
        return dt.isoformat()


def flat_json_serializer(record: Dict[str, Any]) -> str:
    """
    Loguru-compatible serializer that returns a JSON line per record.

    Expected keys (validated by tests): timestamp, level, message, extra
    Additional keys: logger, module, function, line, exception (if any)
    """
    # Timestamp handling: Loguru provides a time object or datetime
    time_obj = record.get("time")
    if isinstance(time_obj, datetime):
        ts = _iso_timestamp(time_obj)
    elif hasattr(time_obj, "datetime"):
        try:
            ts = _iso_timestamp(time_obj.datetime())
        except Exception:
            ts = _iso_timestamp(datetime.utcnow())
    else:
        ts = _iso_timestamp(datetime.utcnow())

    # Level handling: Loguru's level is an object with a .name
    level_obj = record.get("level")
    level_name = "INFO"
    if hasattr(level_obj, "name"):
        level_name = getattr(level_obj, "name", "INFO")
    elif isinstance(level_obj, dict):
        level_name = level_obj.get("name", "INFO")
    elif level_obj:
        level_name = str(level_obj)

    data: Dict[str, Any] = {
        "timestamp": ts,
        "level": level_name,
        "logger": record.get("name", "omoai"),
        "message": record.get("message", ""),
        "extra": record.get("extra", {}) or {},
    }

    # Location details when available
    # Location info may be present at top-level in Loguru's record dict
    data["module"] = record.get("module")
    data["function"] = record.get("function")
    data["line"] = record.get("line")

    # Exception info if present
    exc = record.get("exception")
    if exc:
        _type, _value, _tb = exc
        data["exception"] = {
            "type": getattr(_type, "__name__", str(_type)),
            "message": str(_value),
            "traceback": "".join(_traceback.format_exception(_type, _value, _tb)),
        }

    s = json.dumps(data, ensure_ascii=False)
    # Escape braces so Loguru's formatter does not try to format our JSON
    s = s.replace("{", "{{").replace("}", "}}")
    return s + "\n"


class JSONFormatter(logging.Formatter):
    """Stdlib logging formatter that emits flat JSON lines with extras."""

    DEFAULT_KEYS = {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename", "module",
        "exc_info", "exc_text", "stack_info", "lineno", "funcName", "created", "msecs",
        "relativeCreated", "thread", "threadName", "processName", "process",
    }

    def format(self, record: logging.LogRecord) -> str:
        data: Dict[str, Any] = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": getattr(record, "module", None),
            "function": getattr(record, "funcName", None),
            "line": getattr(record, "lineno", None),
        }

        # Capture extras: any non-default attributes on the record
        extras: Dict[str, Any] = {}
        for k, v in record.__dict__.items():
            if k not in self.DEFAULT_KEYS and not k.startswith("_"):
                extras[k] = v
        if extras:
            data["extra"] = extras

        # Exception information
        if record.exc_info:
            _type, _value, _tb = record.exc_info
            data["exception"] = {
                "type": getattr(_type, "__name__", str(_type)),
                "message": str(_value),
                "traceback": self.formatException(record.exc_info),
            }

        return json.dumps(data, ensure_ascii=False)


class StructuredFormatter(logging.Formatter):
    """Human-readable structured formatter with optional color (disabled in tests)."""

    def __init__(self, color: bool = False):
        super().__init__()
        self.color = color

    def format(self, record: logging.LogRecord) -> str:
        level = record.levelname
        # Shorten logger name by dropping leading namespaces like 'src.' or 'omoai.' when useful
        logger_name = record.name
        for prefix in ("src.", "omoai."):
            if logger_name.startswith(prefix):
                logger_name = logger_name[len(prefix):]
        location = f"{getattr(record, 'module', '')}:{getattr(record, 'funcName', '')}:{getattr(record, 'lineno', '')}"
        base = f"{level} | {logger_name} | {location} - {record.getMessage()}"

        # Append simple extras as key=value
        extras = []
        for k, v in record.__dict__.items():
            if k in JSONFormatter.DEFAULT_KEYS or k.startswith("_"):
                continue
            # Avoid dumping huge or complex objects
            if isinstance(v, (str, int, float, bool)):
                extras.append(f"{k}={v}")
        if extras:
            base += " " + " ".join(extras)
        return base
