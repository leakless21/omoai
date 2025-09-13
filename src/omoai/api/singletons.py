"""Lightweight singletons and accessors used by tests.

This stub provides a minimal API surface expected by tests:
- ModelSingletons singleton with thread safety
- get_* accessors
- preload_all_models()
- get_model_status()
"""

from __future__ import annotations

import threading
from typing import Any, ClassVar


class ModelSingletons:
    """Thread-safe singleton container for model-like objects."""

    _instance: ClassVar[ModelSingletons | None] = None
    _class_lock: ClassVar[threading.Lock] = threading.Lock()

    # Instance attributes (declared for type checkers)
    _init_lock: threading.Lock
    _initialized: bool
    _asr_model: Any | None
    _punctuation_processor: Any | None
    _summarization_processor: Any | None

    def __new__(cls) -> ModelSingletons:
        with cls._class_lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                # Initialize expected fields
                instance._init_lock = threading.Lock()
                instance._initialized = False
                instance._asr_model = None
                instance._punctuation_processor = None
                instance._summarization_processor = None
                cls._instance = instance
            return cls._instance


def get_asr_model():
    """Return ASR model instance if initialized, else None."""
    return ModelSingletons()._asr_model


def get_punctuation_processor():
    """Return punctuation processor if initialized, else None."""
    return ModelSingletons()._punctuation_processor


def get_summarization_processor():
    """Return summarization processor if initialized, else None."""
    return ModelSingletons()._summarization_processor


def preload_all_models() -> dict[str, bool]:
    """Simulate loading models; tests may patch this to inject failures."""
    return {"asr": True, "punctuation": True, "summarization": True}


def get_model_status() -> dict[str, Any]:
    s = ModelSingletons()
    return {
        "asr_loaded": s._asr_model is not None,
        "punctuation_loaded": s._punctuation_processor is not None,
        "summarization_loaded": s._summarization_processor is not None,
        "initialized": getattr(s, "_initialized", False),
    }


__all__ = [
    "ModelSingletons",
    "get_asr_model",
    "get_model_status",
    "get_punctuation_processor",
    "get_summarization_processor",
    "preload_all_models",
]
