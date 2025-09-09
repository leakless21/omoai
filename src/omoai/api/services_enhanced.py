"""
Services module providing script-based implementations.

This module provides direct access to the script-based services without
any in-memory fallback complexity, plus minimal control helpers used by tests.
"""
from __future__ import annotations

import os
import asyncio
import time
from typing import Any, Dict, Optional, Union
from litestar.datastructures import UploadFile

from ..config import get_config
from .exceptions import AudioProcessingException
from .models import (
    PipelineRequest,
    PipelineResponse,
    PreprocessRequest,
    PreprocessResponse,
    ASRRequest,
    ASRResponse,
    PostprocessRequest,
    PostprocessResponse,
    OutputFormatParams,
)

from .services import (
    preprocess_audio_service as _preprocess_audio_script,
    asr_service as _asr_script,
    postprocess_service as _postprocess_script,
    run_full_pipeline as _run_full_pipeline_script,
)


class ServiceMode:
    AUTO = "AUTO"
    SCRIPT_BASED = "SCRIPT_BASED"
    IN_MEMORY = "IN_MEMORY"


def get_service_mode() -> str:
    """
    Resolve service mode from env, defaulting to AUTO.
    """
    mode = os.environ.get("OMOAI_SERVICE_MODE", ServiceMode.AUTO)
    if mode in (ServiceMode.AUTO, ServiceMode.SCRIPT_BASED, ServiceMode.IN_MEMORY):
        return mode
    return ServiceMode.AUTO


async def should_use_in_memory_service() -> bool:
    """
    Decide whether to use the in-memory backend.

    - If mode is SCRIPT_BASED -> False
    - If mode is IN_MEMORY -> True
    - If AUTO -> consult services_v2.health_check_models()
      (supports both sync and async; patched in tests)
    """
    mode = get_service_mode()
    if mode == ServiceMode.SCRIPT_BASED:
        return False
    if mode == ServiceMode.IN_MEMORY:
        return True

    # AUTO: probe v2 health (allow sync or async function, supports patching)
    try:
        from . import services_v2 as _v2
        result = _v2.health_check_models()
        if asyncio.iscoroutine(result):
            result = await result
        return isinstance(result, dict) and result.get("status") == "healthy"
    except Exception:
        return False


async def warmup_services() -> Dict[str, Any]:
    """
    Preload models via singletons and report status.
    Tests patch src.omoai.api.singletons.preload_all_models, so import the module (not the symbol).
    """
    start = time.perf_counter()
    try:
        from . import singletons  # import module so patching works
        models_loaded = singletons.preload_all_models()
        elapsed = time.perf_counter() - start
        status = await get_service_status()
        return {
            "success": True,
            "warmup_time": float(max(elapsed, 1e-6)),
            "models_loaded": models_loaded,
            "service_status": status,
        }
    except Exception as e:
        elapsed = time.perf_counter() - start
        status = await get_service_status()
        return {
            "success": False,
            "warmup_time": float(max(elapsed, 1e-6)),
            "error": str(e),
            "fallback_to_script": True,
            "service_status": status,
        }


class _BytesUploadFile(UploadFile):
    """
    Lightweight UploadFile subclass for tests/benchmarks that supplies the minimal
    surface we need (async read plus filename/content_type).
    """
    def __init__(self, data: bytes, filename: str = "upload.bin", content_type: str = "application/octet-stream"):
        self._data = data
        self.filename = filename
        self.content_type = content_type

    async def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            return self._data
        return self._data[:size]


async def preprocess_audio_service(data: PreprocessRequest) -> PreprocessResponse:
    """Preprocess via script-based implementation."""
    return await _preprocess_audio_script(data)


async def asr_service(data: ASRRequest) -> ASRResponse:
    """ASR via script-based implementation (offloaded to thread)."""
    return await asyncio.to_thread(_asr_script, data)


async def postprocess_service(data: PostprocessRequest, output_params: Optional[dict] = None) -> PostprocessResponse:
    """
    Postprocess via script-based implementation.
    output_params is accepted for compatibility but ignored.
    """
    return await asyncio.to_thread(_postprocess_script, data)


async def run_full_pipeline(
    data: PipelineRequest,
    output_params: Optional[Union[OutputFormatParams, dict]] = None,
) -> PipelineResponse:
    """
    Full pipeline via script-based implementation. If output_params is not an
    OutputFormatParams instance, it will be ignored.
    """
    # Ensure uploaded audio can be re-read by downstream code
    if hasattr(data, "audio_file") and data.audio_file is not None:
        try:
            file_bytes = await data.audio_file.read()
            if not file_bytes:
                raise AudioProcessingException("Uploaded audio file is empty")
        except AudioProcessingException:
            raise
        except Exception:
            raise AudioProcessingException("Failed to read uploaded audio file")

        data.audio_file = _BytesUploadFile(
            data=file_bytes,
            filename=getattr(data.audio_file, "filename", "upload.bin"),
            content_type=getattr(data.audio_file, "content_type", "application/octet-stream"),
        )

    params: Optional[OutputFormatParams] = output_params if isinstance(output_params, OutputFormatParams) else None
    return await _run_full_pipeline_script(data, params)


async def get_service_status() -> Dict[str, Any]:
    """Get service status and configuration for script backend."""
    status: Dict[str, Any] = {
        "service_mode": ServiceMode.SCRIPT_BASED,
        "active_backend": "script",
        "performance_mode": "standard",
        "models": {"status": "script_based"},
    }
    try:
        config = get_config()
        status["config"] = {
            "api_host": config.api.host,
            "temp_dir": str(config.api.temp_dir),
            "cleanup_temp_files": config.api.cleanup_temp_files,
            "progress_output": config.api.enable_progress_output,
        }
    except Exception as e:
        status["config"] = {"error": str(e)}
    return status


__all__ = [
    "ServiceMode",
    "get_service_mode",
    "should_use_in_memory_service",
    "warmup_services",
    "preprocess_audio_service",
    "asr_service",
    "postprocess_service",
    "run_full_pipeline",
    "get_service_status",
]
