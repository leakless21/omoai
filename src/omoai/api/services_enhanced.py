"""
Enhanced services module with backward compatibility.

This module provides both the original script-based services and new 
high-performance in-memory services with automatic fallback.
"""
import os
import asyncio
from typing import Any, Dict, Optional, Union
from pathlib import Path
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
    PostprocessResponse
)

# Import both versions
from .services import (
    preprocess_audio_service as preprocess_v1,
    asr_service as asr_v1,
    postprocess_service as postprocess_v1,
    run_full_pipeline as pipeline_v1
)

from .services_v2 import (
    preprocess_audio_service_v2 as preprocess_v2,
    asr_service_v2 as asr_v2,
    postprocess_service_v2 as postprocess_v2,
    run_full_pipeline_v2 as pipeline_v2,
    health_check_models
)


class ServiceMode:
    """Service mode configuration."""
    SCRIPT_BASED = "script"
    IN_MEMORY = "memory"
    AUTO = "auto"


class _BytesUploadFile(UploadFile):
    """
    Lightweight UploadFile subclass for tests/benchmarks that supplies the minimal
    surface we need (async read plus filename/content_type). This avoids relying
    on specific UploadFile constructor signatures across litestar versions.
    """
    def __init__(self, data: bytes, filename: str = "upload.bin", content_type: str = "application/octet-stream"):
        # Do not call super().__init__ to avoid signature instability; just set attributes used downstream
        self._data = data
        self.filename = filename
        self.content_type = content_type

    async def read(self, size: int = -1) -> bytes:
        # Match common UploadFile signature shape; return full data when size < 0
        if size is None or size < 0:
            return self._data
        return self._data[:size]


def get_service_mode() -> str:
    """
    Determine which service mode to use.

    Priority:
      1) Environment variable OMOAI_SERVICE_MODE (if set and valid)
      2) Config value config.api.service_mode
      3) Fallback to auto

    Returns:
        Service mode: 'script', 'memory', or 'auto'
    """
    # 1) Respect explicit environment override when present
    env_mode = os.environ.get("OMOAI_SERVICE_MODE")
    if env_mode:
        env_mode = str(env_mode).lower()
        if env_mode in [ServiceMode.SCRIPT_BASED, ServiceMode.IN_MEMORY, ServiceMode.AUTO]:
            return env_mode

    # 2) Fall back to configuration
    try:
        config = get_config()
        mode = getattr(config.api, "service_mode", None) or "auto"
        mode = str(mode).lower()
    except Exception:
        # If config cannot be loaded, fall back to auto
        mode = ServiceMode.AUTO

    if mode in [ServiceMode.SCRIPT_BASED, ServiceMode.IN_MEMORY, ServiceMode.AUTO]:
        return mode

    return ServiceMode.AUTO


async def should_use_in_memory_service() -> bool:
    """
    Determine if in-memory services should be used.
    
    Returns:
        True if in-memory services are available and should be used
    """
    mode = get_service_mode()
    
    if mode == ServiceMode.SCRIPT_BASED:
        return False
    elif mode == ServiceMode.IN_MEMORY:
        return True
    else:  # AUTO mode
        try:
            # Check if models can be loaded
            health = await health_check_models()
            return health["status"] in ["healthy", "degraded"]
        except Exception:
            return False


async def preprocess_audio_service(data: PreprocessRequest) -> PreprocessResponse:
    """
    Smart preprocessing service with automatic fallback.
    
    Tries in-memory processing first, falls back to script-based if needed.
    """
    if await should_use_in_memory_service():
        try:
            return await preprocess_v2(data)
        except Exception as e:
            print(f"Warning: In-memory preprocessing failed, falling back to script: {e}")
    
    # Fallback to script-based service
    return await preprocess_v1(data)


async def asr_service(data: ASRRequest) -> ASRResponse:
    """
    Smart ASR service with automatic fallback.
    
    Tries cached model first, falls back to script-based if needed.
    """
    if await should_use_in_memory_service():
        try:
            return await asr_v2(data)
        except Exception as e:
            print(f"Warning: In-memory ASR failed, falling back to script: {e}")
    
    # Fallback to script-based service  
    return asr_v1(data)


async def postprocess_service(data: PostprocessRequest, output_params: Optional[dict] = None) -> PostprocessResponse:
    """
    Smart postprocessing service with automatic fallback.
    
    Tries cached models first, falls back to script-based if needed.
    
    Accepts optional output_params and forwards them to underlying implementations.
    """
    if await should_use_in_memory_service():
        try:
            # Forward output_params if supported by v2 implementation
            try:
                return await postprocess_v2(data, output_params)  # type: ignore
            except TypeError:
                return await postprocess_v2(data)  # fallback if v2 doesn't accept params
        except Exception as e:
            print(f"Warning: In-memory postprocessing failed, falling back to script: {e}")
    
    # Fallback to script-based service; attempt to forward output_params if accepted
    try:
        return postprocess_v1(data, output_params)  # type: ignore
    except TypeError:
        return postprocess_v1(data)


async def run_full_pipeline(data: PipelineRequest, output_params: Optional[dict] = None) -> PipelineResponse:
    """
    Smart full pipeline with automatic fallback.

    Tries high-performance in-memory pipeline first, falls back to script-based.

    Accepts optional output_params and forwards them to underlying implementations.
    """

    # Guard: ensure uploaded audio is not empty to avoid ffmpeg/script failures later.
    # Read the full upload, validate length, and wrap it back into a lightweight UploadFile wrapper.
    if hasattr(data, "audio_file") and data.audio_file is not None:
        try:
            # Read the full content (consumes the UploadFile stream)
            file_bytes = await data.audio_file.read()
        except Exception:
            # If reading fails, raise a clear processing error
            raise AudioProcessingException("Failed to read uploaded audio file")

        if not file_bytes or len(file_bytes) == 0:
            raise AudioProcessingException("Uploaded audio file is empty")

        # Replace with a stable in-memory UploadFile-compatible wrapper so downstream code can re-read
        data.audio_file = _BytesUploadFile(
            data=file_bytes,
            filename=getattr(data.audio_file, "filename", "upload.bin"),
            content_type=getattr(data.audio_file, "content_type", "application/octet-stream"),
        )

    if await should_use_in_memory_service():
        try:
            # Forward output_params if supported by v2 implementation
            try:
                return await pipeline_v2(data, output_params)  # type: ignore
            except TypeError:
                return await pipeline_v2(data)  # fallback if v2 doesn't accept params
        except Exception as e:
            print(f"Warning: In-memory pipeline failed, falling back to script: {e}")

    # Fallback to script-based pipeline; attempt to forward output_params if accepted
    try:
        return await pipeline_v1(data, output_params)  # type: ignore
    except TypeError:
        return await pipeline_v1(data)


async def get_service_status() -> Dict[str, Any]:
    """
    Get comprehensive service status and configuration.
    
    Returns:
        Dictionary with service mode, model status, and performance info
    """
    mode = get_service_mode()
    in_memory_available = await should_use_in_memory_service()
    
    status = {
        "service_mode": mode,
        "in_memory_available": in_memory_available,
        "active_backend": "memory" if in_memory_available else "script",
        "performance_mode": "high" if in_memory_available else "standard",
    }
    
    if in_memory_available:
        try:
            model_health = await health_check_models()
            status["models"] = model_health
        except Exception as e:
            status["models"] = {"error": str(e)}
    else:
        status["models"] = {"status": "script_based"}
    
    # Configuration info
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


async def force_service_mode(mode: str) -> Dict[str, Any]:
    """
    Force a specific service mode (for testing/debugging).
    
    Args:
        mode: 'script', 'memory', or 'auto'
        
    Returns:
        Status after mode change
    """
    if mode not in [ServiceMode.SCRIPT_BASED, ServiceMode.IN_MEMORY, ServiceMode.AUTO]:
        raise ValueError(f"Invalid service mode: {mode}")
    
    os.environ["OMOAI_SERVICE_MODE"] = mode
    
    return {
        "previous_mode": get_service_mode(),
        "new_mode": mode,
        "status": await get_service_status()
    }


async def warmup_services() -> Dict[str, Any]:
    """
    Warm up services by preloading models.
    
    This should be called during application startup for best performance.
    
    Returns:
        Warmup results and timing information
    """
    import time
    
    start_time = time.time()
    
    try:
        from .singletons import preload_all_models
        
        # Preload models in executor to avoid blocking
        loop = asyncio.get_event_loop()
        model_results = await loop.run_in_executor(None, preload_all_models)
        
        warmup_time = time.time() - start_time
        
        return {
            "success": True,
            "warmup_time": warmup_time,
            "models_loaded": model_results,
            "service_status": await get_service_status()
        }
        
    except Exception as e:
        warmup_time = time.time() - start_time
        
        return {
            "success": False,
            "error": str(e),
            "warmup_time": warmup_time,
            "fallback_to_script": True
        }


# Performance monitoring functions
async def benchmark_service_performance(
    test_audio_path: Optional[Union[str, Path]] = None,
    iterations: int = 3
) -> Dict[str, Any]:
    """
    Benchmark performance of both service modes.
    
    Args:
        test_audio_path: Path to test audio file (uses synthetic if None)
        iterations: Number of iterations to average
        
    Returns:
        Performance comparison results
    """
    import time
    import tempfile
    from litestar.datastructures import UploadFile
    
    results = {
        "iterations": iterations,
        "memory_service": {"available": False, "times": []},
        "script_service": {"available": True, "times": []},  # Always available
    }
    
    # Create test audio data
    if test_audio_path:
        with open(test_audio_path, "rb") as f:
            test_data = f.read()
    else:
        # Create minimal test audio (1 second silence WAV)
        test_data = create_test_audio_bytes()
    
    # Test in-memory service
    if await should_use_in_memory_service():
        results["memory_service"]["available"] = True
        
        for i in range(iterations):
            try:
                with tempfile.NamedTemporaryFile() as tmp:
                    tmp.write(test_data)
                    tmp.flush()
                    
                    # Use a lightweight UploadFile-compatible wrapper to avoid constructor mismatches
                    upload_file = _BytesUploadFile(
                        data=test_data,
                        filename="test.wav",
                        content_type="audio/wav",
                    )
                    
                    request = PipelineRequest(audio_file=upload_file)
                    
                    start_time = time.time()
                    
                    # Force memory mode
                    os.environ["OMOAI_SERVICE_MODE"] = ServiceMode.IN_MEMORY
                    await run_full_pipeline(request)
                    
                    elapsed = time.time() - start_time
                    results["memory_service"]["times"].append(elapsed)
                    
            except Exception as e:
                results["memory_service"]["error"] = str(e)
                break
    
    # Test script-based service
    for i in range(iterations):
        try:
            with tempfile.NamedTemporaryFile() as tmp:
                tmp.write(test_data)
                tmp.flush()
                
                # Use a lightweight UploadFile-compatible wrapper to avoid constructor mismatches
                upload_file = _BytesUploadFile(
                    data=test_data,
                    filename="test.wav",
                    content_type="audio/wav",
                )
                
                request = PipelineRequest(audio_file=upload_file)
                
                start_time = time.time()
                
                # Force script mode
                os.environ["OMOAI_SERVICE_MODE"] = ServiceMode.SCRIPT_BASED
                await run_full_pipeline(request)
                
                elapsed = time.time() - start_time
                results["script_service"]["times"].append(elapsed)
                
        except Exception as e:
            results["script_service"]["error"] = str(e)
            break
    
    # Reset to auto mode
    os.environ["OMOAI_SERVICE_MODE"] = ServiceMode.AUTO
    
    # Calculate statistics
    for service in ["memory_service", "script_service"]:
        times = results[service]["times"]
        if times:
            results[service]["avg_time"] = sum(times) / len(times)
            results[service]["min_time"] = min(times)
            results[service]["max_time"] = max(times)
    
    # Calculate speedup
    if (results["memory_service"]["times"] and results["script_service"]["times"]):
        memory_avg = results["memory_service"]["avg_time"]
        script_avg = results["script_service"]["avg_time"]
        results["speedup"] = script_avg / memory_avg
    
    return results


def create_test_audio_bytes() -> bytes:
    """Create minimal test audio data for benchmarking."""
    # Create a minimal WAV file (1 second of silence at 16kHz)
    import struct
    
    sample_rate = 16000
    duration = 1.0
    samples = int(sample_rate * duration)
    
    # WAV header
    header = b'RIFF'
    header += struct.pack('<I', 36 + samples * 2)  # File size
    header += b'WAVE'
    header += b'fmt '
    header += struct.pack('<I', 16)  # Subchunk1 size
    header += struct.pack('<H', 1)   # Audio format (PCM)
    header += struct.pack('<H', 1)   # Number of channels
    header += struct.pack('<I', sample_rate)  # Sample rate
    header += struct.pack('<I', sample_rate * 2)  # Byte rate
    header += struct.pack('<H', 2)   # Block align
    header += struct.pack('<H', 16)  # Bits per sample
    header += b'data'
    header += struct.pack('<I', samples * 2)  # Data size
    
    # Audio data (silence)
    audio_data = b'\x00' * (samples * 2)
    
    return header + audio_data
