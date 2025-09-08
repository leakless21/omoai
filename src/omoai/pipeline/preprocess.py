"""
In-memory audio preprocessing for OMOAI.

This module provides efficient audio preprocessing that works with in-memory
data streams and tensors, eliminating the need for intermediate file I/O.
"""
import io
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import BinaryIO, Union, Tuple, Optional

import torch
import numpy as np
from pydub import AudioSegment  # type: ignore

# Add centralized logging imports
from ..config import OmoAIConfig
from ..logging_system import get_logger, performance_context, log_error, get_performance_logger
from .exceptions import OMOAudioError, OMOConfigError, OMOIOError


def preprocess_audio_bytes(
    audio_data: bytes,
    target_sample_rate: int = 16000,
    target_channels: int = 1,
    target_bit_depth: int = 16,
) -> bytes:
    """
    Preprocess audio bytes to target format using ffmpeg in-memory processing.
    
    Args:
        audio_data: Raw audio bytes in any supported format
        target_sample_rate: Target sample rate (default: 16kHz)
        target_channels: Target channel count (default: 1 = mono)
        target_bit_depth: Target bit depth (default: 16)
        
    Returns:
        Preprocessed audio bytes in PCM16 WAV format
        
    Raises:
        OMOAudioError: If audio processing fails
        OMOIOError: If I/O operations fail
        ValueError: If audio data is invalid
    """
    # Initialize logging
    logger = get_logger("omoai.pipeline.preprocess")
    perf_logger = get_performance_logger()
    
    # Generate unique preprocessing ID for tracing
    preprocess_id = str(uuid.uuid4())
    
    logger.info("Starting audio preprocessing", extra={
        "preprocess_id": preprocess_id,
        "input_size_bytes": len(audio_data),
        "target_sample_rate": target_sample_rate,
        "target_channels": target_channels,
        "target_bit_depth": target_bit_depth,
    })
    
    timing = {}
    start_time = time.time()
    
    try:
        if not audio_data:
            raise OMOAudioError("Audio data is empty")
        
        # Use ffmpeg with pipes for in-memory processing
        cmd = [
            "ffmpeg",
            "-f", "auto",  # Auto-detect input format
            "-i", "pipe:0",  # Read from stdin
            "-ac", str(target_channels),  # Audio channels
            "-ar", str(target_sample_rate),  # Audio sample rate
            "-vn",  # No video
            "-c:a", "pcm_s16le",  # PCM 16-bit little endian
            "-f", "wav",  # Output format
            "-y",  # Overwrite output
            "pipe:1"  # Write to stdout
        ]
        
        with performance_context("ffmpeg_preprocessing", logger=logger):
            preprocess_start = time.time()
            result = subprocess.run(
                cmd,
                input=audio_data,
                capture_output=True,
                check=True
            )
            timing["ffmpeg"] = time.time() - preprocess_start
            
            if not result.stdout:
                raise OMOAudioError("ffmpeg produced no output")
                
            logger.debug("FFmpeg preprocessing completed", extra={
                "preprocess_id": preprocess_id,
                "ffmpeg_time_ms": timing["ffmpeg"] * 1000,
                "output_size_bytes": len(result.stdout),
            })
            
            # Calculate total timing
            timing["total"] = time.time() - start_time
            
            # Log performance metrics
            perf_logger.log_operation(
                operation="audio_preprocessing",
                duration_ms=timing["total"] * 1000,
                success=True,
                preprocess_id=preprocess_id,
                stages_count=len(timing),
                input_size_bytes=len(audio_data),
                output_size_bytes=len(result.stdout),
                real_time_factor=timing["total"] / (len(audio_data) / (target_sample_rate * target_channels * target_bit_depth / 8)) if target_sample_rate > 0 and target_channels > 0 and target_bit_depth > 0 else 0,
            )
            
            logger.info("Audio preprocessing completed successfully", extra={
                "preprocess_id": preprocess_id,
                "total_time_ms": timing["total"] * 1000,
                "stages_completed": list(timing.keys()),
                "input_size_bytes": len(audio_data),
                "output_size_bytes": len(result.stdout),
                "compression_ratio": len(result.stdout) / len(audio_data) if len(audio_data) > 0 else 0,
            })
            
            return result.stdout
            
    except subprocess.CalledProcessError as e:
        error_timing = time.time() - start_time
        
        error_msg = e.stderr.decode('utf-8', errors='ignore') if e.stderr else "Unknown error"
        
        # Log detailed error information
        log_error(
            message=f"FFmpeg preprocessing failed after {error_timing:.2f}s",
            error=e,
            error_type="FFMPEG_PREPROCESSING_FAILURE",
            error_code="PREPROCESS_001",
            remediation="Check input validity and ffmpeg installation",
            logger=logger,
            preprocess_id=preprocess_id,
            stages_completed=list(timing.keys()),
            error_timing_seconds=error_timing,
        )
        
        # Record failed operation
        perf_logger.log_operation(
            operation="audio_preprocessing",
            duration_ms=error_timing * 1000,
            success=False,
            error_type="FFMPEG_PREPROCESSING_FAILURE",
            preprocess_id=preprocess_id,
            stages_completed=list(timing.keys()),
        )
        
        raise subprocess.CalledProcessError(
            e.returncode,
            cmd,
            f"ffmpeg preprocessing failed: {error_msg}"
        )
        
    except Exception as e:
        error_timing = time.time() - start_time
        
        # Log detailed error information
        log_error(
            message=f"Audio preprocessing failed after {error_timing:.2f}s",
            error=e,
            error_type="AUDIO_PREPROCESSING_FAILURE",
            error_code="PREPROCESS_002",
            remediation="Check input validity and preprocessing parameters",
            logger=logger,
            preprocess_id=preprocess_id,
            stages_completed=list(timing.keys()),
            error_timing_seconds=error_timing,
        )
        
        # Record failed operation
        perf_logger.log_operation(
            operation="audio_preprocessing",
            duration_ms=error_timing * 1000,
            success=False,
            error_type="AUDIO_PREPROCESSING_FAILURE",
            preprocess_id=preprocess_id,
            stages_completed=list(timing.keys()),
        )
        
        raise OMOAudioError(f"Failed to preprocess audio: {e}") from e


def preprocess_audio_to_tensor(
    audio_input: Union[bytes, Path, str, BinaryIO],
    target_sample_rate: int = 16000,
    normalize: bool = True,
    return_sample_rate: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, int]]:
    """
    Preprocess audio to PyTorch tensor for direct ASR inference.
    
    Args:
        audio_input: Audio data as bytes, file path, or file-like object
        target_sample_rate: Target sample rate for ASR (default: 16kHz)
        normalize: Whether to normalize audio to [-1, 1] range
        return_sample_rate: Whether to return sample rate along with tensor
        
    Returns:
        torch.Tensor: Audio tensor of shape (1, samples) for mono audio
        If return_sample_rate=True, returns (tensor, sample_rate)
        
    Raises:
        OMOAudioError: If audio processing fails
        OMOIOError: If I/O operations fail
        FileNotFoundError: If file path doesn't exist
        ValueError: If audio data is invalid
    """
    # Initialize logging
    logger = get_logger("omoai.pipeline.preprocess")
    perf_logger = get_performance_logger()
    
    # Generate unique preprocessing ID for tracing
    preprocess_id = str(uuid.uuid4())
    
    logger.info("Starting audio preprocessing to tensor", extra={
        "preprocess_id": preprocess_id,
        "input_type": type(audio_input).__name__,
        "target_sample_rate": target_sample_rate,
        "normalize": normalize,
        "return_sample_rate": return_sample_rate,
    })
    
    timing = {}
    start_time = time.time()
    
    try:
        # Handle different input types
        if isinstance(audio_input, (str, Path)):
            # File path input
            audio_path = Path(audio_input)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            audio = AudioSegment.from_file(str(audio_path))
            
        elif isinstance(audio_input, bytes):
            # Bytes input - use in-memory processing
            audio_io = io.BytesIO(audio_input)
            audio = AudioSegment.from_file(audio_io)
            
        elif hasattr(audio_input, 'read'):
            # File-like object
            audio = AudioSegment.from_file(audio_input)
            
        else:
            raise OMOAudioError(f"Unsupported audio input type: {type(audio_input)}")
        
        # Convert to target format
        audio = audio.set_frame_rate(target_sample_rate)
        audio = audio.set_sample_width(2)  # 16-bit
        audio = audio.set_channels(1)  # Mono
        
        # Convert to tensor
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        
        if normalize:
            # Normalize from int16 range to [-1, 1]
            samples = samples / 32768.0
        
        # Convert to torch tensor with batch dimension
        tensor = torch.from_numpy(samples).unsqueeze(0)  # Shape: (1, samples)
        
        # Calculate timing
        timing["total"] = time.time() - start_time
        
        # Log performance metrics
        perf_logger.log_operation(
            operation="audio_preprocessing_tensor",
            duration_ms=timing["total"] * 1000,
            success=True,
            preprocess_id=preprocess_id,
            stages_count=len(timing),
            input_type=type(audio_input).__name__,
            output_shape=list(tensor.shape),
            output_dtype=str(tensor.dtype),
            real_time_factor=timing["total"] / (len(audio) / 1000.0) if len(audio) > 0 else 0,
        )
        
        logger.info("Audio preprocessing to tensor completed", extra={
            "preprocess_id": preprocess_id,
            "preprocessing_time_ms": timing["total"] * 1000,
            "tensor_shape": list(tensor.shape),
            "output_sample_rate": target_sample_rate,
            "tensor_dtype": str(tensor.dtype),
            "normalize_applied": normalize,
            "real_time_factor": timing["total"] / (len(audio) / 1000.0) if len(audio) > 0 else 0,
        })
        
        if return_sample_rate:
            return tensor, target_sample_rate
        return tensor
        
    except Exception as e:
        error_timing = time.time() - start_time
        
        # Log detailed error information
        log_error(
            message=f"Audio preprocessing to tensor failed after {error_timing:.2f}s",
            error=e,
            error_type="AUDIO_PREPROCESSING_TENSOR_FAILURE",
            error_code="PREPROCESS_003",
            remediation="Check input validity and preprocessing parameters",
            logger=logger,
            preprocess_id=preprocess_id,
            stages_completed=list(timing.keys()),
            error_timing_seconds=error_timing,
        )
        
        # Record failed operation
        perf_logger.log_operation(
            operation="audio_preprocessing_tensor",
            duration_ms=error_timing * 1000,
            success=False,
            error_type="AUDIO_PREPROCESSING_TENSOR_FAILURE",
            preprocess_id=preprocess_id,
            stages_completed=list(timing.keys()),
        )
        
        raise OMOAudioError(f"Failed to preprocess audio to tensor: {e}") from e


def preprocess_file_to_wav_bytes(
    input_path: Union[Path, str],
    target_sample_rate: int = 16000,
) -> bytes:
    """
    Preprocess an audio file to WAV bytes using optimized ffmpeg.
    
    Args:
        input_path: Path to input audio file
        target_sample_rate: Target sample rate (default: 16kHz)
        
    Returns:
        Preprocessed audio as WAV bytes
        
    Raises:
        OMOAudioError: If audio processing fails
        OMOIOError: If I/O operations fail
        FileNotFoundError: If input file doesn't exist
        ValueError: If audio data is invalid
    """
    # Initialize logging
    logger = get_logger("omoai.pipeline.preprocess")
    perf_logger = get_performance_logger()
    
    # Generate unique preprocessing ID for tracing
    preprocess_id = str(uuid.uuid4())
    
    logger.info("Starting file preprocessing to WAV bytes", extra={
        "preprocess_id": preprocess_id,
        "input_path": str(input_path),
        "target_sample_rate": target_sample_rate,
    })
    
    timing = {}
    start_time = time.time()
    
    try:
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input audio file not found: {input_path}")
        
        # Enhanced ffmpeg command with error handling and optimization
        cmd = [
            "ffmpeg",
            "-nostdin",  # Don't read from stdin
            "-hide_banner",  # Hide banner for cleaner output
            "-loglevel", "error",  # Only show errors
            "-i", str(input_path),
            "-ac", "1",  # Mono
            "-ar", str(target_sample_rate),  # Sample rate
            "-vn",  # No video
            "-sn",  # No subtitles
            "-dn",  # No data streams
            "-map_metadata", "-1",  # Remove metadata
            "-c:a", "pcm_s16le",  # PCM 16-bit
            "-f", "wav",  # WAV format
            "-y",  # Overwrite
            "pipe:1"  # Output to stdout
        ]
        
        with performance_context("ffmpeg_file_preprocessing", logger=logger):
            preprocess_start = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                check=True
            )
            timing["ffmpeg"] = time.time() - preprocess_start
            
            if not result.stdout:
                raise OMOAudioError("ffmpeg produced no output")
                
            logger.debug("FFmpeg file preprocessing completed", extra={
                "preprocess_id": preprocess_id,
                "ffmpeg_time_ms": timing["ffmpeg"] * 1000,
                "output_size_bytes": len(result.stdout),
            })
            
            # Calculate total timing
            timing["total"] = time.time() - start_time
            
            # Log performance metrics
            perf_logger.log_operation(
                operation="file_preprocessing_wav_bytes",
                duration_ms=timing["total"] * 1000,
                success=True,
                preprocess_id=preprocess_id,
                stages_count=len(timing),
                input_path=str(input_path),
                output_size_bytes=len(result.stdout),
                target_sample_rate=target_sample_rate,
                real_time_factor=timing["total"] / (len(result.stdout) / (target_sample_rate * 2)) if target_sample_rate > 0 else 0,
            )
            
            logger.info("File preprocessing to WAV bytes completed successfully", extra={
                "preprocess_id": preprocess_id,
                "preprocessing_time_ms": timing["total"] * 1000,
                "stages_completed": list(timing.keys()),
                "output_size_bytes": len(result.stdout),
                "target_sample_rate": target_sample_rate,
                "compression_ratio": len(result.stdout) / input_path.stat().st_size if input_path.exists() else 0,
            })
            
            return result.stdout
            
    except subprocess.CalledProcessError as e:
        error_timing = time.time() - start_time
        error_msg = e.stderr.decode('utf-8', errors='ignore') if e.stderr else "Unknown error"
        
        # Log detailed error information
        log_error(
            message=f"FFmpeg file preprocessing failed after {error_timing:.2f}s",
            error=e,
            error_type="FFMPEG_FILE_PREPROCESSING_FAILURE",
            error_code="PREPROCESS_006",
            remediation=f"Check input file validity ({input_path}) and ffmpeg installation",
            logger=logger,
            preprocess_id=preprocess_id,
            stages_completed=list(timing.keys()),
            error_timing_seconds=error_timing,
            input_path=str(input_path),
        )
        
        # Record failed operation
        perf_logger.log_operation(
            operation="file_preprocessing_wav_bytes",
            duration_ms=error_timing * 1000,
            success=False,
            error_type="FFMPEG_FILE_PREPROCESSING_FAILURE",
            preprocess_id=preprocess_id,
            stages_completed=list(timing.keys()),
            input_path=str(input_path),
        )
        
        raise subprocess.CalledProcessError(
            e.returncode,
            cmd,
            f"ffmpeg preprocessing failed for {input_path}: {error_msg}"
        ) from e
        
    except Exception as e:
        error_timing = time.time() - start_time
        
        # Log detailed error information
        log_error(
            message=f"File preprocessing to WAV bytes failed after {error_timing:.2f}s",
            error=e,
            error_type="FILE_PREPROCESSING_WAV_BYTES_FAILURE",
            error_code="PREPROCESS_007",
            remediation=f"Check input file validity ({input_path}) and preprocessing parameters",
            logger=logger,
            preprocess_id=preprocess_id,
            stages_completed=list(timing.keys()),
            error_timing_seconds=error_timing,
            input_path=str(input_path),
        )
        
        # Record failed operation
        perf_logger.log_operation(
            operation="file_preprocessing_wav_bytes",
            duration_ms=error_timing * 1000,
            success=False,
            error_type="FILE_PREPROCESSING_WAV_BYTES_FAILURE",
            preprocess_id=preprocess_id,
            stages_completed=list(timing.keys()),
            input_path=str(input_path),
        )
        
        raise OMOAudioError(f"Failed to preprocess file to WAV bytes: {e}") from e


def get_audio_info(audio_input: Union[bytes, Path, str]) -> dict:
    """
    Get audio information without full processing.
    
    Args:
        audio_input: Audio data as bytes or file path
        
    Returns:
        Dictionary with audio metadata:
        {
            "duration_seconds": float,
            "sample_rate": int,
            "channels": int,
            "format": str
        }
        
    Raises:
        OMOAudioError: If audio processing fails
        FileNotFoundError: If file path doesn't exist
        ValueError: If audio data is invalid
    """
    # Initialize logging
    logger = get_logger("omoai.pipeline.preprocess")
    perf_logger = get_performance_logger()
    
    # Generate unique info ID for tracing
    info_id = str(uuid.uuid4())
    
    logger.info("Getting audio info", extra={
        "info_id": info_id,
        "input_type": type(audio_input).__name__,
    })
    
    timing = {}
    start_time = time.time()
    
    try:
        if isinstance(audio_input, (str, Path)):
            audio_path = Path(audio_input)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            audio = AudioSegment.from_file(str(audio_path))
        elif isinstance(audio_input, bytes):
            audio_io = io.BytesIO(audio_input)
            audio = AudioSegment.from_file(audio_io)
        else:
            raise OMOAudioError(f"Unsupported input type: {type(audio_input)}")
        
        result = {
            "duration_seconds": len(audio) / 1000.0,
            "sample_rate": audio.frame_rate,
            "channels": audio.channels,
            "format": "Unknown",  # pydub doesn't expose original format
            "frame_count": audio.frame_count(),
        }
        
        # Calculate timing
        timing["total"] = time.time() - start_time
        
        # Log performance metrics
        perf_logger.log_operation(
            operation="get_audio_info",
            duration_ms=timing["total"] * 1000,
            success=True,
            info_id=info_id,
            stages_count=len(timing),
            input_type=type(audio_input).__name__,
            audio_duration_seconds=result["duration_seconds"],
            sample_rate=result["sample_rate"],
            channels=result["channels"],
        )
        
        logger.info("Audio info extracted", extra={
            "info_id": info_id,
            "info_extraction_time_ms": timing["total"] * 1000,
            "duration_seconds": result["duration_seconds"],
            "sample_rate": result["sample_rate"],
            "channels": result["channels"],
            "format": result["format"],
            "frame_count": result["frame_count"],
        })
        
        return result
        
    except Exception as e:
        error_timing = time.time() - start_time
        
        # Log detailed error information
        log_error(
            message=f"Failed to get audio info after {error_timing:.2f}s",
            error=e,
            error_type="AUDIO_INFO_EXTRACTION_FAILURE",
            error_code="PREPROCESS_004",
            remediation="Check input validity and audio format",
            logger=logger,
            info_id=info_id,
            stages_completed=list(timing.keys()),
            error_timing_seconds=error_timing,
        )
        
        # Record failed operation
        perf_logger.log_operation(
            operation="get_audio_info",
            duration_ms=error_timing * 1000,
            success=False,
            error_type="AUDIO_INFO_EXTRACTION_FAILURE",
            info_id=info_id,
            stages_completed=list(timing.keys()),
        )
        
        raise OMOAudioError(f"Failed to get audio info: {e}") from e


def validate_audio_input(
    audio_input: Union[bytes, Path, str, BinaryIO],
    max_duration_seconds: Optional[float] = None,
    min_duration_seconds: float = 0.1,
) -> bool:
    """
    Validate audio input before processing.
    
    Args:
        audio_input: Audio data (bytes, file path, or file-like object)
        max_duration_seconds: Maximum allowed duration (None = no limit)
        min_duration_seconds: Minimum required duration (default: 0.1s)
        
    Returns:
        True if valid
        
    Raises:
        OMOAudioError: If audio is invalid or outside duration limits
        FileNotFoundError: If file path doesn't exist
        ValueError: If audio data is invalid or malformed
    """
    # Initialize logging
    logger = get_logger("omoai.pipeline.preprocess")
    perf_logger = get_performance_logger()
    
    # Generate unique validation ID for tracing
    validation_id = str(uuid.uuid4())
    
    logger.info("Starting audio validation", extra={
        "validation_id": validation_id,
        "input_type": type(audio_input).__name__,
        "max_duration_seconds": max_duration_seconds,
        "min_duration_seconds": min_duration_seconds,
    })
    
    timing = {}
    start_time = time.time()
    
    try:
        # Check for empty input first
        if isinstance(audio_input, bytes) and len(audio_input) == 0:
            raise ValueError("Audio data is empty")
        
        # Get audio info for validation
        with performance_context("audio_info_extraction", logger=logger):
            info_start = time.time()
            info = get_audio_info(audio_input)
            timing["info_extraction"] = time.time() - info_start
            
            logger.debug("Audio info extracted for validation", extra={
                "validation_id": validation_id,
                "info_extraction_time_ms": timing["info_extraction"] * 1000,
                "duration_seconds": info.get("duration_seconds", "unknown"),
                "sample_rate": info.get("sample_rate", "unknown"),
                "channels": info.get("channels", "unknown"),
                "format": info.get("format", "unknown"),
            })
        
        duration = info["duration_seconds"]
        
        # Validate duration limits
        if duration < min_duration_seconds:
            raise ValueError(
                f"Audio too short: {duration:.2f}s < {min_duration_seconds:.2f}s minimum"
            )
        
        if max_duration_seconds and duration > max_duration_seconds:
            raise ValueError(
                f"Audio too long: {duration:.2f}s > {max_duration_seconds:.2f}s maximum"
            )
        
        # Validate sample rate and channels
        if info["sample_rate"] <= 0:
            raise OMOAudioError("Invalid sample rate")
        
        if info["channels"] <= 0:
            raise OMOAudioError("Invalid channel count")
        
        # Calculate total timing
        timing["total"] = time.time() - start_time
        
        # Log performance metrics
        perf_logger.log_operation(
            operation="audio_validation",
            duration_ms=timing["total"] * 1000,
            success=True,
            validation_id=validation_id,
            stages_count=len(timing),
            audio_duration_seconds=duration,
            sample_rate=info["sample_rate"],
            channels=info["channels"],
            real_time_factor=timing["total"] / duration if duration > 0 else 0,
        )
        
        logger.info("Audio validation completed successfully", extra={
            "validation_id": validation_id,
            "validation_time_ms": timing["total"] * 1000,
            "duration_seconds": duration,
            "sample_rate": info["sample_rate"],
            "channels": info["channels"],
            "format": info.get("format", "Unknown"),
            "within_limits": min_duration_seconds <= duration <= (max_duration_seconds or float('inf')),
        })
        
        return True
        
    except Exception as e:
        error_timing = time.time() - start_time
        
        # Enhanced error reporting with timing info
        log_error(
            message=f"Audio validation failed after {error_timing:.2f}s",
            error=e,
            error_type="AUDIO_VALIDATION_FAILURE",
            error_code="PREPROCESS_005",
            remediation="Check input validity, format, and duration limits",
            logger=logger,
            validation_id=validation_id,
            stages_completed=list(timing.keys()),
            error_timing_seconds=error_timing,
        )
        
        # Record failed operation
        perf_logger.log_operation(
            operation="audio_validation",
            duration_ms=error_timing * 1000,
            success=False,
            error_type="AUDIO_VALIDATION_FAILURE",
            validation_id=validation_id,
            stages_completed=list(timing.keys()),
        )
        
        if isinstance(e, ValueError):
            raise
        elif isinstance(e, OMOAudioError):
            raise ValueError(str(e)) from e
        raise OMOAudioError(f"Audio validation failed: {e}") from e



