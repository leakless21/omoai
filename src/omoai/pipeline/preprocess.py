"""
In-memory audio preprocessing for OMOAI.

This module provides efficient audio preprocessing that works with in-memory
data streams and tensors, eliminating the need for intermediate file I/O.
"""
import io
import subprocess
import tempfile
from pathlib import Path
from typing import BinaryIO, Union, Tuple, Optional

import torch
import numpy as np
from pydub import AudioSegment  # type: ignore


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
        subprocess.CalledProcessError: If ffmpeg processing fails
        ValueError: If audio data is invalid
    """
    if not audio_data:
        raise ValueError("Audio data is empty")
    
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
    
    try:
        result = subprocess.run(
            cmd,
            input=audio_data,
            capture_output=True,
            check=True
        )
        
        if not result.stdout:
            raise ValueError("ffmpeg produced no output")
            
        return result.stdout
        
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode('utf-8', errors='ignore') if e.stderr else "Unknown error"
        raise subprocess.CalledProcessError(
            e.returncode,
            cmd,
            f"ffmpeg preprocessing failed: {error_msg}"
        )


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
        ValueError: If audio cannot be loaded or processed
        FileNotFoundError: If file path doesn't exist
    """
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
            raise ValueError(f"Unsupported audio input type: {type(audio_input)}")
        
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
        
        if return_sample_rate:
            return tensor, target_sample_rate
        return tensor
        
    except Exception as e:
        raise ValueError(f"Failed to preprocess audio: {e}")


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
        FileNotFoundError: If input file doesn't exist
        subprocess.CalledProcessError: If ffmpeg processing fails
    """
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
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            check=True
        )
        
        if not result.stdout:
            raise ValueError("ffmpeg produced no output")
            
        return result.stdout
        
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode('utf-8', errors='ignore') if e.stderr else "Unknown error"
        raise subprocess.CalledProcessError(
            e.returncode,
            cmd,
            f"ffmpeg preprocessing failed for {input_path}: {error_msg}"
        )


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
    """
    try:
        if isinstance(audio_input, (str, Path)):
            audio = AudioSegment.from_file(str(audio_input))
        elif isinstance(audio_input, bytes):
            audio_io = io.BytesIO(audio_input)
            audio = AudioSegment.from_file(audio_io)
        else:
            raise ValueError(f"Unsupported input type: {type(audio_input)}")
        
        return {
            "duration_seconds": len(audio) / 1000.0,
            "sample_rate": audio.frame_rate,
            "channels": audio.channels,
            "format": "Unknown",  # pydub doesn't expose original format
            "frame_count": audio.frame_count(),
        }
        
    except Exception as e:
        raise ValueError(f"Failed to get audio info: {e}")


def validate_audio_input(
    audio_input: Union[bytes, Path, str],
    max_duration_seconds: Optional[float] = None,
    min_duration_seconds: float = 0.1,
) -> bool:
    """
    Validate audio input before processing.
    
    Args:
        audio_input: Audio data to validate
        max_duration_seconds: Maximum allowed duration (None = no limit)
        min_duration_seconds: Minimum required duration
        
    Returns:
        True if valid, raises ValueError if invalid
        
    Raises:
        ValueError: If audio is invalid or outside duration limits
    """
    try:
        info = get_audio_info(audio_input)
        
        duration = info["duration_seconds"]
        
        if duration < min_duration_seconds:
            raise ValueError(
                f"Audio too short: {duration:.2f}s < {min_duration_seconds:.2f}s minimum"
            )
        
        if max_duration_seconds and duration > max_duration_seconds:
            raise ValueError(
                f"Audio too long: {duration:.2f}s > {max_duration_seconds:.2f}s maximum"
            )
        
        if info["sample_rate"] <= 0:
            raise ValueError("Invalid sample rate")
        
        if info["channels"] <= 0:
            raise ValueError("Invalid channel count")
        
        return True
        
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise ValueError(f"Audio validation failed: {e}")


# Legacy compatibility function
def preprocess_to_wav(input_path: Path, output_path: Path) -> None:
    """
    Legacy compatibility function for file-to-file processing.
    
    This maintains compatibility with existing scripts while using
    the optimized in-memory processing internally.
    """
    wav_bytes = preprocess_file_to_wav_bytes(input_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(wav_bytes)
