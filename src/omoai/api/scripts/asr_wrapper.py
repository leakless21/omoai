"""Wrapper module for the ASR script."""
import subprocess
import sys
from pathlib import Path
from typing import Union, Optional

from src.omoai.api.exceptions import AudioProcessingException


def run_asr_script(
    audio_path: Union[str, Path],
    output_path: Union[str, Path],
    model_dir: Optional[Union[str, Path]] = None,
    config_path: Optional[Union[str, Path]] = None,
    total_batch_duration: Optional[int] = None,
    chunk_size: Optional[int] = None,
    left_context_size: Optional[int] = None,
    right_context_size: Optional[int] = None,
    device: Optional[str] = None,
    autocast_dtype: Optional[str] = None,
) -> None:
    """
    Run the ASR script to transcribe audio.
    
    Args:
        audio_path: Path to the preprocessed audio file
        output_path: Path where the ASR output will be saved
        model_dir: Path to the model checkpoint directory
        config_path: Path to the config.yaml file
        total_batch_duration: Total audio duration per batch in seconds
        chunk_size: Chunk size
        left_context_size: Left context size
        right_context_size: Right context size
        device: Device to run the model on (cuda/cpu)
        autocast_dtype: Autocast dtype
        
    Raises:
        AudioProcessingException: If ASR processing fails
    """
    cmd = [
        sys.executable,
        "-m",
        "scripts.asr",
        "--audio", str(audio_path),
        "--out", str(output_path),
    ]
    
    if config_path:
        cmd.extend(["--config", str(config_path)])
        
    if model_dir:
        cmd.extend(["--model-dir", str(model_dir)])
        
    if total_batch_duration:
        cmd.extend(["--total-batch-duration", str(total_batch_duration)])
        
    if chunk_size:
        cmd.extend(["--chunk-size", str(chunk_size)])
        
    if left_context_size:
        cmd.extend(["--left-context-size", str(left_context_size)])
        
    if right_context_size:
        cmd.extend(["--right-context-size", str(right_context_size)])
        
    if device:
        cmd.extend(["--device", device])
        
    if autocast_dtype:
        cmd.extend(["--autocast-dtype", autocast_dtype])
    
    try:
        # Allow stdout/stderr to be visible for progress monitoring
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        # Log stdout for debugging
        if result.stdout:
            print(f"ASR stdout: {result.stdout}")
    except subprocess.CalledProcessError as e:
        # Capture both stderr and stdout for better error reporting
        error_message = f"ASR processing failed with return code {e.returncode}"
        if e.stderr:
            error_message += f", stderr: {e.stderr}"
        if e.stdout:
            error_message += f", stdout: {e.stdout}"
        raise AudioProcessingException(error_message)
    except Exception as e:
        raise AudioProcessingException(f"Unexpected error during ASR processing: {str(e)}")