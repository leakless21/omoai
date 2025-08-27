"""Wrapper module for the preprocess script."""
import subprocess
from pathlib import Path
from typing import Union

from omoai.api.exceptions import AudioProcessingException


def run_preprocess_script(input_path: Union[str, Path], output_path: Union[str, Path]) -> None:
    """
    Run the preprocess script to convert audio to 16kHz mono PCM16 WAV format.
    
    Args:
        input_path: Path to the input audio file
        output_path: Path where the preprocessed audio will be saved
        
    Raises:
        AudioProcessingException: If preprocessing fails
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-ac", "1",
        "-ar", "16000",
        "-vn",
        "-c:a", "pcm_s16le",
        str(output_path),
    ]
    
    try:
        # Allow stdout/stderr to be visible for progress monitoring  
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise AudioProcessingException(f"Audio preprocessing failed with return code {e.returncode}")
    except Exception as e:
        raise AudioProcessingException(f"Unexpected error during preprocessing: {str(e)}")