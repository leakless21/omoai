"""Wrapper module for the postprocessing script."""
import subprocess
import sys
from pathlib import Path
from typing import Union, Optional

from omoai.api.exceptions import AudioProcessingException


def run_postprocess_script(
    asr_json_path: Union[str, Path],
    output_path: Union[str, Path],
    config_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Run the postprocessing script to add punctuation and generate summary.
    
    Args:
        asr_json_path: Path to the ASR output JSON file
        output_path: Path where the final output will be saved
        config_path: Path to the config.yaml file
        
    Raises:
        AudioProcessingException: If postprocessing fails
    """
    cmd = [
        sys.executable,
        "-m",
        "scripts.post",
        "--asr-json", str(asr_json_path),
        "--out", str(output_path),
    ]
    
    if config_path:
        cmd.extend(["--config", str(config_path)])
    
    try:
        # Enhanced logging for debugging
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Postprocess command: {' '.join(cmd)}")
        
        # Change to project root directory to ensure scripts module is accessible
        project_root = Path(__file__).resolve().parents[4]  # Go up 5 levels from src/omoai/api/scripts/postprocess_wrapper.py to project root
        logger.info(f"Changing working directory to: {project_root}")
        
        # Allow stdout/stderr to be visible for progress monitoring
        subprocess.run(cmd, check=True, cwd=project_root)
    except subprocess.CalledProcessError as e:
        # Enhanced logging for debugging
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Postprocessing failed with return code {e.returncode}")
        logger.error(f"Postprocess command that failed: {' '.join(cmd)}")
        raise AudioProcessingException(f"Postprocessing failed with return code {e.returncode}")
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Unexpected error during postprocessing: {str(e)}")
        logger.error(f"Postprocess command: {' '.join(cmd)}")
        raise AudioProcessingException(f"Unexpected error during postprocessing: {str(e)}")