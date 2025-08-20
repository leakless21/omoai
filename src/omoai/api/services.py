"""Services module containing core business logic."""
import os
import json
import tempfile
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from src.omoai.api.config import get_config
from src.omoai.api.exceptions import AudioProcessingException
from src.omoai.api.models import (
    PipelineRequest,
    PipelineResponse,
    PreprocessRequest,
    PreprocessResponse,
    ASRRequest,
    ASRResponse,
    PostprocessRequest,
    PostprocessResponse
)
from src.omoai.api.scripts.preprocess_wrapper import run_preprocess_script
from src.omoai.api.scripts.asr_wrapper import run_asr_script
from src.omoai.api.scripts.postprocess_wrapper import run_postprocess_script


async def preprocess_audio_service(data: PreprocessRequest) -> PreprocessResponse:
    """
    Preprocess an audio file by converting it to 16kHz mono PCM16 WAV format.
    
    Args:
        data: PreprocessRequest containing the uploaded audio file
        
    Returns:
        PreprocessResponse containing the path to the preprocessed file
        
    Raises:
        AudioProcessingException: If preprocessing fails
    """
    try:
        # Create temporary directories for input and output files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Save uploaded file to temporary location
            input_path = temp_path / "input_audio"
            content = await data.audio_file.read()
            with open(input_path, "wb") as f:
                f.write(content)

            # Define final output path for processed file
            config = get_config()
            final_output_path = Path(config.api.temp_dir) / f"preprocessed_{os.urandom(8).hex()}.wav"

            # Use existing preprocess script via wrapper
            run_preprocess_script(input_path=input_path, output_path=final_output_path)

            return PreprocessResponse(output_path=str(final_output_path))
            
    except subprocess.CalledProcessError as e:
        raise AudioProcessingException(f"Audio preprocessing failed: {e.stderr}")
    except Exception as e:
        raise AudioProcessingException(f"Unexpected error during preprocessing: {str(e)}")


def asr_service(data: ASRRequest) -> ASRResponse:
    """
    Run ASR using the existing scripts.asr module via the wrapper and return structured output.
    """
    audio_path = Path(data.preprocessed_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Preprocessed audio file not found: {data.preprocessed_path}")

    try:
        # Run ASR to a temporary JSON file
        config = get_config()
        asr_json_path = Path(config.api.temp_dir) / f"asr_{os.urandom(8).hex()}.json"
        config_path = config.config_path

        run_asr_script(
            audio_path=audio_path,
            output_path=asr_json_path,
            config_path=config_path,
        )

        with open(asr_json_path, "r", encoding="utf-8") as f:
            asr_obj: Dict[str, Any] = json.load(f)

        return ASRResponse(
            segments=list(asr_obj.get("segments", []) or []),
        )
    except subprocess.CalledProcessError as e:
        raise AudioProcessingException(f"ASR processing failed: {e.stderr}")
    except Exception as e:
        raise AudioProcessingException(f"Unexpected error during ASR: {str(e)}")


def postprocess_service(data: PostprocessRequest) -> PostprocessResponse:
    """
    Run punctuation and summarization via scripts.post wrapper on provided ASR output dict.
    """
    try:
        # Write ASR output to a temp file for the script
        config = get_config()
        tmp_asr_json = Path(config.api.temp_dir) / f"asr_{os.urandom(8).hex()}.json"
        with open(tmp_asr_json, "w", encoding="utf-8") as f:
            json.dump(data.asr_output, f, ensure_ascii=False)

        final_json_path = Path(config.api.temp_dir) / f"final_{os.urandom(8).hex()}.json"
        config_path = config.config_path

        run_postprocess_script(
            asr_json_path=tmp_asr_json,
            output_path=final_json_path,
            config_path=config_path,
        )

        with open(final_json_path, "r", encoding="utf-8") as f:
            final_obj: Dict[str, Any] = json.load(f)

        return PostprocessResponse(
            summary=dict(final_obj.get("summary", {}) or {}),
            segments=list(final_obj.get("segments", []) or []),
        )
    except subprocess.CalledProcessError as e:
        raise AudioProcessingException(f"Post-processing failed: {e.stderr}")
    except Exception as e:
        raise AudioProcessingException(f"Unexpected error during post-processing: {str(e)}")


async def run_full_pipeline(data: PipelineRequest) -> PipelineResponse:
    """
    Run the full pipeline: preprocess -> ASR -> post-process.

    Args:
        data: PipelineRequest containing the uploaded audio file
        state: Application state holding initialized models

    Returns:
        PipelineResponse with final transcript, summary, and segments
    """
    # 1) Save upload to a temporary file
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        upload_path = temp_path / "input_audio"
        content = await data.audio_file.read()
        with open(upload_path, "wb") as f:
            f.write(content)

        # 2) Preprocess via script
        config = get_config()
        preprocessed_path = Path(config.api.temp_dir) / f"preprocessed_{os.urandom(8).hex()}.wav"
        run_preprocess_script(input_path=upload_path, output_path=preprocessed_path)

        # 3) ASR via script
        asr_json_path = Path(config.api.temp_dir) / f"asr_{os.urandom(8).hex()}.json"
        config_path = config.config_path
        run_asr_script(audio_path=preprocessed_path, output_path=asr_json_path, config_path=config_path)

        # 4) Post-process via script
        final_json_path = Path(config.api.temp_dir) / f"final_{os.urandom(8).hex()}.json"
        run_postprocess_script(asr_json_path=asr_json_path, output_path=final_json_path, config_path=config_path)

        # 5) Load final output
        with open(final_json_path, "r", encoding="utf-8") as f:
            final_obj: Dict[str, Any] = json.load(f)

        return PipelineResponse(
            summary=dict(final_obj.get("summary", {}) or {}),
            segments=list(final_obj.get("segments", []) or []),
        )