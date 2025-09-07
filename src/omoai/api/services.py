"""Services module containing core business logic."""
import os
import json
import tempfile
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from omoai.config.schemas import get_config
from omoai.api.exceptions import AudioProcessingException
from omoai.api.models import (
    PipelineRequest,
    PipelineResponse,
    PreprocessRequest,
    PreprocessResponse,
    ASRRequest,
    ASRResponse,
    PostprocessRequest,
    PostprocessResponse,
    OutputFormatParams
)
from omoai.api.scripts.preprocess_wrapper import run_preprocess_script
from omoai.api.scripts.asr_wrapper import run_asr_script
from omoai.api.scripts.postprocess_wrapper import run_postprocess_script

# Compatibility alias: some tests import modules under the "src.omoai" package path.
# Ensure this module object is also available under that name so patches target the same object.
import sys as _sys
_module = _sys.modules.get(__name__)
if _module is not None:
    _sys.modules.setdefault("src.omoai.api.services", _module)

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
        config_path = None

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
        config_path = None

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


async def run_full_pipeline(data: PipelineRequest, output_params: Optional[OutputFormatParams] = None) -> PipelineResponse:
    """
    Run the full pipeline: preprocess -> ASR -> post-process.

    Args:
        data: PipelineRequest containing the uploaded audio file
        state: Application state holding initialized models

    Returns:
        PipelineResponse with final transcript, summary, and segments
    """
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info("Starting full pipeline execution")
    
    # 1) Save upload to a temporary file
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        upload_path = temp_path / "input_audio"
        content = await data.audio_file.read()
        with open(upload_path, "wb") as f:
            f.write(content)
        
        logger.info(f"Saved uploaded audio to temporary file: {upload_path}")
        logger.info(f"Audio file size: {len(content)} bytes")

        # 2) Preprocess via script
        config = get_config()
        preprocessed_path = Path(config.api.temp_dir) / f"preprocessed_{os.urandom(8).hex()}.wav"
        logger.info(f"Starting audio preprocessing to: {preprocessed_path}")
        try:
            run_preprocess_script(input_path=upload_path, output_path=preprocessed_path)
            logger.info("Audio preprocessing completed successfully")
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {str(e)}")
            raise

        # 3) ASR via script
        asr_json_path = Path(config.api.temp_dir) / f"asr_{os.urandom(8).hex()}.json"
        config_path = None
        logger.info(f"Starting ASR processing, output will be saved to: {asr_json_path}")
        logger.info(f"Using config path: {config_path}")
        try:
            run_asr_script(audio_path=preprocessed_path, output_path=asr_json_path, config_path=config_path)
            logger.info("ASR processing completed successfully")
        except Exception as e:
            logger.error(f"ASR processing failed: {str(e)}")
            raise

        # 4) Post-process via script
        final_json_path = Path(config.api.temp_dir) / f"final_{os.urandom(8).hex()}.json"
        logger.info(f"Starting post-processing, output will be saved to: {final_json_path}")
        try:
            run_postprocess_script(asr_json_path=asr_json_path, output_path=final_json_path, config_path=config_path)
            logger.info("Post-processing completed successfully")
        except Exception as e:
            logger.error(f"Post-processing failed: {str(e)}")
            raise

        # 5) Load final output
        logger.info(f"Loading final output from: {final_json_path}")
        try:
            with open(final_json_path, "r", encoding="utf-8") as f:
                final_obj: Dict[str, Any] = json.load(f)
            logger.info("Final output loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load final output: {str(e)}")
            raise

        # Apply output parameter filtering if provided
        if output_params:
            filtered_summary = final_obj.get("summary", {})
            filtered_segments = final_obj.get("segments", [])
            filtered_transcript_punct = final_obj.get("transcript_punct", "")

            # Filter summary based on parameters
            if output_params.summary:
                if output_params.summary == "none":
                    filtered_summary = {}
                elif output_params.summary == "bullets":
                    filtered_summary = {"bullets": filtered_summary.get("bullets", [])}
                elif output_params.summary == "abstract":
                    filtered_summary = {"abstract": filtered_summary.get("abstract", "")}
                # "both" keeps everything as-is

                # Apply bullet limit if specified
                if output_params.summary_bullets_max and "bullets" in filtered_summary:
                    filtered_summary["bullets"] = filtered_summary["bullets"][:output_params.summary_bullets_max]

            # Filter segments based on include parameters
            if output_params.include:
                include_set = set(output_params.include)
                if "segments" not in include_set:
                    filtered_segments = []
                if "transcript_punct" not in include_set:
                    filtered_transcript_punct = ""

            return PipelineResponse(
                summary=filtered_summary,
                segments=filtered_segments,
                transcript_punct=filtered_transcript_punct,
            )

        # Default behavior (backward compatibility)
        return PipelineResponse(
            summary=dict(final_obj.get("summary", {}) or {}),
            segments=list(final_obj.get("segments", []) or []),
            transcript_punct=str(final_obj.get("transcript_punct", "")) or None,
        )