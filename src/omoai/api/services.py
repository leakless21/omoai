"""Services module containing core business logic.

This file consolidates the script-based implementations and the async wrappers
previously present in services_enhanced.py. The service-mode selection enums and
health/probing helpers for switching to in-memory services were intentionally
removed as requested.
"""
import os
import json
import tempfile
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import asyncio
import time
import sys

from litestar.datastructures import UploadFile

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
    OutputFormatParams,
)

# NOTE: This module remains script-based by default. The async wrappers call into
# the script functions (offloading blocking work to threads) so controllers can
# await them uniformly.


def _normalize_summary(raw_summary: Any) -> dict:
    """
    Normalize summary output into a structured dict:
    {
      "title": str,
      "summary": str,
      "points": List[str]
    }
    """
    import re
    import unicodedata

    def _strip_label_re(s: str):
        if not isinstance(s, str):
            return s or ""
        s = s.strip()
        return re.sub(r"^(?:\s*(?:Tiêu đề|Tóm tắt|Điểm chính|Title|Summary)\s*:?\s*)+", "", s, flags=re.IGNORECASE).strip()

    def _parse_text_summary(text: str) -> dict:
        text = (text or "").strip()
        title = ""
        abstract = ""
        points: List[str] = []

        # Normalize text for consistent character matching
        normalized_text = unicodedata.normalize('NFC', text)
        
        # Extract title using robust pattern matching
        title_match = re.search(r"(?i)tiêu đề:\s*(.+?)(?=\n|$)", normalized_text)
        if title_match:
            title = title_match.group(1).strip()
        else:
            title_match = re.search(r"(?i)title:\s*(.+?)(?=\n|$)", normalized_text)
            if title_match:
                title = title_match.group(1).strip()

        # Extract abstract using robust pattern matching
        abstract_match = re.search(
            r"(?i)tóm tắt:\s*(.+?)(?=\n\s*điểm chính|\n\s*điểm chính:|\n-|$)",
            normalized_text,
            flags=re.MULTILINE | re.DOTALL
        )
        if abstract_match:
            abstract = abstract_match.group(1).strip()
        else:
            abstract_match = re.search(
                r"(?i)summary:\s*(.+?)(?=\n\s*điểm chính|\n\s*điểm chính:|\n-|$)",
                normalized_text,
                flags=re.MULTILINE | re.DOTALL
            )
            if abstract_match:
                abstract = abstract_match.group(1).strip()
            else:
                # Fallback: use first paragraph
                parts = text.split("\n\n")
                if parts:
                    abstract = parts[0].strip()

        # Extract points using robust pattern matching
        points_match = re.search(
            r"(?i)điểm chính:\s*(.*)",
            normalized_text,
            flags=re.MULTILINE | re.DOTALL
        )
        if points_match:
            block = points_match.group(1)
            for line in block.splitlines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("-"):
                    points.append(line.lstrip("-").strip())
                else:
                    points.append(line)
        else:
            # Fallback: extract bullet points from entire text
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("-"):
                    points.append(line.lstrip("-").strip())

        # Clean up extracted content
        title = _strip_label_re(title)
        abstract = _strip_label_re(abstract)

        # If nothing was extracted, use the entire text as abstract
        if not (title or abstract or points):
            abstract = text

        return {"title": title, "summary": abstract, "abstract": abstract, "points": points}

    if isinstance(raw_summary, dict):
        title = raw_summary.get("title") or raw_summary.get("Tiêu đề") or raw_summary.get("tiêu đề") or raw_summary.get("Title") or ""
        abstract = raw_summary.get("summary") or raw_summary.get("abstract") or raw_summary.get("Tóm tắt") or raw_summary.get("tóm tắt") or ""
        points = raw_summary.get("points") or raw_summary.get("bullets") or raw_summary.get("Bullets") or []

        combined_text = ""
        if title:
            combined_text += str(title).strip() + "\n\n"
        if abstract:
            combined_text += str(abstract).strip()

        if isinstance(points, str):
            pts: List[str] = []
            for line in points.splitlines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("-"):
                    pts.append(line.lstrip("-").strip())
                else:
                    pts.append(line)
            points = pts

        # Check if the abstract contains Vietnamese labels that need parsing
        if abstract and (re.search(r"(?i)tiêu đề:|tóm tắt:|điểm chính", abstract, flags=re.MULTILINE)):
            # Parse the abstract text to extract structured content
            parsed = _parse_text_summary(abstract)
            return {
                "title": parsed.get("title", "") or _strip_label_re(title),
                "summary": parsed.get("summary", "") or _strip_label_re(abstract),
                "abstract": parsed.get("summary", "") or _strip_label_re(abstract),
                "points": parsed.get("points", []) or list(points or [])
            }
        
        # Also check combined text for labels (original logic)
        if combined_text and (re.search(r"(?i)t[oó]m\s*t[aá]t|đi[eê]m\s*ch[ií]nh|^-", combined_text, flags=re.MULTILINE)):
            parsed = _parse_text_summary(combined_text)
            return {
                "title": parsed.get("title", "") or _strip_label_re(title),
                "summary": parsed.get("summary", "") or _strip_label_re(abstract),
                "abstract": parsed.get("summary", "") or _strip_label_re(abstract),
                "points": parsed.get("points", []) or list(points or [])
            }

        title = _strip_label_re(title)
        abstract = _strip_label_re(abstract)
        clean_points: List[str] = []
        for p in list(points or []):
            if not isinstance(p, str):
                continue
            clean_points.append(_strip_label_re(p) if _strip_label_re(p) else p.strip())
        return {"title": title, "summary": abstract, "abstract": abstract, "points": clean_points}

    if isinstance(raw_summary, str):
        return _parse_text_summary(raw_summary)

    return {"title": "", "summary": "", "abstract": "", "points": []}


# Script-based helper implementations

def run_preprocess_script(input_path, output_path):
    """Fallback preprocess implementation using ffmpeg directly."""
    import subprocess
    import logging
    logger = logging.getLogger(__name__)

    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-ac", "1", "-ar", "16000", "-vn", "-c:a", "pcm_s16le",
        str(output_path)
    ]

    logger.info(
        "Running preprocess command",
        extra={"cmd": " ".join(cmd), "input_path": str(input_path), "output_path": str(output_path)},
    )
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(
            "Preprocess completed successfully",
            extra={"return_code": result.returncode, "stdout": (result.stdout or "").strip()},
        )
    except subprocess.CalledProcessError as e:
        logger.error(
            "Preprocess failed",
            exc_info=e,
            extra={
                "return_code": e.returncode,
                "stderr": (e.stderr or "").strip(),
                "stdout": (e.stdout or "").strip(),
            },
        )
        raise AudioProcessingException(f"Audio preprocessing failed: {e.stderr}")
    except Exception as e:
        logger.error("Preprocess failed with unexpected error", exc_info=e, extra={"error": str(e)})
        raise AudioProcessingException(f"Audio preprocessing failed: {str(e)}")


def run_asr_script(audio_path, output_path, config_path=None):
    """Fallback ASR implementation using subprocess call to legacy script."""
    import subprocess
    import sys
    from pathlib import Path
    import logging

    logger = logging.getLogger(__name__)

    project_root = Path(__file__).resolve().parents[3]

    cmd = [sys.executable, "-m", "scripts.asr", "--audio", str(audio_path), "--out", str(output_path)]
    if config_path:
        cmd.extend(["--config", str(config_path)])

    logger.info("ASR command", extra={"cmd": " ".join(cmd), "cwd": str(project_root)})

    try:
        result = subprocess.run(cmd, check=True, cwd=project_root, capture_output=True, text=True)
        logger.info(
            "ASR completed successfully",
            extra={"return_code": result.returncode, "stdout": (result.stdout or "").strip()},
        )
    except subprocess.CalledProcessError as e:
        logger.error(
            "ASR failed",
            exc_info=e,
            extra={
                "return_code": e.returncode,
                "stderr": (e.stderr or "").strip(),
                "stdout": (e.stdout or "").strip(),
            },
        )
        raise AudioProcessingException(f"ASR processing failed: {e.stderr}")
    except Exception as e:
        logger.error("ASR failed with unexpected error", exc_info=e, extra={"error": str(e)})
        raise AudioProcessingException(f"ASR processing failed: {str(e)}")


def run_postprocess_script(asr_json_path, output_path, config_path=None):
    """Fallback postprocess implementation using subprocess call to legacy script."""
    import subprocess
    import sys
    from pathlib import Path
    import logging

    logger = logging.getLogger(__name__)

    project_root = Path(__file__).resolve().parents[3]

    cmd = [sys.executable, "-m", "scripts.post", "--asr-json", str(asr_json_path), "--out", str(output_path)]
    if config_path:
        cmd.extend(["--config", str(config_path)])

    logger.info("Postprocess command", extra={"cmd": " ".join(cmd), "cwd": str(project_root)})

    try:
        result = subprocess.run(cmd, check=True, cwd=project_root, capture_output=True, text=True)
        logger.info(
            "Postprocess completed successfully",
            extra={"return_code": result.returncode, "stdout": (result.stdout or "").strip()},
        )
    except subprocess.CalledProcessError as e:
        logger.error(
            "Postprocess failed",
            exc_info=e,
            extra={
                "return_code": e.returncode,
                "stderr": (e.stderr or "").strip(),
                "stdout": (e.stdout or "").strip(),
            },
        )
        raise AudioProcessingException(f"Post-processing failed: {e.stderr}")
    except Exception as e:
        logger.error("Postprocess failed with unexpected error", exc_info=e, extra={"error": str(e)})
        raise AudioProcessingException(f"Post-processing failed: {str(e)}")


# Compatibility alias: some tests import modules under the "src.omoai" package path.
import sys as _sys
_module = _sys.modules.get(__name__)
if _module is not None:
    _sys.modules.setdefault("src.omoai.api.services", _module)
    # Backwards compatibility: some tests and older code import services_enhanced
    _sys.modules.setdefault("src.omoai.api.services_enhanced", _module)


# Script-based service implementations (internal names)

async def preprocess_audio_service(data: PreprocessRequest) -> PreprocessResponse:
    """
    Preprocess an audio file by converting it to 16kHz mono PCM16 WAV format.
    """
    try:
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


def _asr_script(data: ASRRequest) -> ASRResponse:
    """
    Run ASR using the existing scripts.asr module via the wrapper and return structured output.
    """
    audio_path = Path(data.preprocessed_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Preprocessed audio file not found: {data.preprocessed_path}")

    try:
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


def _postprocess_script(data: PostprocessRequest) -> PostprocessResponse:
    """
    Run punctuation and summarization via scripts.post wrapper on provided ASR output dict.
    """
    try:
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
            summary_raw_text=str(final_obj.get("summary_raw_text", "")) or None,
        )
    except subprocess.CalledProcessError as e:
        raise AudioProcessingException(f"Post-processing failed: {e.stderr}")
    except Exception as e:
        raise AudioProcessingException(f"Unexpected error during post-processing: {str(e)}")


# Full pipeline script implementation
async def _run_full_pipeline_script(data: PipelineRequest, output_params: Optional[OutputFormatParams] = None) -> PipelineResponse:
    """
    Run the full pipeline: preprocess -> ASR -> post-process.
    """
    import logging
    logger = logging.getLogger(__name__)

    logger.info("Starting full pipeline execution")

    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        logger.info(f"Memory usage at start: {memory_info.rss / 1024 / 1024:.2f} MB")
    except ImportError:
        logger.warning("psutil not available for memory monitoring")
        process = None
    except Exception as e:
        logger.warning(f"Memory monitoring failed: {str(e)}")
        process = None

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        upload_path = temp_path / "input_audio"
        content = await data.audio_file.read()
        with open(upload_path, "wb") as f:
            f.write(content)

        logger.info(f"Saved uploaded audio to temporary file: {upload_path}")
        logger.info(f"Audio file size: {len(content)} bytes")

        config = get_config()
        preprocessed_path = Path(config.api.temp_dir) / f"preprocessed_{os.urandom(8).hex()}.wav"
        logger.info(f"Starting audio preprocessing to: {preprocessed_path}")
        try:
            logger.info("Starting audio preprocessing")
            run_preprocess_script(input_path=upload_path, output_path=preprocessed_path)

            try:
                if process is not None:
                    memory_info = process.memory_info()
                    logger.info(f"Memory usage after preprocessing: {memory_info.rss / 1024 / 1024:.2f} MB")
            except Exception:
                pass

            logger.info("Audio preprocessing completed successfully")
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {str(e)}")
            raise

        asr_json_path = Path(config.api.temp_dir) / f"asr_{os.urandom(8).hex()}.json"
        config_path = None
        logger.info(f"Starting ASR processing, output will be saved to: {asr_json_path}")
        logger.info(f"Using config path: {config_path}")
        try:
            logger.info("Starting ASR processing")
            run_asr_script(audio_path=preprocessed_path, output_path=asr_json_path, config_path=config_path)

            try:
                if process is not None:
                    memory_info = process.memory_info()
                    logger.info(f"Memory usage after ASR: {memory_info.rss / 1024 / 1024:.2f} MB")
            except Exception:
                pass

            logger.info("ASR processing completed successfully")
        except Exception as e:
            logger.error(f"ASR processing failed: {str(e)}")
            raise

        final_json_path = Path(config.api.temp_dir) / f"final_{os.urandom(8).hex()}.json"
        logger.info(f"Starting post-processing, output will be saved to: {final_json_path}")
        try:
            logger.info("Starting post-processing")
            run_postprocess_script(asr_json_path=asr_json_path, output_path=final_json_path, config_path=config_path)

            try:
                if process is not None:
                    memory_info = process.memory_info()
                    logger.info(f"Memory usage after post-processing: {memory_info.rss / 1024 / 1024:.2f} MB")
            except Exception:
                pass

            logger.info("Post-processing completed successfully")
        except Exception as e:
            logger.error(f"Post-processing failed: {str(e)}")
            raise

        logger.info(f"Loading final output from: {final_json_path}")
        try:
            with open(final_json_path, "r", encoding="utf-8") as f:
                final_obj: Dict[str, Any] = json.load(f)
            logger.info("Final output loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load final output: {str(e)}")
            raise

        if output_params:
            filtered_summary = final_obj.get("summary", {})
            filtered_segments = final_obj.get("segments", [])
            filtered_transcript_punct = final_obj.get("transcript_punct", "")

            if output_params.summary:
                if output_params.summary == "none":
                    filtered_summary = {}
                elif output_params.summary == "bullets":
                    filtered_summary = {"bullets": filtered_summary.get("bullets", [])}
                elif output_params.summary == "abstract":
                    filtered_summary = {"abstract": filtered_summary.get("abstract", "")}

                if output_params.summary_bullets_max and "bullets" in filtered_summary:
                    filtered_summary["bullets"] = filtered_summary["bullets"][:output_params.summary_bullets_max]

            if output_params.include:
                include_set = set(output_params.include)
                if "segments" not in include_set:
                    filtered_segments = []
                if "transcript_punct" not in include_set:
                    filtered_transcript_punct = ""

            quality_metrics = None
            diffs = None

            if output_params.include_quality_metrics and "quality_metrics" in final_obj:
                quality_metrics_data = final_obj["quality_metrics"]
                from omoai.api.models import QualityMetrics
                quality_metrics = QualityMetrics(**quality_metrics_data)

            if output_params.include_diffs and "diffs" in final_obj:
                diffs_data = final_obj["diffs"]
                from omoai.api.models import HumanReadableDiff
                if isinstance(diffs_data, list) and diffs_data:
                    diffs = HumanReadableDiff(**diffs_data[0])
                elif isinstance(diffs_data, dict):
                    diffs = HumanReadableDiff(**diffs_data)

            return PipelineResponse(
                summary=filtered_summary,
                segments=filtered_segments,
                transcript_punct=filtered_transcript_punct,
                quality_metrics=quality_metrics,
                diffs=diffs,
                summary_raw_text=(str(final_obj.get("summary_raw_text", "")) or None)
                if getattr(output_params, "return_summary_raw", None)
                else None,
            )

        quality_metrics = None
        diffs = None

        if output_params and output_params.include_quality_metrics and "quality_metrics" in final_obj:
            quality_metrics_data = final_obj["quality_metrics"]
            from omoai.api.models import QualityMetrics
            quality_metrics = QualityMetrics(**quality_metrics_data)

        if output_params and output_params.include_diffs and "diffs" in final_obj:
            diffs_data = final_obj["diffs"]
            from omoai.api.models import HumanReadableDiff
            if isinstance(diffs_data, list) and diffs_data:
                diffs = HumanReadableDiff(**diffs_data[0])
            elif isinstance(diffs_data, dict):
                diffs = HumanReadableDiff(**diffs_data)

        summary_data = final_obj.get("summary", {}) or {}

        if isinstance(summary_data, dict):
            final_summary = summary_data
        elif isinstance(summary_data, (list, tuple)) and len(summary_data) > 0 and isinstance(summary_data[0], dict):
            final_summary = summary_data[0]
        else:
            raise AudioProcessingException(
                "Post-processing produced unexpected summary format; expected dict with keys 'title','summary','points'"
            )

        # Normalize final_summary using _normalize_summary
        final_summary = _normalize_summary(final_summary)

        return PipelineResponse(
            summary=final_summary,
            segments=list(final_obj.get("segments", []) or []),
            transcript_punct=str(final_obj.get("transcript_punct", "")) or None,
            quality_metrics=quality_metrics,
            diffs=diffs,
            summary_raw_text=(str(final_obj.get("summary_raw_text", "")) or None)
            if (output_params and getattr(output_params, "return_summary_raw", None))
            else None,
        )


# Async wrappers and helpers

def asr_service(data: ASRRequest):
    """
    Dual-mode ASR service helper.

    - If called from synchronous code (no running asyncio event loop), this function
      will execute the underlying _asr_script synchronously and return an ASRResponse.
      This preserves legacy tests and callers that invoke the service without awaiting.

    - If called from async code (an event loop is running), this function returns an
      awaitable (a ThreadPool-backed task) which callers can `await` to get the ASRResponse.
      Example usage in async controllers: `return await asr_service(data)`
    """
    try:
        # If there's a running event loop, return an awaitable that runs the blocking work
        # in a thread so callers can `await` it without blocking the loop.
        asyncio.get_running_loop()
        return asyncio.to_thread(_asr_script, data)
    except RuntimeError:
        # No running loop — synchronous context (e.g., tests). Execute synchronously.
        return _asr_script(data)


async def postprocess_service(data: PostprocessRequest, output_params: Optional[dict] = None) -> PostprocessResponse:
    return await asyncio.to_thread(_postprocess_script, data)


class _BytesUploadFile(UploadFile):
    def __init__(self, data: bytes, filename: str = "upload.bin", content_type: str = "application/octet-stream"):
        self._data = data
        self.filename = filename
        self.content_type = content_type

    async def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            return self._data
        return self._data[:size]


async def run_full_pipeline(data: PipelineRequest, output_params: Optional[Union[OutputFormatParams, dict]] = None) -> PipelineResponse:
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


async def warmup_services() -> Dict[str, Any]:
    start = time.perf_counter()
    try:
        from . import singletons
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


async def get_service_status() -> Dict[str, Any]:
    status: Dict[str, Any] = {
        "service_mode": "SCRIPT_BASED",
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
    "warmup_services",
    "preprocess_audio_service",
    "asr_service",
    "postprocess_service",
    "run_full_pipeline",
    "get_service_status",
]
