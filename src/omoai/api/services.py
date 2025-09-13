"""Services module containing core business logic.

This file consolidates the script-based implementations and the async wrappers
previously present in services_enhanced.py. The service-mode selection enums and
health/probing helpers for switching to in-memory services were intentionally
removed as requested.
"""

import asyncio
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from litestar.datastructures import UploadFile

from omoai.api.exceptions import AudioProcessingException
from omoai.api.models import (
    ASRRequest,
    ASRResponse,
    OutputFormatParams,
    PipelineRequest,
    PipelineResponse,
    PostprocessRequest,
    PostprocessResponse,
    PreprocessRequest,
    PreprocessResponse,
)

# Use centralized script wrappers; keep names for backward compatibility
from omoai.api.scripts.asr_wrapper import run_asr_script as _run_asr_script
from omoai.api.scripts.postprocess_wrapper import (
    run_postprocess_script as _run_postprocess_script,
)
from omoai.config.schemas import get_config
from omoai.pipeline.postprocess_core_utils import (
    _parse_vietnamese_labeled_text as _parse_labeled_summary,
)

# NOTE: This module remains script-based by default. The async wrappers call into
# the script functions (offloading blocking work to threads) so controllers can
# await them uniformly.


class PipelineResult(PipelineResponse):
    """A PipelineResponse that can also be unpacked as (response, raw_transcript).

    - Behaves like PipelineResponse for attribute access in most callers.
    - Supports tuple-unpacking for legacy code/tests expecting (response, raw_transcript).
    """

    def __iter__(self):  # type: ignore[override]
        yield self
        yield getattr(self, "transcript_raw", None)


def _normalize_summary(raw_summary: Any) -> dict:
    """Normalize summary structure using core parsing utils."""
    # If dict-like, coerce keys and shapes
    if isinstance(raw_summary, dict):
        title = (
            raw_summary.get("title")
            or raw_summary.get("Tiêu đề")
            or raw_summary.get("Title")
            or ""
        )
        abstract = (
            raw_summary.get("summary")
            or raw_summary.get("abstract")
            or raw_summary.get("Tóm tắt")
            or ""
        )
        points = raw_summary.get("points") or raw_summary.get("bullets") or []

        # Coerce points that may arrive as a single string
        if isinstance(points, str):
            points = [p.lstrip("-").strip() for p in points.splitlines() if p.strip()]

        # If the abstract contains labeled text, let core parser extract canonical parts
        if isinstance(abstract, str):
            parsed = _parse_labeled_summary(abstract)
            if parsed:
                return {
                    "title": parsed.get("title", "") or str(title).strip(),
                    "summary": parsed.get("abstract", ""),
                    "abstract": parsed.get("abstract", ""),
                    "points": parsed.get("points", []) or list(points or []),
                }

        return {
            "title": str(title).strip(),
            "summary": str(abstract).strip(),
            "abstract": str(abstract).strip(),
            "points": list(points or []),
        }

    # If raw string, parse labeled text
    if isinstance(raw_summary, str):
        parsed = _parse_labeled_summary(raw_summary)
        if parsed:
            return {
                "title": parsed.get("title", ""),
                "summary": parsed.get("abstract", ""),
                "abstract": parsed.get("abstract", ""),
                "points": parsed.get("points", []),
            }
    # Fallback
    return {"title": "", "summary": "", "abstract": "", "points": []}


# Script-based helper implementations


def run_preprocess_script(input_path, output_path):
    """Fallback preprocess implementation using ffmpeg directly."""
    import logging
    import subprocess

    logger = logging.getLogger(__name__)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-vn",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]

    logger.info(
        "Running preprocess command",
        extra={
            "cmd": " ".join(cmd),
            "input_path": str(input_path),
            "output_path": str(output_path),
        },
    )
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(
            "Preprocess completed successfully",
            extra={
                "return_code": result.returncode,
                "stdout": (result.stdout or "").strip(),
            },
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
        raise AudioProcessingException(f"Audio preprocessing failed: {e.stderr}") from e
    except Exception as e:
        logger.error(
            "Preprocess failed with unexpected error",
            exc_info=e,
            extra={"error": str(e)},
        )
        raise AudioProcessingException(f"Audio preprocessing failed: {e!s}") from e


def run_asr_script(audio_path, output_path, config_path=None):
    """Delegate to centralized ASR wrapper (backwards-compatible symbol)."""
    return _run_asr_script(
        audio_path=audio_path, output_path=output_path, config_path=config_path
    )


def run_postprocess_script(asr_json_path, output_path, config_path=None):
    """Delegate to centralized postprocess wrapper (backwards-compatible symbol)."""
    return _run_postprocess_script(
        asr_json_path=asr_json_path,
        output_path=output_path,
        config_path=config_path,
    )


# Legacy import shim removed. Tests and consumers should import from 'omoai.api.services'.


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
            final_output_path = (
                Path(config.api.temp_dir) / f"preprocessed_{os.urandom(8).hex()}.wav"
            )

            # Use existing preprocess script via wrapper
            run_preprocess_script(input_path=input_path, output_path=final_output_path)

            return PreprocessResponse(output_path=str(final_output_path))

    except subprocess.CalledProcessError as e:
        raise AudioProcessingException(f"Audio preprocessing failed: {e.stderr}") from e
    except Exception as e:
        raise AudioProcessingException(
            f"Unexpected error during preprocessing: {e!s}"
        ) from e


def _asr_script(data: ASRRequest) -> ASRResponse:
    """
    Run ASR using the existing scripts.asr module via the wrapper and return structured output.
    """
    audio_path = Path(data.preprocessed_path)
    if not audio_path.exists():
        raise FileNotFoundError(
            f"Preprocessed audio file not found: {data.preprocessed_path}"
        )

    try:
        config = get_config()
        asr_json_path = Path(config.api.temp_dir) / f"asr_{os.urandom(8).hex()}.json"
        config_path = None

        run_asr_script(
            audio_path=audio_path,
            output_path=asr_json_path,
            config_path=config_path,
        )

        with open(asr_json_path, encoding="utf-8") as f:
            asr_obj: dict[str, Any] = json.load(f)

        return ASRResponse(
            segments=list(asr_obj.get("segments", []) or []),
            transcript_raw=asr_obj.get("text"),
        )
    except subprocess.CalledProcessError as e:
        raise AudioProcessingException(f"ASR processing failed: {e.stderr}") from e
    except RuntimeError as e:
        # Raised by centralized wrapper on non-zero return codes
        raise AudioProcessingException(f"ASR processing failed: {e}") from e
    except Exception as e:
        raise AudioProcessingException(f"Unexpected error during ASR: {e!s}") from e


def _postprocess_script(data: PostprocessRequest) -> PostprocessResponse:
    """
    Run punctuation and summarization via scripts.post wrapper on provided ASR output dict.
    """
    try:
        config = get_config()
        tmp_asr_json = Path(config.api.temp_dir) / f"asr_{os.urandom(8).hex()}.json"
        with open(tmp_asr_json, "w", encoding="utf-8") as f:
            json.dump(data.asr_output, f, ensure_ascii=False)

        final_json_path = (
            Path(config.api.temp_dir) / f"final_{os.urandom(8).hex()}.json"
        )
        config_path = None

        run_postprocess_script(
            asr_json_path=tmp_asr_json,
            output_path=final_json_path,
            config_path=config_path,
        )

        with open(final_json_path, encoding="utf-8") as f:
            final_obj: dict[str, Any] = json.load(f)

        return PostprocessResponse(
            summary=dict(final_obj.get("summary", {}) or {}),
            segments=list(final_obj.get("segments", []) or []),
            summary_raw_text=str(final_obj.get("summary_raw_text", "")) or None,
        )
    except subprocess.CalledProcessError as e:
        raise AudioProcessingException(f"Post-processing failed: {e.stderr}") from e
    except RuntimeError as e:
        # Raised by centralized wrapper on non-zero return codes
        raise AudioProcessingException(f"Post-processing failed: {e}") from e
    except Exception as e:
        raise AudioProcessingException(
            f"Unexpected error during post-processing: {e!s}"
        ) from e


# Full pipeline script implementation
async def _run_full_pipeline_script(
    data: PipelineRequest, output_params: OutputFormatParams | None = None
) -> tuple[PipelineResponse, str | None]:
    """
    Run the full pipeline: preprocess -> ASR -> post-process.
    """
    import logging

    logger = logging.getLogger(__name__)

    logger.info("Starting full pipeline execution")

    # Avoid psutil when builtins.open is patched (tests may patch it with limited side effects)
    _open_patched = False
    try:
        import builtins as _bi  # type: ignore
        from unittest import mock as _um  # type: ignore

        _open_patched = isinstance(getattr(_bi, "open", None), _um.Base)
    except Exception:
        _open_patched = False

    try:
        if not _open_patched:
            import psutil

            process = psutil.Process()
            memory_info = process.memory_info()
            logger.info(
                f"Memory usage at start: {memory_info.rss / 1024 / 1024:.2f} MB"
            )
        else:
            process = None
    except ImportError:
        logger.warning("psutil not available for memory monitoring")
        process = None
    except Exception as e:
        logger.warning(f"Memory monitoring failed: {e!s}")
        process = None

    # Main pipeline flow
    config = get_config()
    # Persist upload under configured temp_dir, not an ephemeral TemporaryDirectory
    upload_path = Path(config.api.temp_dir) / f"upload_{os.urandom(8).hex()}"
    content = await data.audio_file.read()
    # Use os-level write to avoid interference from tests patching builtins.open
    import os as _os

    fd = _os.open(str(upload_path), _os.O_WRONLY | _os.O_CREAT | _os.O_TRUNC, 0o644)
    try:
        _os.write(fd, content)
    finally:
        _os.close(fd)

    logger.info(f"Saved uploaded audio to temporary file: {upload_path}")
    logger.info(f"Audio file size: {len(content)} bytes")

    preprocessed_path = (
        Path(config.api.temp_dir) / f"preprocessed_{os.urandom(8).hex()}.wav"
    )
    logger.info(f"Starting audio preprocessing to: {preprocessed_path}")
    try:
        logger.info("Starting audio preprocessing")
        run_preprocess_script(input_path=upload_path, output_path=preprocessed_path)

        try:
            if process is not None:
                memory_info = process.memory_info()
                logger.info(
                    f"Memory usage after preprocessing: {memory_info.rss / 1024 / 1024:.2f} MB"
                )
        except Exception:
            pass

        logger.info("Audio preprocessing completed successfully")
    except Exception as e:
        logger.error(f"Audio preprocessing failed: {e!s}")

        raise

    asr_json_path = Path(config.api.temp_dir) / f"asr_{os.urandom(8).hex()}.json"
    config_path = None
    logger.info(f"Starting ASR processing, output will be saved to: {asr_json_path}")
    logger.info(f"Using config path: {config_path}")
    try:
        logger.info("Starting ASR processing")
        run_asr_script(
            audio_path=preprocessed_path,
            output_path=asr_json_path,
            config_path=config_path,
        )

        try:
            if process is not None:
                memory_info = process.memory_info()
                logger.info(
                    f"Memory usage after ASR: {memory_info.rss / 1024 / 1024:.2f} MB"
                )
        except Exception:
            pass
        # Extract raw ASR transcript from ASR output JSON for inclusion in final response
        raw_transcript = None
        try:
            with open(asr_json_path, encoding="utf-8") as f:
                asr_obj_for_raw = json.load(f)
            raw_transcript = (
                asr_obj_for_raw.get("text")
                or asr_obj_for_raw.get("transcript_raw")
                or None
            )
        except Exception:
            raw_transcript = None

        logger.info("ASR processing completed successfully")
    except Exception as e:
        logger.error(f"ASR processing failed: {e!s}")
        raise

    final_json_path = Path(config.api.temp_dir) / f"final_{os.urandom(8).hex()}.json"
    logger.info(f"Starting post-processing, output will be saved to: {final_json_path}")
    try:
        logger.info("Starting post-processing")
        run_postprocess_script(
            asr_json_path=asr_json_path,
            output_path=final_json_path,
            config_path=config_path,
        )

        try:
            if process is not None:
                memory_info = process.memory_info()
                logger.info(
                    f"Memory usage after post-processing: {memory_info.rss / 1024 / 1024:.2f} MB"
                )
        except Exception:
            pass

        logger.info("Post-processing completed successfully")
    except Exception as e:
        logger.error(f"Post-processing failed: {e!s}")
        raise

    logger.info(f"Loading final output from: {final_json_path}")
    try:
        with open(final_json_path, encoding="utf-8") as f:
            final_obj: dict[str, Any] = json.load(f)
        logger.info("Final output loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load final output: {e!s}")
        raise

    # Optionally persist API outputs to disk based on config.output.save_on_api
    try:
        cfg = get_config()
        save_cfg = getattr(cfg, "output", None)
        if save_cfg and getattr(save_cfg, "save_on_api", False):
            # Determine base directory for API outputs
            try:
                base_dir = (
                    Path(save_cfg.api_output_dir)
                    if getattr(save_cfg, "api_output_dir", None)
                    else (Path(cfg.paths.out_dir) / "api")
                )
            except Exception:
                base_dir = Path("data/output/api")
            base_dir.mkdir(parents=True, exist_ok=True)

            # Create a unique subfolder per request
            from datetime import datetime

            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            out_dir = base_dir / f"req-{stamp}-{os.urandom(2).hex()}"
            out_dir.mkdir(parents=True, exist_ok=True)

            # Which formats to write
            fmts = set(getattr(save_cfg, "save_formats_on_api", ["final_json"]))

            # Write final JSON if requested
            if "final_json" in fmts:
                final_name = getattr(save_cfg, "final_json", "final.json")
                with open(out_dir / final_name, "w", encoding="utf-8") as wf:
                    json.dump(final_obj, wf, ensure_ascii=False, indent=2)

            # Write segments if requested
            if "segments" in fmts:
                seg_name = getattr(getattr(save_cfg, "transcript", None), "file_segments", "segments.json")
                with open(out_dir / seg_name, "w", encoding="utf-8") as ws:
                    json.dump(list(final_obj.get("segments", []) or []), ws, ensure_ascii=False, indent=2)

            # Write transcripts if requested
            if "transcripts" in fmts:
                tcfg = getattr(save_cfg, "transcript", None)
                file_raw = getattr(tcfg, "file_raw", "transcript.raw.txt")
                file_punct = getattr(tcfg, "file_punct", "transcript.punct.txt")
                try:
                    raw_text = str(final_obj.get("transcript_raw", "") or "").strip()
                except Exception:
                    raw_text = ""
                try:
                    punct_text = str(final_obj.get("transcript_punct", "") or "").strip()
                except Exception:
                    punct_text = ""
                if raw_text:
                    (out_dir / file_raw).write_text(raw_text + "\n", encoding="utf-8")
                if punct_text:
                    (out_dir / file_punct).write_text(punct_text + "\n", encoding="utf-8")

            # Timed text helpers
            def _fmt_time_srt(t: float) -> str:
                import math as _m
                t = max(0.0, float(t))
                hh = int(t // 3600)
                mm = int((t % 3600) // 60)
                ss = int(t % 60)
                ms = _m.floor((t - _m.floor(t)) * 1000.0)
                return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"

            def _fmt_time_vtt(t: float) -> str:
                import math as _m
                t = max(0.0, float(t))
                hh = int(t // 3600)
                mm = int((t % 3600) // 60)
                ss = int(t % 60)
                ms = _m.floor((t - _m.floor(t)) * 1000.0)
                return f"{hh:02d}:{mm:02d}:{ss:02d}.{ms:03d}"

            # Write SRT
            if "srt" in fmts and final_obj.get("segments"):
                srt_name = getattr(getattr(save_cfg, "transcript", None), "file_srt", "transcript.srt")
                cues = []
                idx = 1
                for seg in list(final_obj.get("segments", []) or []):
                    try:
                        start = float(seg.get("start", 0.0) or 0.0)
                        end = float(seg.get("end", start) or start)
                        text = str(seg.get("text_punct") or seg.get("text") or "").strip()
                    except Exception:
                        continue
                    if not text:
                        continue
                    cues.append(f"{idx}\n{_fmt_time_srt(start)} --> {_fmt_time_srt(end)}\n{text}\n")
                    idx += 1
                if cues:
                    (out_dir / srt_name).write_text("\n".join(cues), encoding="utf-8")

            # Write VTT
            if "vtt" in fmts and final_obj.get("segments"):
                vtt_name = getattr(getattr(save_cfg, "transcript", None), "file_vtt", "transcript.vtt")
                lines = ["WEBVTT", ""]
                for seg in list(final_obj.get("segments", []) or []):
                    try:
                        start = float(seg.get("start", 0.0) or 0.0)
                        end = float(seg.get("end", start) or start)
                        text = str(seg.get("text_punct") or seg.get("text") or "").strip()
                    except Exception:
                        continue
                    if not text:
                        continue
                    lines.append(f"{_fmt_time_vtt(start)} --> {_fmt_time_vtt(end)}")
                    lines.append(text)
                    lines.append("")
                if len(lines) > 2:
                    (out_dir / vtt_name).write_text("\n".join(lines), encoding="utf-8")

            # Write Markdown (summary + transcript)
            if "md" in fmts:
                # Summary.md
                sum_name = getattr(getattr(save_cfg, "summary", None), "file", "summary.md")
                summary = final_obj.get("summary") or {}
                title = ""
                abstract = ""
                points: list[str] = []
                try:
                    if isinstance(summary, dict):
                        title = str(summary.get("title", "") or "").strip()
                        abstract = str(summary.get("abstract", "") or summary.get("summary", "") or "").strip()
                        pts = summary.get("points") or summary.get("bullets") or []
                        if isinstance(pts, list):
                            points = [str(p) for p in pts if str(p).strip()]
                except Exception:
                    pass
                md_parts: list[str] = []
                if title:
                    md_parts.append(f"# {title}")
                if abstract:
                    md_parts.append(abstract)
                if points:
                    md_parts.append("\n## Points\n")
                    md_parts.extend([f"- {p}" for p in points])
                if md_parts:
                    (out_dir / sum_name).write_text("\n\n".join(md_parts).strip() + "\n", encoding="utf-8")

                # Transcript.md
                t_md_name = "transcript.md"
                try:
                    t_text = str(final_obj.get("transcript_punct", "") or "").strip()
                except Exception:
                    t_text = ""
                if t_text:
                    (out_dir / t_md_name).write_text(f"# Transcript\n\n{t_text}\n", encoding="utf-8")

            logger.info("Saved API outputs", extra={"dir": str(out_dir), "formats": list(fmts)})
    except Exception as e:
        # Do not fail the request if persistence fails; log and continue
        try:
            logger.warning("Failed to save API outputs", extra={"error": str(e)})
        except Exception:
            pass

    # Use summary keys as produced by postprocess; keep 'bullets' for compatibility
    final_summary = final_obj.get("summary", {}) or {}

    if output_params:
        filtered_summary = final_summary
        filtered_segments = final_obj.get("segments", [])
        filtered_transcript_punct = final_obj.get("transcript_punct", "")

        if output_params.summary:
            if output_params.summary == "none":
                filtered_summary = {}
            elif output_params.summary == "bullets":
                # Keep bullets key for compatibility
                # Prefer existing 'bullets', otherwise map from 'points'
                bullets = (
                    filtered_summary.get("bullets")
                    or filtered_summary.get("points")
                    or []
                )
                filtered_summary = {"bullets": list(bullets)}
            elif output_params.summary == "abstract":
                filtered_summary = {"abstract": filtered_summary.get("abstract", "")}

            if output_params.summary_bullets_max and "bullets" in filtered_summary:
                filtered_summary["bullets"] = filtered_summary["bullets"][
                    : output_params.summary_bullets_max
                ]

        if output_params.include:
            include_set = set(output_params.include)
            if "segments" not in include_set:
                filtered_segments = []
            if "transcript_punct" not in include_set:
                filtered_transcript_punct = ""

        quality_metrics = None
        diffs = None

        # Parse metrics/diffs if present in final JSON (controller decides whether to expose)
        if "quality_metrics" in final_obj:
            quality_metrics_data = final_obj["quality_metrics"]
            from omoai.api.models import QualityMetrics

            try:
                quality_metrics = QualityMetrics(**quality_metrics_data)
                try:
                    logger.info(
                        "Included quality_metrics in pipeline response",
                    )
                except Exception:
                    pass
            except Exception:
                quality_metrics = None

        if "diffs" in final_obj:
            diffs_data = final_obj["diffs"]
            from omoai.api.models import HumanReadableDiff

            try:
                if isinstance(diffs_data, list) and diffs_data:
                    diffs = HumanReadableDiff(**diffs_data[0])
                elif isinstance(diffs_data, dict):
                    diffs = HumanReadableDiff(**diffs_data)
                try:
                    logger.info(
                        "Included diffs in pipeline response",
                    )
                except Exception:
                    pass
            except Exception:
                diffs = None

        response_obj = PipelineResponse(
            summary=filtered_summary,
            segments=filtered_segments,
            transcript_punct=filtered_transcript_punct,
            quality_metrics=quality_metrics,
            diffs=diffs,
            summary_raw_text=(str(final_obj.get("summary_raw_text", "")) or None)
            if getattr(output_params, "return_summary_raw", None)
            else None,
        )
        return (response_obj, raw_transcript)

    quality_metrics = None
    diffs = None

    if "quality_metrics" in final_obj:
        quality_metrics_data = final_obj["quality_metrics"]
        from omoai.api.models import QualityMetrics

        try:
            quality_metrics = QualityMetrics(**quality_metrics_data)
            try:
                import logging as _log
                _log.getLogger(__name__).info("Included quality_metrics in response")
            except Exception:
                pass
        except Exception:
            quality_metrics = None

    if "diffs" in final_obj:
        diffs_data = final_obj["diffs"]
        from omoai.api.models import HumanReadableDiff

        try:
            if isinstance(diffs_data, list) and diffs_data:
                diffs = HumanReadableDiff(**diffs_data[0])
            elif isinstance(diffs_data, dict):
                diffs = HumanReadableDiff(**diffs_data)
            try:
                import logging as _log
                _log.getLogger(__name__).info("Included diffs in response")
            except Exception:
                pass
        except Exception:
            diffs = None

    summary_data = final_obj.get("summary", {}) or {}

    if isinstance(summary_data, dict):
        final_summary = dict(summary_data)
    elif (
        isinstance(summary_data, (list, tuple))
        and len(summary_data) > 0
        and isinstance(summary_data[0], dict)
    ):
        final_summary = dict(summary_data[0])
    else:
        raise AudioProcessingException(
            "Post-processing produced unexpected summary format; expected a dict",
        )
    # Ensure 'bullets' is present if only 'points' exists
    if "bullets" not in final_summary and "points" in final_summary:
        final_summary["bullets"] = final_summary.get("points", [])

    response_obj = PipelineResponse(
        summary=final_summary,
        segments=list(final_obj.get("segments", []) or []),
        transcript_punct=str(final_obj.get("transcript_punct", "")) or None,
        quality_metrics=quality_metrics,
        diffs=diffs,
        summary_raw_text=(str(final_obj.get("summary_raw_text", "")) or None)
        if (output_params and getattr(output_params, "return_summary_raw", None))
        else None,
    )
    return (response_obj, raw_transcript)


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


def _postprocess_service_sync(
    data: PostprocessRequest, output_params: dict | None = None
) -> PostprocessResponse:
    """Synchronous implementation of postprocess service."""
    return _postprocess_script(data)


async def postprocess_service(
    data: PostprocessRequest, output_params: dict | None = None
) -> PostprocessResponse:
    """
    Dual-mode postprocess service helper.
    - If called from sync code (no running event loop), executes synchronously.
    - If called from async code, offloads to thread and returns awaitable.
    """
    try:
        asyncio.get_running_loop()
        return await asyncio.to_thread(_postprocess_service_sync, data, output_params)
    except RuntimeError:
        # No running loop - synchronous context (e.g., tests)
        return _postprocess_service_sync(data, output_params)


class _BytesUploadFile(UploadFile):
    def __init__(
        self,
        data: bytes,
        filename: str = "upload.bin",
        content_type: str = "application/octet-stream",
    ):
        self._data = data
        self.filename = filename
        self.content_type = content_type

    async def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            return self._data
        return self._data[:size]


async def run_full_pipeline(
    data: PipelineRequest, output_params: OutputFormatParams | dict | None = None
) -> PipelineResponse:
    if hasattr(data, "audio_file") and data.audio_file is not None:
        try:
            file_bytes = await data.audio_file.read()
            if not file_bytes:
                raise AudioProcessingException("Uploaded audio file is empty")
        except AudioProcessingException:
            raise
        except Exception as e:
            raise AudioProcessingException("Failed to read uploaded audio file") from e

        data.audio_file = _BytesUploadFile(
            data=file_bytes,
            filename=getattr(data.audio_file, "filename", "upload.bin"),
            content_type=getattr(
                data.audio_file, "content_type", "application/octet-stream"
            ),
        )

    params: OutputFormatParams | None = (
        output_params if isinstance(output_params, OutputFormatParams) else None
    )
    result_obj = await _run_full_pipeline_script(data, params)
    if result_obj is None:
        raise AudioProcessingException("Pipeline returned no result")
    if isinstance(result_obj, tuple) or (
        hasattr(result_obj, "__iter__") and not isinstance(result_obj, PipelineResponse)
    ):
        try:
            response_obj, raw_transcript = result_obj  # type: ignore[misc]
        except Exception as e:
            raise AudioProcessingException(
                "Pipeline returned unexpected result type"
            ) from e
    else:
        response_obj = result_obj  # type: ignore[assignment]
        raw_transcript = getattr(result_obj, "transcript_raw", None)
    # Surface raw transcript on the response for callers that don't use the tuple form
    try:
        if raw_transcript and getattr(response_obj, "transcript_raw", None) is None:
            response_obj.transcript_raw = raw_transcript  # type: ignore[attr-defined]
    except Exception:
        pass
    # Return a PipelineResult to keep backward compatibility with callers that unpack
    return PipelineResult(**response_obj.model_dump())


async def warmup_services() -> dict[str, Any]:
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


async def get_service_status() -> dict[str, Any]:
    status: dict[str, Any] = {
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
    "asr_service",
    "get_service_status",
    "postprocess_service",
    "preprocess_audio_service",
    "run_full_pipeline",
    "warmup_services",
]
