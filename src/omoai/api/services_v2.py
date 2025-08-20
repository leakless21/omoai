"""
Enhanced API services using in-memory pipeline and singleton models.

This module provides high-performance API services that use cached models
and eliminate subprocess calls for dramatic performance improvements.
"""
import asyncio
import time
from typing import Any, Dict, List
from pathlib import Path

from ..config import get_config
from ..pipeline import (
    run_full_pipeline_memory,
    preprocess_audio_bytes,
    run_asr_inference, 
    postprocess_transcript,
    ASRResult,
    ASRSegment,
    PipelineResult,
)
from .singletons import get_asr_model, get_punctuation_processor, get_summarization_processor
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


async def preprocess_audio_service_v2(data: PreprocessRequest) -> PreprocessResponse:
    """
    Enhanced preprocessing service using in-memory processing.
    
    Args:
        data: PreprocessRequest containing the uploaded audio file
        
    Returns:
        PreprocessResponse containing processed audio bytes (base64 encoded)
        
    Raises:
        AudioProcessingException: If preprocessing fails
    """
    try:
        # Read uploaded file content
        content = await data.audio_file.read()
        
        # Process audio in memory
        loop = asyncio.get_event_loop()
        processed_bytes = await loop.run_in_executor(
            None, 
            preprocess_audio_bytes,
            content
        )
        
        # For compatibility with existing API, we could save to temp file
        # But ideally we'd return the bytes directly or use a different flow
        config = get_config()
        temp_path = Path(config.api.temp_dir) / f"preprocessed_v2_{int(time.time() * 1000)}.wav"
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(temp_path, "wb") as f:
            f.write(processed_bytes)
        
        return PreprocessResponse(output_path=str(temp_path))
        
    except Exception as e:
        raise AudioProcessingException(f"Enhanced preprocessing failed: {str(e)}")


async def asr_service_v2(data: ASRRequest) -> ASRResponse:
    """
    Enhanced ASR service using cached models and in-memory processing.
    
    Args:
        data: ASRRequest containing path to preprocessed audio
        
    Returns:
        ASRResponse with transcription segments
        
    Raises:
        AudioProcessingException: If ASR processing fails
    """
    try:
        audio_path = Path(data.preprocessed_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Preprocessed audio file not found: {data.preprocessed_path}")
        
        # Get cached ASR model
        asr_model = get_asr_model()
        
        # Run ASR inference using cached model
        loop = asyncio.get_event_loop()
        result: ASRResult = await loop.run_in_executor(
            None,
            run_asr_inference,
            audio_path,
            None,  # Use default config
            None,  # Use default model checkpoint
            None   # Use default device
        )
        
        # Convert to API response format
        segments = [
            {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "confidence": seg.confidence
            }
            for seg in result.segments
        ]
        
        return ASRResponse(segments=segments)
        
    except Exception as e:
        raise AudioProcessingException(f"Enhanced ASR processing failed: {str(e)}")


async def postprocess_service_v2(data: PostprocessRequest) -> PostprocessResponse:
    """
    Enhanced postprocessing service using cached models.
    
    Args:
        data: PostprocessRequest containing ASR output
        
    Returns:
        PostprocessResponse with punctuation and summary
        
    Raises:
        AudioProcessingException: If postprocessing fails
    """
    try:
        # Convert API format to ASRResult
        segments = [
            ASRSegment(
                start=seg.get("start", 0.0),
                end=seg.get("end", 0.0),
                text=seg.get("text", ""),
                confidence=seg.get("confidence")
            )
            for seg in data.asr_output.get("segments", [])
        ]
        
        asr_result = ASRResult(
            segments=segments,
            transcript=" ".join(seg.text for seg in segments if seg.text),
            audio_duration_seconds=data.asr_output.get("audio_duration", 0.0),
            sample_rate=data.asr_output.get("sample_rate", 16000),
            metadata=data.asr_output.get("metadata", {})
        )
        
        # Run postprocessing using cached models
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            postprocess_transcript,
            asr_result,
            None  # Use default config
        )
        
        # Convert to API response format
        segments = [
            {
                "start": seg.start,
                "end": seg.end,
                "text_raw": (
                    asr_result.segments[i].text 
                    if i < len(asr_result.segments) 
                    else ""
                ),
                "text_punct": seg.text,
                "confidence": seg.confidence
            }
            for i, seg in enumerate(result.segments)
        ]
        
        summary = {
            "bullets": result.summary.bullets,
            "abstract": result.summary.abstract
        }
        
        return PostprocessResponse(summary=summary, segments=segments)
        
    except Exception as e:
        raise AudioProcessingException(f"Enhanced postprocessing failed: {str(e)}")


async def run_full_pipeline_v2(data: PipelineRequest) -> PipelineResponse:
    """
    Enhanced full pipeline using in-memory processing and cached models.
    
    This is the most efficient way to process audio, eliminating all
    subprocess calls and disk I/O between stages.
    
    Args:
        data: PipelineRequest containing the uploaded audio file
        
    Returns:
        PipelineResponse with final transcript, summary, and segments
        
    Raises:
        AudioProcessingException: If pipeline processing fails
    """
    try:
        # Read uploaded file content
        content = await data.audio_file.read()
        
        # Determine output directory for debugging/logging if needed
        config = get_config()
        save_intermediates = config.api.enable_progress_output  # Use progress setting
        output_dir = None
        if save_intermediates:
            output_dir = Path(config.api.temp_dir) / f"pipeline_v2_{int(time.time() * 1000)}"
        
        # Run complete pipeline in memory
        loop = asyncio.get_event_loop()
        result: PipelineResult = await loop.run_in_executor(
            None,
            run_full_pipeline_memory,
            content,  # Audio bytes
            None,     # Use default config
            save_intermediates,
            output_dir,
            True,     # Validate input
            None      # No max duration limit
        )
        
        # Convert to API response format
        segments = [
            {
                "start": seg.start,
                "end": seg.end,
                "text_raw": (
                    result.asr_result.segments[i].text 
                    if i < len(result.asr_result.segments) 
                    else ""
                ),
                "text_punct": seg.text,
                "confidence": seg.confidence
            }
            for i, seg in enumerate(result.segments)
        ]
        
        summary = {
            "bullets": result.summary.bullets,
            "abstract": result.summary.abstract,
            "metadata": {
                "processing_time": result.timing["total"],
                "real_time_factor": result.metadata["performance"]["real_time_factor"],
                "audio_duration": result.metadata["performance"]["audio_duration"],
                "model_info": result.metadata.get("config_summary", {}),
                "quality_metrics": result.metadata.get("quality_metrics", {})
            }
        }
        
        return PipelineResponse(summary=summary, segments=segments)
        
    except Exception as e:
        raise AudioProcessingException(f"Enhanced pipeline processing failed: {str(e)}")


async def run_pipeline_with_performance_metrics(data: PipelineRequest) -> Dict[str, Any]:
    """
    Run the enhanced pipeline and return detailed performance metrics.
    
    This is useful for monitoring and optimization purposes.
    
    Args:
        data: PipelineRequest containing the uploaded audio file
        
    Returns:
        Dictionary with results and detailed performance metrics
    """
    start_time = time.time()
    
    try:
        # Run enhanced pipeline
        response = await run_full_pipeline_v2(data)
        
        total_time = time.time() - start_time
        
        # Extract performance data from response
        metadata = response.summary.get("metadata", {})
        
        return {
            "success": True,
            "response": response.model_dump(),
            "performance": {
                "api_total_time": total_time,
                "pipeline_time": metadata.get("processing_time", 0),
                "api_overhead": total_time - metadata.get("processing_time", 0),
                "real_time_factor": metadata.get("real_time_factor", 0),
                "audio_duration": metadata.get("audio_duration", 0),
                "quality_metrics": metadata.get("quality_metrics", {}),
                "model_info": metadata.get("model_info", {})
            },
            "timing_breakdown": {
                "file_upload": "included_in_api_overhead",
                "preprocessing": "included_in_pipeline_time",
                "asr": "included_in_pipeline_time", 
                "postprocessing": "included_in_pipeline_time",
                "response_formatting": "included_in_api_overhead"
            }
        }
        
    except Exception as e:
        total_time = time.time() - start_time
        return {
            "success": False,
            "error": str(e),
            "performance": {
                "api_total_time": total_time,
                "failed_after": total_time
            }
        }


# Health check functions for monitoring
async def health_check_models() -> Dict[str, Any]:
    """
    Check the health of all cached models.
    
    Returns:
        Dictionary with model health status
    """
    try:
        from .singletons import model_singletons
        
        model_info = model_singletons.get_model_info()
        
        # Test ASR model
        asr_healthy = False
        try:
            asr_model = get_asr_model()
            asr_healthy = asr_model._is_initialized
        except Exception:
            pass
        
        # Test punctuation processor
        punct_healthy = False
        try:
            punct_processor = get_punctuation_processor()
            punct_healthy = punct_processor._is_initialized
        except Exception:
            pass
        
        # Test summarization processor
        summ_healthy = False
        try:
            summ_processor = get_summarization_processor()
            summ_healthy = summ_processor._is_initialized
        except Exception:
            pass
        
        return {
            "status": "healthy" if all([asr_healthy, punct_healthy, summ_healthy]) else "degraded",
            "models": {
                "asr": {
                    "healthy": asr_healthy,
                    "loaded": model_info["asr"]["loaded"],
                    "device": model_info["asr"]["device"]
                },
                "punctuation": {
                    "healthy": punct_healthy,
                    "loaded": model_info["punctuation"]["loaded"],
                    "model_id": model_info["punctuation"]["model_id"]
                },
                "summarization": {
                    "healthy": summ_healthy,
                    "loaded": model_info["summarization"]["loaded"],
                    "reuses_punctuation": model_info["summarization"]["reuses_punctuation"]
                }
            },
            "security": model_info["config"]["security"],
            "timestamp": time.time()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }
