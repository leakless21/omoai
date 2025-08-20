"""
Complete in-memory pipeline orchestrator for OMOAI.

This module provides the main pipeline function that coordinates preprocessing,
ASR, and postprocessing entirely in memory for maximum efficiency.
"""
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, BinaryIO

from .preprocess import preprocess_audio_to_tensor, validate_audio_input, get_audio_info
from .asr import run_asr_inference, ASRResult, ASRSegment  
from .postprocess import postprocess_transcript, PostprocessResult, SummaryResult
from ..config import OmoAIConfig
from ..logging import get_logger, performance_context, log_error, get_performance_logger


@dataclass
class PipelineResult:
    """Complete pipeline processing result."""
    # Final outputs
    segments: List[ASRSegment]
    transcript_raw: str
    transcript_punctuated: str
    summary: SummaryResult
    
    # Intermediate results (for debugging/analysis)
    asr_result: ASRResult
    postprocess_result: PostprocessResult
    
    # Performance metrics
    timing: Dict[str, float]
    metadata: Dict[str, Any]


def run_full_pipeline_memory(
    audio_input: Union[bytes, Path, str, BinaryIO],
    config: Optional[OmoAIConfig] = None,
    save_intermediates: bool = False,
    output_dir: Optional[Path] = None,
    validate_input: bool = True,
    max_audio_duration: Optional[float] = None,
) -> PipelineResult:
    """
    Run the complete OMOAI pipeline entirely in memory.
    
    Args:
        audio_input: Audio data (bytes, file path, or file-like object)
        config: Configuration object (loads default if None)
        save_intermediates: Whether to save intermediate results to disk
        output_dir: Directory for intermediate files (if save_intermediates=True)
        validate_input: Whether to validate audio before processing
        max_audio_duration: Maximum allowed audio duration in seconds
        
    Returns:
        PipelineResult with all outputs and timing information
        
    Raises:
        ValueError: If input is invalid or processing fails
        FileNotFoundError: If input file doesn't exist
        RuntimeError: If any pipeline stage fails
    """
    # Initialize logging
    logger = get_logger("omoai.pipeline")
    perf_logger = get_performance_logger()
    
    # Generate unique pipeline ID for tracing
    pipeline_id = str(uuid.uuid4())
    
    logger.info("Starting full pipeline", extra={
        "pipeline_id": pipeline_id,
        "input_type": type(audio_input).__name__,
        "save_intermediates": save_intermediates,
        "validate_input": validate_input,
        "max_audio_duration": max_audio_duration,
    })
    
    timing = {}
    start_time = time.time()
    
    # Load configuration
    if config is None:
        from ..config import get_config
        config = get_config()
    
    try:
        # Stage 0: Input validation (optional)
        if validate_input:
            with performance_context("input_validation", logger=logger):
                validation_start = time.time()
                validate_audio_input(
                    audio_input,
                    max_duration_seconds=max_audio_duration,
                    min_duration_seconds=0.1,
                )
                timing["validation"] = time.time() - validation_start
                logger.debug("Input validation completed", extra={
                    "pipeline_id": pipeline_id,
                    "validation_time_ms": timing["validation"] * 1000,
                })
        
        # Get audio info for metadata
        audio_info = get_audio_info(audio_input)
        logger.info("Audio info extracted", extra={
            "pipeline_id": pipeline_id,
            "duration_seconds": audio_info.get("duration", "unknown"),
            "sample_rate": audio_info.get("sample_rate", "unknown"),
            "channels": audio_info.get("channels", "unknown"),
        })
        
        # Stage 1: Preprocessing  
        with performance_context("audio_preprocessing", logger=logger):
            preprocess_start = time.time()
            audio_tensor, sample_rate = preprocess_audio_to_tensor(
                audio_input,
                target_sample_rate=16000,
                normalize=True,
                return_sample_rate=True,
            )
            timing["preprocessing"] = time.time() - preprocess_start
            
            logger.info("Audio preprocessing completed", extra={
                "pipeline_id": pipeline_id,
                "preprocessing_time_ms": timing["preprocessing"] * 1000,
                "tensor_shape": list(audio_tensor.shape),
                "output_sample_rate": sample_rate,
                "tensor_dtype": str(audio_tensor.dtype),
            })
        
        # Save preprocessed audio if requested
        if save_intermediates and output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            import torchaudio
            preprocessed_path = output_dir / "preprocessed.wav"
            torchaudio.save(str(preprocessed_path), audio_tensor, sample_rate)
        
        # Stage 2: ASR
        with performance_context("asr_inference", logger=logger):
            asr_start = time.time()
            asr_result = run_asr_inference(
                audio_tensor,
                config=config,
                sample_rate=sample_rate,
            )
            timing["asr"] = time.time() - asr_start
            
            # Calculate real-time factor
            audio_duration = audio_info.get("duration", 0)
            rtf = timing["asr"] / audio_duration if audio_duration > 0 else 0
            
            logger.info("ASR inference completed", extra={
                "pipeline_id": pipeline_id,
                "asr_time_ms": timing["asr"] * 1000,
                "audio_duration_seconds": audio_duration,
                "real_time_factor": rtf,
                "segments_count": len(asr_result.segments),
                "transcript_length": len(asr_result.transcript),
                "confidence_avg": sum(seg.confidence or 0 for seg in asr_result.segments) / len(asr_result.segments) if asr_result.segments else 0,
            })
        
        # Save ASR results if requested
        if save_intermediates and output_dir:
            import json
            asr_path = output_dir / "asr.json"
            asr_json = {
                "audio": {
                    "sr": asr_result.sample_rate,
                    "duration_s": asr_result.audio_duration_seconds,
                },
                "segments": [
                    {
                        "start": seg.start,
                        "end": seg.end,
                        "text_raw": seg.text,
                        "confidence": seg.confidence,
                    }
                    for seg in asr_result.segments
                ],
                "transcript_raw": asr_result.transcript,
                "metadata": asr_result.metadata,
            }
            with open(asr_path, "w", encoding="utf-8") as f:
                json.dump(asr_json, f, ensure_ascii=False, indent=2)
        
        # Stage 3: Postprocessing
        with performance_context("postprocessing", logger=logger):
            postprocess_start = time.time()
            postprocess_result = postprocess_transcript(asr_result, config)
            timing["postprocessing"] = time.time() - postprocess_start
            
            logger.info("Postprocessing completed", extra={
                "pipeline_id": pipeline_id,
                "postprocessing_time_ms": timing["postprocessing"] * 1000,
                "punctuated_length": len(postprocess_result.transcript_punctuated),
                "summary_length": len(postprocess_result.summary.abstract) if postprocess_result.summary else 0,
                "has_summary": postprocess_result.summary is not None,
            })
        
        # Save final results if requested
        if save_intermediates and output_dir:
            import json
            final_path = output_dir / "final.json"
            final_json = {
                "segments": [
                    {
                        "start": seg.start,
                        "end": seg.end,
                        "text_raw": asr_result.segments[i].text if i < len(asr_result.segments) else "",
                        "text_punct": seg.text,
                        "confidence": seg.confidence,
                    }
                    for i, seg in enumerate(postprocess_result.segments)
                ],
                "transcript_raw": asr_result.transcript,
                "transcript_punct": postprocess_result.transcript_punctuated,
                "summary": {
                    "bullets": postprocess_result.summary.bullets,
                    "abstract": postprocess_result.summary.abstract,
                },
                "metadata": postprocess_result.metadata,
            }
            with open(final_path, "w", encoding="utf-8") as f:
                json.dump(final_json, f, ensure_ascii=False, indent=2)
            
            # Save separate text files if configured
            if config.output.write_separate_files:
                # Transcript file
                transcript_path = output_dir / config.output.transcript_file
                with open(transcript_path, "w", encoding="utf-8") as f:
                    f.write(postprocess_result.transcript_punctuated)
                
                # Summary file
                summary_path = output_dir / config.output.summary_file
                with open(summary_path, "w", encoding="utf-8") as f:
                    for bullet in postprocess_result.summary.bullets:
                        f.write(f"- {bullet}\n")
                    if postprocess_result.summary.bullets and postprocess_result.summary.abstract:
                        f.write("\n")
                    if postprocess_result.summary.abstract:
                        f.write(postprocess_result.summary.abstract + "\n")
        
        # Calculate total timing
        timing["total"] = time.time() - start_time
        
        # Prepare final metadata
        metadata = {
            "pipeline_version": "in_memory_v1",
            "audio_info": audio_info,
            "config_summary": {
                "asr_device": config.asr.device,
                "asr_autocast": config.asr.autocast_dtype,
                "punctuation_model": config.punctuation.llm.model_id,
                "summarization_model": config.summarization.llm.model_id,
            },
            "performance": {
                "audio_duration": asr_result.audio_duration_seconds,
                "processing_time": timing["total"],
                "real_time_factor": timing["total"] / asr_result.audio_duration_seconds if asr_result.audio_duration_seconds > 0 else 0,
                "stages": timing,
            },
            "quality_metrics": {
                "segments_count": len(asr_result.segments),
                "transcript_length": len(asr_result.transcript),
                "punctuated_length": len(postprocess_result.transcript_punctuated),
                "summary_bullets": len(postprocess_result.summary.bullets),
                "has_abstract": bool(postprocess_result.summary.abstract),
            },
        }
        
        # Calculate total time and performance metrics
        total_time = time.time() - start_time
        timing["total"] = total_time
        
        # Log final pipeline completion
        logger.info("Pipeline completed successfully", extra={
            "pipeline_id": pipeline_id,
            "total_time_ms": total_time * 1000,
            "stages_completed": list(timing.keys()),
            "performance_breakdown": {k: round(v * 1000, 2) for k, v in timing.items()},
            "real_time_factor_total": total_time / audio_info.get("duration", 1),
                            "final_transcript_length": len(postprocess_result.transcript_punctuated),
            "segments_count": len(postprocess_result.segments),
        })
        
        # Record performance metrics
        perf_logger.log_operation(
            operation="full_pipeline",
            duration_ms=total_time * 1000,
            success=True,
            pipeline_id=pipeline_id,
            stages_count=len(timing),
            audio_duration_seconds=audio_info.get("duration", 0),
            real_time_factor=total_time / audio_info.get("duration", 1) if audio_info.get("duration", 0) > 0 else 0,
        )
        
        return PipelineResult(
            # Final outputs
            segments=postprocess_result.segments,
            transcript_raw=asr_result.transcript,
            transcript_punctuated=postprocess_result.transcript_punctuated,
            summary=postprocess_result.summary,
            
            # Intermediate results
            asr_result=asr_result,
            postprocess_result=postprocess_result,
            
            # Performance and metadata
            timing=timing,
            metadata=metadata,
        )
        
    except Exception as e:
        # Enhanced error reporting with timing info
        error_timing = time.time() - start_time
        
        # Log detailed error information
        log_error(
            message=f"Pipeline failed after {error_timing:.2f}s",
            error=e,
            error_type="PIPELINE_FAILURE",
            error_code="PIPELINE_001",
            remediation="Check input validity, configuration, and model availability",
            logger=logger,
            pipeline_id=pipeline_id,
            stages_completed=list(timing.keys()),
            error_timing_seconds=error_timing,
        )
        
        # Record failed operation
        perf_logger.log_operation(
            operation="full_pipeline",
            duration_ms=error_timing * 1000,
            success=False,
            error_type="PIPELINE_FAILURE",
            pipeline_id=pipeline_id,
            stages_completed=list(timing.keys()),
        )
        
        raise RuntimeError(
            f"Pipeline failed after {error_timing:.2f}s: {e}. "
            f"Completed stages: {list(timing.keys())}"
        )


def run_pipeline_batch(
    audio_inputs: List[Union[bytes, Path, str]],
    config: Optional[OmoAIConfig] = None,
    max_concurrent: int = 1,
    validate_inputs: bool = True,
    save_intermediates: bool = False,
    output_base_dir: Optional[Path] = None,
) -> List[PipelineResult]:
    """
    Run pipeline on multiple audio inputs.
    
    Args:
        audio_inputs: List of audio inputs to process
        config: Configuration object
        max_concurrent: Maximum concurrent processing (currently serial only)
        validate_inputs: Whether to validate each input
        save_intermediates: Whether to save intermediate files
        output_base_dir: Base directory for outputs (creates subdirs per input)
        
    Returns:
        List of PipelineResult objects
        
    Note:
        Currently processes serially. Concurrent processing can be added later.
    """
    if config is None:
        from ..config import get_config
        config = get_config()
    
    results = []
    
    for i, audio_input in enumerate(audio_inputs):
        try:
            # Determine output directory for this input
            output_dir = None
            if save_intermediates and output_base_dir:
                if isinstance(audio_input, (str, Path)):
                    input_name = Path(audio_input).stem
                else:
                    input_name = f"input_{i:03d}"
                output_dir = Path(output_base_dir) / input_name
            
            # Process single input
            result = run_full_pipeline_memory(
                audio_input=audio_input,
                config=config,
                save_intermediates=save_intermediates,
                output_dir=output_dir,
                validate_input=validate_inputs,
            )
            
            results.append(result)
            
        except Exception as e:
            # Continue processing other inputs on error
            print(f"Warning: Failed to process input {i}: {e}")
            continue
    
    return results


# Legacy compatibility function for script-based workflows
def run_pipeline_legacy(
    audio_path: Path,
    output_dir: Path,
    config_path: Optional[Path] = None,
    model_dir: Optional[Path] = None,
) -> int:
    """
    Legacy compatibility function that mimics the original main.py interface.
    
    Args:
        audio_path: Input audio file path
        output_dir: Output directory path  
        config_path: Configuration file path
        model_dir: Model directory (overrides config)
        
    Returns:
        Exit code (0 = success, non-zero = error)
    """
    try:
        # Load configuration
        if config_path:
            from ..config import load_config
            config = load_config(config_path)
        else:
            from ..config import get_config
            config = get_config()
        
        # Override model directory if provided
        if model_dir:
            config.paths.chunkformer_checkpoint = model_dir
        
        # Run pipeline with intermediate file saving
        result = run_full_pipeline_memory(
            audio_input=audio_path,
            config=config,
            save_intermediates=True,
            output_dir=output_dir,
            validate_input=True,
        )
        
        print(f"[orchestrator] Completed. Output: {output_dir / 'final.json'}")
        print(f"[orchestrator] Processing time: {result.timing['total']:.2f}s")
        print(f"[orchestrator] Real-time factor: {result.metadata['performance']['real_time_factor']:.2f}x")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"[orchestrator] File not found: {e}")
        return 1
    except ValueError as e:
        print(f"[orchestrator] Invalid input: {e}")
        return 2
    except RuntimeError as e:
        print(f"[orchestrator] Processing failed: {e}")
        return 3
    except Exception as e:
        print(f"[orchestrator] Unexpected error: {e}")
        return 4
