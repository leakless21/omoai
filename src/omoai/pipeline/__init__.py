"""
In-memory pipeline module for OMOAI audio processing.

This module provides efficient in-memory data processing functions that eliminate
disk I/O bottlenecks between pipeline stages.
"""

from .preprocess import (
    preprocess_audio_to_tensor, 
    preprocess_audio_bytes,
    get_audio_info,
    validate_audio_input,
    preprocess_file_to_wav_bytes,
)
from .asr import run_asr_inference, ASRResult, ASRSegment, ChunkFormerASR
from .postprocess import (
    postprocess_transcript, 
    PostprocessResult, 
    SummaryResult,
    punctuate_transcript,
    summarize_text,
)
from .pipeline import (
    run_full_pipeline_memory, 
    PipelineResult,
    run_pipeline_batch,
)
from .exceptions import (
    OMOPipelineError,
    OMOAudioError,
    OMOASRError,
    OMOPostprocessError,
    OMOConfigError,
    OMOModelError,
    OMOIOError,
)

__all__ = [
    # Preprocessing
    "preprocess_audio_to_tensor",
    "preprocess_audio_bytes",
    "get_audio_info",
    "validate_audio_input", 
    "preprocess_file_to_wav_bytes",
    
    # ASR
    "run_asr_inference",
    "ASRResult",
    "ASRSegment",
    "ChunkFormerASR",
    
    # Postprocessing
    "postprocess_transcript",
    "PostprocessResult",
    "SummaryResult",
    "punctuate_transcript",
    "summarize_text",
    
    # Pipeline
    "run_full_pipeline_memory",
    "PipelineResult",
    "run_pipeline_batch",
    
    # Exceptions
    "OMOPipelineError",
    "OMOAudioError",
    "OMOASRError",
    "OMOPostprocessError",
    "OMOConfigError",
    "OMOModelError",
    "OMOIOError",
]