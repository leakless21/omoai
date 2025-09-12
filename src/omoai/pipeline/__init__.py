"""
Pipeline package exports for OMOAI.

The in-memory postprocessing and full in-memory pipeline were removed to avoid
confusion with the production script-based pipeline. This package now exposes
preprocessing and ASR utilities only; postprocessing should be performed via
the scripts in the scripts/ directory (invoked by wrappers in
src.omoai.api.scripts).
"""
from .preprocess import (
    preprocess_audio_to_tensor,
    preprocess_audio_bytes,
    get_audio_info,
    validate_audio_input,
    preprocess_file_to_wav_bytes,
)
from .asr import run_asr_inference, ASRResult, ASRSegment, ChunkFormerASR
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

    # Exceptions
    "OMOPipelineError",
    "OMOAudioError",
    "OMOASRError",
    "OMOPostprocessError",
    "OMOConfigError",
    "OMOModelError",
    "OMOIOError",
]