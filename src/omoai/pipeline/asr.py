"""
In-memory ASR inference for OMOAI using ChunkFormer.

This module provides efficient ASR processing that works with tensors and
returns structured data without intermediate file I/O.
"""
import io
import os
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, BinaryIO

import torch
import numpy as np
from pydub import AudioSegment  # type: ignore

from ..config import OmoAIConfig, ASRConfig
from ..logging_system import get_logger, performance_context, log_error, get_performance_logger
from .exceptions import OMOASRError, OMOConfigError, OMOModelError, OMOAudioError

# Module logger
logger = get_logger("omoai.pipeline.asr")


@dataclass
class ASRSegment:
    """Represents a single ASR segment with timing information."""
    start: float
    end: float
    text: str
    confidence: Optional[float] = None


@dataclass  
class ASRResult:
    """Complete ASR processing result."""
    segments: List[ASRSegment]
    transcript: str
    audio_duration_seconds: float
    sample_rate: int
    metadata: Dict[str, Any]


class ChunkFormerASR:
    """ChunkFormer ASR engine for in-memory inference."""
    
    def __init__(
        self,
        model_checkpoint: Union[Path, str],
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        Initialize ChunkFormer ASR engine.
        
        Args:
            model_checkpoint: Path to ChunkFormer model checkpoint
            device: Target device (auto-detect if None)
        """
        self.model_checkpoint = Path(model_checkpoint)
        self.device = self._resolve_device(device)
        
        # Model components (loaded lazily)
        self.model: Optional[Any] = None
        self.char_dict: Optional[Dict[str, int]] = None
        self._is_initialized = False
        
    def _resolve_device(self, device: Optional[Union[str, torch.device]]) -> torch.device:
        """Resolve device with auto-detection."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(device)
    
    def initialize(self) -> None:
        """Initialize the model (lazy loading)."""
        if self._is_initialized:
            return
        
        try:
            # Import ChunkFormer components
            from chunkformer import decode as cfdecode  # type: ignore
            
            # Initialize model
            model, char_dict = cfdecode.init(str(self.model_checkpoint), self.device)
            self.model = model
            self.char_dict = char_dict
            self._is_initialized = True
            
        except ImportError as e:
            raise OMOModelError(
                f"Failed to import ChunkFormer components. "
                f"Ensure ChunkFormer is properly installed. Error: {e}"
            ) from e
        except Exception as e:
            raise OMOASRError(f"Failed to initialize ChunkFormer model: {e}") from e
    
    def process_tensor(
        self,
        audio_tensor: torch.Tensor,
        sample_rate: int = 16000,
        chunk_size: int = 64,
        left_context_size: int = 128,
        right_context_size: int = 128,
        total_batch_duration_s: int = 1800,
        autocast_dtype: Optional[str] = "fp16",
    ) -> ASRResult:
        """
        Process audio tensor directly for ASR inference.
        
        Args:
            audio_tensor: Audio tensor of shape (1, samples) or (samples,)
            sample_rate: Audio sample rate (should be 16kHz)
            chunk_size: ChunkFormer chunk size
            left_context_size: Left context size
            right_context_size: Right context size
            total_batch_duration_s: Max batch duration in seconds
            autocast_dtype: Autocast precision (fp16, bf16, fp32, None)
            
        Returns:
            ASRResult with segments and transcript
        """
        self.initialize()

        # Diagnostic: log tensor shape and incoming sample_rate
        try:
            tensor_shape = tuple(audio_tensor.shape) if hasattr(audio_tensor, "shape") else str(type(audio_tensor))
        except Exception:
            tensor_shape = "unknown"
        logger.debug(
            "ChunkFormerASR.process_tensor entry | audio_tensor_shape=%s sample_rate=%s chunk_params=(chunk_size=%s,left_ctx=%s,right_ctx=%s)",
            tensor_shape,
            sample_rate,
            chunk_size,
            left_context_size,
            right_context_size,
        )

        # Import dependencies
        import torchaudio.compliance.kaldi as kaldi  # type: ignore
        from chunkformer.model.utils.ctc_utils import get_output_with_timestamps  # type: ignore

        # Validate input
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension
        elif audio_tensor.dim() != 2 or audio_tensor.size(0) != 1:
            raise OMOAudioError(f"Expected audio tensor of shape (1, samples), got {audio_tensor.shape}")

        audio_duration_s = audio_tensor.size(1) / sample_rate
        
        # Setup autocast
        dtype_map = {
            None: None,
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }
        amp_dtype = dtype_map.get(autocast_dtype, None)
        
        # Extract log-mel filterbank features
        xs = kaldi.fbank(
            audio_tensor,
            num_mel_bins=80,
            frame_length=25,
            frame_shift=10,
            dither=0.0,
            energy_floor=0.0,
            sample_frequency=sample_rate,
        ).unsqueeze(0)  # Add batch dimension
        
        # Compute internal parameters
        assert self.model is not None, "Model must be initialized before processing"
        assert self.char_dict is not None, "Character dictionary must be initialized before processing"
        
        subsampling_factor = self.model.encoder.embed.subsampling_factor
        conv_lorder = self.model.encoder.cnn_module_kernel // 2
        
        # Batch duration setup
        max_length_limited_context = total_batch_duration_s
        max_length_limited_context = int((max_length_limited_context // 0.01)) // 2  # in 10ms
        
        multiply_n = max_length_limited_context // chunk_size // subsampling_factor
        truncated_context_size = chunk_size * multiply_n
        
        # Relative right context
        def get_max_input_context(c: int, r: int, n: int) -> int:
            return r + max(c, r) * (n - 1)
        
        rel_right_context_size = get_max_input_context(
            chunk_size, max(right_context_size, conv_lorder), self.model.encoder.num_blocks
        )
        rel_right_context_size = rel_right_context_size * subsampling_factor
        
        # Initialize caches
        offset = torch.zeros(1, dtype=torch.int, device=self.device)
        att_cache = torch.zeros(
            (
                self.model.encoder.num_blocks,
                left_context_size,
                self.model.encoder.attention_heads,
                self.model.encoder._output_size * 2 // self.model.encoder.attention_heads,
            )
        ).to(self.device)
        cnn_cache = torch.zeros(
            (self.model.encoder.num_blocks, self.model.encoder._output_size, conv_lorder)
        ).to(self.device)
        
        # Process in chunks
        hyps: List[torch.Tensor] = []
        ctx = torch.autocast(self.device.type, dtype=amp_dtype, enabled=(amp_dtype is not None))
        
        with torch.inference_mode(), ctx:
            for idx, _ in enumerate(range(0, xs.shape[1], truncated_context_size * subsampling_factor)):
                start = max(truncated_context_size * subsampling_factor * idx, 0)
                end = min(
                    truncated_context_size * subsampling_factor * (idx + 1) + 7,
                    xs.shape[1],
                )
                
                x = xs[:, start : end + rel_right_context_size]
                x_len = torch.tensor([x[0].shape[0]], dtype=torch.int).to(self.device)
                
                (
                    encoder_outs,
                    encoder_lens,
                    _,
                    att_cache,
                    cnn_cache,
                    offset,
                ) = self.model.encoder.forward_parallel_chunk(
                    xs=x,
                    xs_origin_lens=x_len,
                    chunk_size=chunk_size,
                    left_context_size=left_context_size,
                    right_context_size=right_context_size,
                    att_cache=att_cache,
                    cnn_cache=cnn_cache,
                    truncated_context_size=truncated_context_size,
                    offset=offset,
                )
                
                encoder_outs = encoder_outs.reshape(1, -1, encoder_outs.shape[-1])[:, :encoder_lens]
                if (
                    chunk_size * multiply_n * subsampling_factor * idx + rel_right_context_size
                    < xs.shape[1]
                ):
                    encoder_outs = encoder_outs[:, :truncated_context_size]
                
                offset = offset - encoder_lens + encoder_outs.shape[1]
                
                hyp = self.model.encoder.ctc_forward(encoder_outs).squeeze(0)
                hyps.append(hyp)
                
                # Check for debug cache clearing
                debug_empty_cache = os.environ.get("OMOAI_DEBUG_EMPTY_CACHE", "false").lower() == "true"
                if debug_empty_cache and self.device.type == "cuda":
                    torch.cuda.empty_cache()
                
                if (
                    chunk_size * multiply_n * subsampling_factor * idx + rel_right_context_size
                    >= xs.shape[1]
                ):
                    break
        
        # Process results
        if len(hyps) == 0:
            segments = []
            transcript = ""
        else:
            hyps_cat = torch.cat(hyps)
            decode_results = get_output_with_timestamps([hyps_cat], self.char_dict)[0]
            
            segments = [
                ASRSegment(
                    start=float(item["start"]),
                    end=float(item["end"]),
                    text=str(item["decode"]).strip()
                )
                for item in decode_results
            ]
            
            transcript = " ".join(seg.text for seg in segments if seg.text).replace("  ", " ").strip()
        
        # Prepare metadata
        metadata = {
            "model_checkpoint": str(self.model_checkpoint),
            "device": str(self.device),
            "parameters": {
                "chunk_size": chunk_size,
                "left_context_size": left_context_size,
                "right_context_size": right_context_size,
                "autocast_dtype": autocast_dtype or "none",
                "total_batch_duration_s": total_batch_duration_s,
            },
            "audio_info": {
                "sample_rate": sample_rate,
                "duration_seconds": audio_duration_s,
                "tensor_shape": list(audio_tensor.shape),
            }
        }
        
        return ASRResult(
            segments=segments,
            transcript=transcript,
            audio_duration_seconds=audio_duration_s,
            sample_rate=sample_rate,
            metadata=metadata
        )


def run_asr_inference(
    audio_input: torch.Tensor,
    config: Optional[Union[OmoAIConfig, ASRConfig, Dict[str, Any]]] = None,
    model_checkpoint: Optional[Union[Path, str]] = None,
    sample_rate: int = 16000,
    device: Optional[Union[str, torch.device]] = None,
    **kwargs,
) -> ASRResult:
    """
    Run ASR inference on audio tensor with flexible configuration.
    
    Args:
        audio_input: Audio tensor (should be preprocessed to 16kHz mono)
        config: Configuration object or dict (loads default if None)
        model_checkpoint: Model checkpoint path (overrides config)
        sample_rate: Audio sample rate (should be 16kHz)
        device: Target device (auto-detect if None)
        **kwargs: Additional ASR parameters (chunk_size, left_context_size, etc.)
        
    Returns:
        ASRResult with segments and transcript
        
    Raises:
        OMOASRError: If ASR processing fails
        OMOConfigError: If configuration is invalid
        OMOModelError: If model loading fails
        ValueError: If input is invalid
    """
    # Initialize logging
    logger = get_logger("omoai.pipeline.asr")
    perf_logger = get_performance_logger()
    
    # Generate unique ASR ID for tracing
    asr_id = str(uuid.uuid4())
    
    logger.info("Starting ASR inference", extra={
        "asr_id": asr_id,
        "input_type": type(audio_input).__name__,
        "model_checkpoint": str(model_checkpoint) if model_checkpoint else "from_config",
        "sample_rate": sample_rate,
        "device": str(device) if device else "auto",
        "kwargs": kwargs,
    })
    
    timing = {}
    start_time = time.time()
    
    try:
        # Handle configuration
        if config is None:
            from ..config import get_config
            config = get_config()
        
        # Convert dict to OmoAIConfig if needed
        if isinstance(config, dict):
            try:
                config = OmoAIConfig(**config)
            except Exception as e:
                raise OMOConfigError(f"Failed to convert dict to OmoAIConfig: {e}") from e
        
        # Extract ASR configuration
        if isinstance(config, OmoAIConfig):
            asr_config = config.asr
            model_checkpoint = model_checkpoint or config.paths.chunkformer_checkpoint
        elif isinstance(config, ASRConfig):
            asr_config = config
            if not model_checkpoint:
                raise OMOConfigError("model_checkpoint is required when using ASRConfig directly")
        else:
            raise OMOConfigError(f"Unsupported config type: {type(config)}")
        
        if not model_checkpoint:
            raise OMOConfigError("model_checkpoint not found in configuration")
        
        # Prepare ASR parameters
        asr_params = {
            "chunk_size": kwargs.get("chunk_size", asr_config.chunk_size),
            "left_context_size": kwargs.get("left_context_size", asr_config.left_context_size),
            "right_context_size": kwargs.get("right_context_size", asr_config.right_context_size),
            "total_batch_duration_s": kwargs.get("total_batch_duration_s", asr_config.total_batch_duration_s),
            "autocast_dtype": kwargs.get("autocast_dtype", asr_config.autocast_dtype),
            "device_str": kwargs.get("device", asr_config.device),
        }
        
        # Initialize ASR engine
        with performance_context("asr_initialization", logger=logger):
            init_start = time.time()
            engine = ChunkFormerASR(
                model_checkpoint=model_checkpoint,
                device=device or asr_config.device,
            )
            engine.initialize()
            timing["initialization"] = time.time() - init_start
            
            logger.debug("ASR engine initialized", extra={
                "asr_id": asr_id,
                "initialization_time_ms": timing["initialization"] * 1000,
                "model_checkpoint": str(model_checkpoint),
                "device": str(engine.device),
            })
        
        # Process audio
        with performance_context("asr_processing", logger=logger):
            process_start = time.time()
            result = engine.process_tensor(
                audio_tensor=audio_input,
                sample_rate=sample_rate,
                **asr_params,
            )
            timing["processing"] = time.time() - process_start
            
            logger.info("ASR processing completed", extra={
                "asr_id": asr_id,
                "processing_time_ms": timing["processing"] * 1000,
                "segments_count": len(result.segments),
                "transcript_length": len(result.transcript),
                "audio_duration_seconds": result.audio_duration_seconds,
                "sample_rate": result.sample_rate,
                "confidence_avg": sum(seg.confidence or 0 for seg in result.segments) / len(result.segments) if result.segments else 0,
            })
        
        # Calculate total timing
        timing["total"] = time.time() - start_time
        
        # Log performance metrics
        perf_logger.log_operation(
            operation="asr_inference",
            duration_ms=timing["total"] * 1000,
            success=True,
            asr_id=asr_id,
            stages_count=len(timing),
            audio_duration_seconds=result.audio_duration_seconds,
            real_time_factor=timing["total"] / result.audio_duration_seconds if result.audio_duration_seconds > 0 else 0,
        )
        
        logger.info("ASR inference completed successfully", extra={
            "asr_id": asr_id,
            "total_time_ms": timing["total"] * 1000,
            "stages_completed": list(timing.keys()),
            "performance_breakdown": {k: round(v * 1000, 2) for k, v in timing.items()},
            "real_time_factor_total": timing["total"] / result.audio_duration_seconds if result.audio_duration_seconds > 0 else 0,
            "final_transcript_length": len(result.transcript),
            "segments_count": len(result.segments),
        })
        
        return result
        
    except Exception as e:
        # Enhanced error reporting with timing info
        error_timing = time.time() - start_time
        
        # Log detailed error information
        log_error(
            message=f"ASR failed after {error_timing:.2f}s",
            error=e,
            error_type="ASR_FAILURE",
            error_code="ASR_001",
            remediation="Check input validity, configuration, and model availability",
            logger=logger,
            asr_id=asr_id,
            stages_completed=list(timing.keys()),
            error_timing_seconds=error_timing,
        )
        
        # Record failed operation
        perf_logger.log_operation(
            operation="asr_inference",
            duration_ms=error_timing * 1000,
            success=False,
            error_type="ASR_FAILURE",
            asr_id=asr_id,
            stages_completed=list(timing.keys()),
        )
        
        raise OMOASRError(
            f"ASR failed after {error_timing:.2f}s: {e}. "
            f"Completed stages: {list(timing.keys())}"
        ) from e



