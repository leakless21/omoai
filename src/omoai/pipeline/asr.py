"""
In-memory ASR inference for OMOAI using ChunkFormer.

This module provides efficient ASR processing that works with tensors and
returns structured data without intermediate file I/O.
"""
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

import torch
import numpy as np
from pydub import AudioSegment  # type: ignore
import logging

from ..config import OmoAIConfig, ASRConfig

# Module logger
logger = logging.getLogger("omoai.pipeline.asr")


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
        chunkformer_dir: Optional[Union[Path, str]] = None,
    ):
        """
        Initialize ChunkFormer ASR engine.
        
        Args:
            model_checkpoint: Path to ChunkFormer checkpoint
            device: Target device (auto-detect if None)
            chunkformer_dir: Path to ChunkFormer source (for imports)
        """
        self.model_checkpoint = Path(model_checkpoint)
        self.device = self._resolve_device(device)
        self.chunkformer_dir = Path(chunkformer_dir) if chunkformer_dir else None
        
        # Model components (loaded lazily)
        self.model = None
        self.char_dict = None
        self._is_initialized = False
        
    def _resolve_device(self, device: Optional[Union[str, torch.device]]) -> torch.device:
        """Resolve device with auto-detection."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(device)
    
    def _ensure_chunkformer_path(self) -> None:
        """Add ChunkFormer to Python path if needed."""
        if self.chunkformer_dir and str(self.chunkformer_dir) not in sys.path:
            sys.path.insert(0, str(self.chunkformer_dir))
    
    def initialize(self) -> None:
        """Initialize the model (lazy loading)."""
        if self._is_initialized:
            return
        
        self._ensure_chunkformer_path()
        
        try:
            # Import ChunkFormer components
            from omoai.chunkformer import decode as cfdecode  # type: ignore
            
            # Initialize model
            self.model, self.char_dict = cfdecode.init(str(self.model_checkpoint), self.device)
            self._is_initialized = True
            
        except ImportError as e:
            raise ImportError(
                f"Failed to import ChunkFormer components. "
                f"Ensure chunkformer_dir is correct: {self.chunkformer_dir}. Error: {e}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ChunkFormer model: {e}")
    
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
        from omoai.chunkformer.model.utils.ctc_utils import get_output_with_timestamps  # type: ignore

        # Validate input
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension
        elif audio_tensor.dim() != 2 or audio_tensor.size(0) != 1:
            raise ValueError(f"Expected audio tensor of shape (1, samples), got {audio_tensor.shape}")

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
    audio_input: Union[torch.Tensor, np.ndarray, bytes, Path, str],
    config: Optional[Union[OmoAIConfig, ASRConfig, Dict[str, Any]]] = None,
    model_checkpoint: Optional[Union[Path, str]] = None,
    device: Optional[Union[str, torch.device]] = None,
    **kwargs
) -> ASRResult:
    """
    Run ASR inference on audio input with flexible configuration.
    
    Args:
        audio_input: Audio data (tensor, array, bytes, or file path)
        config: Configuration object or dict
        model_checkpoint: Model path (overrides config)
        device: Target device (overrides config)
        **kwargs: Additional ASR parameters
        
    Returns:
        ASRResult with segments and transcript
        
    Raises:
        ValueError: If input is invalid or config is missing required fields
        RuntimeError: If ASR processing fails
    """
    # Handle configuration
    if config is None:
        from ..config import get_config
        config = get_config()
    
    if isinstance(config, dict):
        # Extract ASR config from dict
        asr_config = config.get("asr", {})
        paths_config = config.get("paths", {})
        model_checkpoint = model_checkpoint or paths_config.get("chunkformer_checkpoint")
        chunkformer_dir = paths_config.get("chunkformer_dir")
    elif isinstance(config, OmoAIConfig):
        asr_config = config.asr
        model_checkpoint = model_checkpoint or config.paths.chunkformer_checkpoint
        chunkformer_dir = config.paths.chunkformer_dir
    elif isinstance(config, ASRConfig):
        asr_config = config
        if not model_checkpoint:
            raise ValueError("model_checkpoint is required when using ASRConfig directly")
        chunkformer_dir = None
    else:
        raise ValueError(f"Unsupported config type: {type(config)}")
    
    if not model_checkpoint:
        raise ValueError("model_checkpoint not found in configuration")
    
    # Prepare ASR parameters
    asr_params = {
        "chunk_size": getattr(asr_config, "chunk_size", 64),
        "left_context_size": getattr(asr_config, "left_context_size", 128),
        "right_context_size": getattr(asr_config, "right_context_size", 128),
        "total_batch_duration_s": getattr(asr_config, "total_batch_duration_s", 1800),
        "autocast_dtype": getattr(asr_config, "autocast_dtype", "fp16"),
    }

    # Diagnostic: check if caller passed sample_rate via kwargs which would be merged into asr_params
    sample_rate_override = kwargs.get("sample_rate", None)
    logger.debug(
        "Preparing ASR params | initial_keys=%s kwargs_keys=%s sample_rate_override=%s",
        list(asr_params.keys()),
        list(kwargs.keys()),
        sample_rate_override,
    )

    # Merge overrides, then ensure sample_rate is not present in asr_params to avoid duplicate keyword error
    asr_params.update(kwargs)  # Allow parameter overrides
    removed_sample_rate = asr_params.pop("sample_rate", None)
    if removed_sample_rate is not None:
        logger.debug("Removed duplicate 'sample_rate' from asr_params (value=%s) to avoid TypeError", removed_sample_rate)
    
    # Initialize ASR engine
    asr_device = device or getattr(asr_config, "device", "auto")
    engine = ChunkFormerASR(
        model_checkpoint=model_checkpoint,
        device=asr_device,
        chunkformer_dir=chunkformer_dir,
    )
    
    # Process different input types
    if isinstance(audio_input, torch.Tensor):
        # Direct tensor input
        sample_rate = kwargs.pop("sample_rate", 16000)  # Remove from kwargs to avoid duplicate
        return engine.process_tensor(audio_input, sample_rate=sample_rate, **asr_params)
        
    elif isinstance(audio_input, np.ndarray):
        # Convert numpy array to tensor
        audio_tensor = torch.from_numpy(audio_input.astype(np.float32))
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        sample_rate = kwargs.pop("sample_rate", 16000)  # Remove from kwargs to avoid duplicate
        return engine.process_tensor(audio_tensor, sample_rate=sample_rate, **asr_params)
        
    else:
        # File path or bytes - need preprocessing
        from .preprocess import preprocess_audio_to_tensor
        
        audio_tensor, sample_rate = preprocess_audio_to_tensor(
            audio_input, 
            target_sample_rate=16000,
            return_sample_rate=True
        )
        return engine.process_tensor(audio_tensor, sample_rate=sample_rate, **asr_params)


# Legacy compatibility function for scripts
def run_asr_from_config(
    audio_path: Path,
    model_checkpoint: Path,
    out_path: Path,
    total_batch_duration: int,
    chunk_size: int,
    left_context_size: int,
    right_context_size: int,
    device_str: str,
    autocast_dtype: Optional[str],
    chunkformer_dir: Path,
) -> None:
    """
    Legacy compatibility function that mimics the original script interface.
    
    This function processes audio and saves results to JSON, maintaining
    compatibility with existing script-based workflows.
    """
    import json
    
    # Convert parameters to new format
    asr_config = {
        "chunk_size": chunk_size,
        "left_context_size": left_context_size, 
        "right_context_size": right_context_size,
        "total_batch_duration_s": total_batch_duration,
        "autocast_dtype": autocast_dtype,
        "device": device_str,
    }
    
    # Run inference
    result = run_asr_inference(
        audio_input=audio_path,
        config={"asr": asr_config, "paths": {"chunkformer_checkpoint": model_checkpoint, "chunkformer_dir": chunkformer_dir}},
    )
    
    # Convert to legacy format
    legacy_output = {
        "audio": {
            "sr": result.sample_rate,
            "path": str(audio_path.resolve()),
            "duration_s": result.audio_duration_seconds,
        },
        "segments": [
            {
                "start": seg.start,
                "end": seg.end,
                "text_raw": seg.text,
            }
            for seg in result.segments
        ],
        "transcript_raw": result.transcript,
        "metadata": result.metadata,
    }
    
    # Save to file
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(legacy_output, f, ensure_ascii=False, indent=2)
