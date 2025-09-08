"""
In-memory postprocessing for OMOAI transcripts.

This module provides efficient punctuation and summarization processing
that works with structured data without intermediate file I/O.
"""
import gc
import os
import sys
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .asr import ASRResult, ASRSegment
from ..config import OmoAIConfig, PunctuationConfig, SummarizationConfig
from .exceptions import OMOPostprocessError, OMOConfigError, OMOModelError


@dataclass
class SummaryResult:
    """Represents a text summary with bullets and abstract."""
    bullets: List[str]
    abstract: str
    metadata: Dict[str, Any]


@dataclass
class PostprocessResult:
    """Complete postprocessing result."""
    segments: List[ASRSegment]
    transcript_punctuated: str
    summary: SummaryResult
    metadata: Dict[str, Any]


class VLLMProcessor:
    """vLLM-based processor for punctuation and summarization."""
    
    def __init__(
        self,
        model_id: str,
        quantization: Optional[str] = "auto",
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.85,
        max_num_seqs: int = 1,
        max_num_batched_tokens: Optional[int] = None,
        trust_remote_code: bool = False,
        device: Optional[str] = None,
    ):
        """Initialize vLLM processor with configuration."""
        self.model_id = model_id
        self.config = {
            "quantization": quantization,
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_num_seqs": max_num_seqs,
            "max_num_batched_tokens": max_num_batched_tokens,
            "trust_remote_code": trust_remote_code,
        }
        
        # Lazy-loaded components
        self.llm = None
        self._is_initialized = False
        
    def initialize(self) -> None:
        """Initialize vLLM engine (lazy loading)."""
        if self._is_initialized:
            return
            
        try:
            from vllm import LLM  # type: ignore
            
            # Build vLLM configuration
            kwargs = {
                "model": self.model_id,
                "max_model_len": self.config["max_model_len"],
                "gpu_memory_utilization": self.config["gpu_memory_utilization"],
                "trust_remote_code": self.config["trust_remote_code"],
                "max_num_seqs": self.config["max_num_seqs"],
                "enforce_eager": True,  # Performance optimization
            }
            
            if self.config["max_num_batched_tokens"]:
                kwargs["max_num_batched_tokens"] = self.config["max_num_batched_tokens"]
            
            # Handle quantization
            quant = self.config["quantization"]
            if quant and quant not in ("auto", "infer", "compressed-tensors", "model"):
                kwargs["quantization"] = quant
            
            # Initialize LLM
            try:
                self.llm = LLM(**kwargs)
            except Exception as e:
                # Retry without quantization if it fails
                if "quantization" in kwargs and "Quantization method specified in the model config" in str(e):
                    del kwargs["quantization"]
                    self.llm = LLM(**kwargs)
                else:
                    raise
                    
            self._is_initialized = True
            
        except ImportError as e:
            raise OMOModelError(f"vLLM not available for postprocessing: {e}") from e
        except Exception as e:
            raise OMOPostprocessError(f"Failed to initialize vLLM processor: {e}") from e
    
    def generate_text(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        """Generate text using chat-style prompting."""
        self.initialize()
        
        from vllm import SamplingParams  # type: ignore
        
        # Import postprocessing utilities
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "scripts"))
        try:
            from post import apply_chat_template, generate_chat  # type: ignore
            return generate_chat(self.llm, messages, temperature, max_tokens)
        except ImportError:
            # Fallback implementation
            return self._fallback_generate(messages, temperature, max_tokens)
    
    def _fallback_generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Fallback text generation without post.py dependencies."""
        from vllm import SamplingParams  # type: ignore
        
        # Simple chat template
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "").strip().lower()
            content = msg.get("content", "").strip()
            if not content:
                continue
            if role == "system":
                prompt_parts.append(f"[SYSTEM]\n{content}")
            elif role == "user":
                prompt_parts.append(f"[USER]\n{content}")
            elif role == "assistant":
                prompt_parts.append(f"[ASSISTANT]\n{content}")
            else:
                prompt_parts.append(content)
        
        prompt = "\n\n".join(prompt_parts) + "\n\n[ASSISTANT]\n"
        
        params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        outputs = self.llm.generate([prompt], params)
        
        return outputs[0].outputs[0].text if outputs[0].outputs else ""


def punctuate_transcript(
    asr_result: ASRResult,
    config: Union[PunctuationConfig, Dict[str, Any]],
    processor: Optional[VLLMProcessor] = None,
) -> List[ASRSegment]:
    """
    Add punctuation to ASR transcript segments using enhanced alignment algorithm.
    
    Args:
        asr_result: ASR processing result
        config: Punctuation configuration
        processor: vLLM processor (created if None)
        
    Returns:
        List of segments with punctuation added
    """
    if not asr_result.segments or not asr_result.transcript:
        return asr_result.segments
    
    # Handle configuration
    # Convert dict to PunctuationConfig if needed
    if isinstance(config, dict):
        from ..config import PunctuationConfig
        try:
            # Extract relevant fields for PunctuationConfig
            punct_dict = {
                "llm": config.get("llm", {}),
                "system_prompt": config.get("system_prompt", ""),
                "sampling": config.get("sampling", {}),
                "preserve_original_words": config.get("preserve_original_words", True),
            }
            config = PunctuationConfig(**punct_dict)
        except Exception as e:
            raise OMOConfigError(f"Failed to convert dict to PunctuationConfig: {e}") from e
        temperature = config.get("sampling", {}).get("temperature", 0.0)
        preserve_words = config.get("preserve_original_words", True)
    else:
        llm_config = config.llm.__dict__
        system_prompt = config.system_prompt
        temperature = config.sampling.temperature
        preserve_words = config.preserve_original_words
    
    # Initialize processor if needed
    if processor is None:
        processor = VLLMProcessor(**llm_config.model_dump())
    
    # Import enhanced punctuation functions from scripts/post.py
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "scripts"))
    
    try:
        from post import (
            join_punctuated_segments,
            _segmentwise_punctuate_segments,
            _safe_distribute_punct_to_segments,
        )
        
        # Convert ASR segments to the format expected by scripts/post.py
        segments_for_post = [
            {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "confidence": seg.confidence
            }
            for seg in asr_result.segments
        ]
        
        # Use enhanced punctuation with alignment
        try:
            # First, punctuate individual segments using the enhanced method
            punctuated_segments = _segmentwise_punctuate_segments(
                llm=processor.llm,
                system_prompt=system_prompt,
                max_model_len=processor.config.get("max_model_len", 8192),
                segments=segments_for_post,
                temperature=temperature,
            )
            
            # Join punctuated segments using enhanced alignment algorithm
            # This handles word-level and character-level alignment
            final_segments = _safe_distribute_punct_to_segments(punctuated_segments)
            
            # Convert back to ASRSegment format
            result_segments = []
            for seg in final_segments:
                result_segments.append(ASRSegment(
                    start=seg.get("start", 0.0),
                    end=seg.get("end", 0.0),
                    text=seg.get("text_punct", seg.get("text", "")),
                    confidence=seg.get("confidence", 0.0)
                ))
            
            return result_segments
            
        except Exception as e:
            print(f"Warning: Enhanced punctuation failed, using fallback: {e}")
            # Fallback to simple segment-wise punctuation
            return _fallback_punctuate_segments(asr_result, processor, system_prompt, temperature)
            
    except ImportError:
        print("Warning: Enhanced punctuation functions not available, using fallback")
        # Fallback to simple segment-wise punctuation
        return _fallback_punctuate_segments(asr_result, processor, system_prompt, temperature)


def _fallback_punctuate_segments(
    asr_result: ASRResult,
    processor: VLLMProcessor,
    system_prompt: str,
    temperature: float,
) -> List[ASRSegment]:
    """
    Fallback punctuation method when enhanced functions are not available.
    
    Args:
        asr_result: ASR processing result
        processor: vLLM processor
        system_prompt: System prompt for punctuation
        temperature: Sampling temperature
        
    Returns:
        List of segments with punctuation added
    """
    punctuated_segments = []
    
    for segment in asr_result.segments:
        if not segment.text.strip():
            punctuated_segments.append(segment)
            continue
        
        # Generate punctuated text
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": segment.text},
        ]
        
        try:
            punctuated_text = processor.generate_text(
                messages,
                temperature=temperature,
                max_tokens=len(segment.text.split()) + 20  # Allow for punctuation
            )
            
            # Clean up the response
            punctuated_text = punctuated_text.strip()
            
            # Create new segment with punctuation
            punctuated_segments.append(ASRSegment(
                start=segment.start,
                end=segment.end,
                text=punctuated_text,
                confidence=segment.confidence,
            ))
            
        except Exception as e:
            # Fallback to original text on error
            print(f"Warning: Punctuation failed for segment, using original: {e}")
            punctuated_segments.append(segment)
    
    return punctuated_segments


def summarize_text(
    text: str,
    config: Union[SummarizationConfig, Dict[str, Any]],
    processor: Optional[VLLMProcessor] = None,
) -> SummaryResult:
    """
    Generate summary of text with bullets and abstract.
    
    Args:
        text: Input text to summarize
        config: Summarization configuration
        processor: vLLM processor (created if None)
        
    Returns:
        SummaryResult with bullets and abstract
    """
    if not text.strip():
        return SummaryResult(
            bullets=[],
            abstract="",
            metadata={"input_length": 0, "method": "empty"}
        )
    
    # Handle configuration
    from ..config import SummarizationConfig
    
    if isinstance(config, dict):
        try:
            # Extract relevant fields for SummarizationConfig
            summ_dict = {
                "llm": config.get("llm", {}),
                "system_prompt": config.get("system_prompt", ""),
                "sampling": config.get("sampling", {"temperature": 0.7}),
                "map_reduce": config.get("map_reduce", False),
            }
            config = SummarizationConfig(**summ_dict)
        except Exception as e:
            raise OMOConfigError(f"Failed to convert dict to SummarizationConfig: {e}") from e
    
    # Extract configuration components
    if isinstance(config, SummarizationConfig):
        llm_config = config.llm
        system_prompt = config.system_prompt
        temperature = config.sampling.temperature
        use_map_reduce = config.map_reduce
    else:
        raise OMOConfigError(f"Unsupported config type for summarization: {type(config)}")
    
    # Initialize processor if needed
    if processor is None:
        processor = VLLMProcessor(**llm_config.model_dump())
    
    try:
        # For now, implement simple single-pass summarization
        # Map-reduce can be added later for very long texts
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Hãy tóm tắt đoạn văn sau:\n\n{text}"},
        ]
        
        summary_text = processor.generate_text(
            messages,
            temperature=temperature,
            max_tokens=800
        )
        
        # Parse JSON response
        import json
        try:
            parsed = json.loads(summary_text.strip())
            bullets = parsed.get("bullets", [])
            abstract = parsed.get("abstract", "")
        except json.JSONDecodeError:
            # Fallback parsing
            lines = summary_text.strip().split('\n')
            bullets = [line.strip('- ') for line in lines if line.strip().startswith('-')]
            abstract = summary_text.strip() if not bullets else ""
        
        return SummaryResult(
            bullets=bullets[:7],  # Limit to 7 bullets
            abstract=abstract[:1000],  # Limit abstract length
            metadata={
                "input_length": len(text),
                "method": "single_pass",
                "model": processor.model_id,
            }
        )
        
    except Exception as e:
        # Fallback summary
        return SummaryResult(
            bullets=["Processing failed"],
            abstract=f"Summary generation failed: {e}",
            metadata={"input_length": len(text), "method": "fallback", "error": str(e)}
        )


def postprocess_transcript(
    asr_result: ASRResult,
    config: Optional[Union[OmoAIConfig, Dict[str, Any]]] = None,
    punctuation_config: Optional[Union[PunctuationConfig, Dict[str, Any]]] = None,
    summarization_config: Optional[Union[SummarizationConfig, Dict[str, Any]]] = None,
) -> PostprocessResult:
    """
    Complete postprocessing: punctuation + summarization.
    
    Args:
        asr_result: ASR processing result
        config: Complete configuration object
        punctuation_config: Punctuation-specific config (overrides config)
        summarization_config: Summarization-specific config (overrides config)
        
    Returns:
        PostprocessResult with punctuated transcript and summary
    """
    # Handle configuration
    if config is None:
        from ..config import get_config
        config = get_config()
    
    # Convert dict to OmoAIConfig if needed
    if isinstance(config, dict):
        from ..config import OmoAIConfig
        try:
            config = OmoAIConfig(**config)
        except Exception as e:
            raise ValueError(f"Failed to convert dict to OmoAIConfig: {e}")
    
    # Extract configuration components
    if isinstance(config, OmoAIConfig):
        punct_config = punctuation_config or config.punctuation
        summ_config = summarization_config or config.summarization
    else:
        if not punctuation_config or not summarization_config:
            raise ValueError("punctuation_config and summarization_config required when config is not OmoAIConfig")
        punct_config = punctuation_config
        summ_config = summarization_config
    
    # Check if models can be reused
    # Handle the case where llm might be an object or a dict
    if hasattr(punct_config.llm, '__dict__'):
        # llm is an object, get its __dict__
        punct_llm_config = punct_config.llm.__dict__
    else:
        # llm is already a dict, use it directly
        punct_llm_config = punct_config.llm
        
    if hasattr(summ_config.llm, '__dict__'):
        # llm is an object, get its __dict__
        summ_llm_config = summ_config.llm.__dict__
    else:
        # llm is already a dict, use it directly
        summ_llm_config = summ_config.llm
    
    can_reuse_model = (
            punct_llm_config["model_id"] == summ_llm_config["model_id"] and
            punct_llm_config["quantization"] == summ_llm_config["quantization"] and
            punct_llm_config["max_model_len"] == summ_llm_config["max_model_len"]
        )
    
    processor = None
    
    try:
        # Step 1: Punctuation
        if can_reuse_model:
            processor = VLLMProcessor(**punct_llm_config)
        
        punctuated_segments = punctuate_transcript(asr_result, punct_config, processor)
        punctuated_transcript = " ".join(seg.text for seg in punctuated_segments if seg.text).strip()
        
        # Step 2: Summarization
        if can_reuse_model and processor:
            summary = summarize_text(punctuated_transcript, summ_config, processor)
        else:
            # Clean up punctuation processor if different model needed
            if processor:
                del processor
                # Only clear cache if debug flag is set
                debug_empty_cache = os.environ.get("OMOAI_DEBUG_EMPTY_CACHE", "false").lower() == "true"
                if debug_empty_cache:
                    with suppress(Exception):
                        import torch
                        torch.cuda.empty_cache()
                gc.collect()
            
            summary = summarize_text(punctuated_transcript, summ_config)
        
        # Prepare metadata
        metadata = {
            "processing_info": {
                "model_reuse": can_reuse_model,
                "punctuation_model": punct_llm_config.get("model_id"),
                "summarization_model": summ_llm_config.get("model_id"),
            },
            "quality_metrics": {
                "segments_processed": len(punctuated_segments),
                "transcript_length": len(punctuated_transcript),
                "summary_bullets": len(summary.bullets),
                "has_abstract": bool(summary.abstract),
            },
            "original_asr": asr_result.metadata,
        }
        
        return PostprocessResult(
            segments=punctuated_segments,
            transcript_punctuated=punctuated_transcript,
            summary=summary,
            metadata=metadata,
        )
        
    finally:
        # Clean up processor
        if processor:
            del processor
            gc.collect()


def _parse_time_to_seconds(value):
    """Parse a timestamp value into seconds."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        value = value.strip()
        try:
            return float(value)
        except ValueError:
            pass
        # Try HH:MM:SS:mmm format (with milliseconds)
        parts = value.split(":")
        if len(parts) == 4:
            h, m, s, ms = parts
            return int(h) * 3600 + int(m) * 60 + float(s) + float(ms) / 1000
        elif len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        elif len(parts) == 2:
            m, s = parts
            return int(m) * 60 + float(s)
    return None


def _dedup_overlap(prev, nxt, max_tokens=8):
    """Remove duplicated token overlap between strings."""
    pt = prev.strip().split()
    nt = nxt.strip().split()
    max_k = min(max_tokens, len(pt), len(nt))
    for k in range(max_k, 0, -1):
        if pt[-k:] == nt[:k]:
            return " ".join(nt[k:])
    return nxt


def join_punctuated_segments(segments, join_separator=" ", paragraph_gap_seconds=3.0):
    """Join punctuated segments into coherent text."""
    if not segments:
        return ""
    
    parts = []
    last_end = None
    
    for seg in segments:
        text = seg.get("text_punct") or seg.get("text") or ""
        if not text.strip():
            continue
            
        start = _parse_time_to_seconds(seg.get("start"))
        
        # Add paragraph break if gap is large
        if (last_end is not None and start is not None and 
            start - last_end >= paragraph_gap_seconds and parts):
            parts.append("\n\n")
        
        if parts and not parts[-1].endswith("\n\n"):
            parts.append(join_separator)
        parts.append(text.strip())
        
        last_end = _parse_time_to_seconds(seg.get("end"))
    
    return "".join(parts).strip()


