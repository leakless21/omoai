import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from litestar import Controller, post
from litestar.datastructures import State

from omoai.api.models import PostprocessRequest, PostprocessResponse, OutputFormatParams


class PostprocessModel:
    """Singleton class to hold the post-processing model and configuration."""

    _instance = None
    _is_initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PostprocessModel, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._is_initialized:
            self.llm = None
            self.config = {}
            self._is_initialized = True

    def initialize(self, config_path: Path = Path("/home/cetech/omoai/config.yaml")):
        """Initialize the post-processing model with configuration."""
        # Load configuration
        try:
            import yaml  # type: ignore
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {}

        # Store configuration
        self.config = {
            "model_id": cfg.get("postprocess", {}).get("model_id", "microsoft/DialoGPT-medium"),
            "quantization": cfg.get("postprocess", {}).get("quantization", None),
            "max_model_len": cfg.get("postprocess", {}).get("max_model_len", 2048),
            "gpu_memory_utilization": cfg.get("postprocess", {}).get("gpu_memory_utilization", 0.9),
            "max_num_seqs": cfg.get("postprocess", {}).get("max_num_seqs", 1),
            "max_num_batched_tokens": cfg.get("postprocess", {}).get("max_num_batched_tokens", None),
            "temperature": cfg.get("postprocess", {}).get("temperature", 0.0),
            "trust_remote_code": cfg.get("postprocess", {}).get("trust_remote_code", False),
            "punctuation_system_prompt": cfg.get("postprocess", {}).get("punctuation_system_prompt",
                "Bạn là một trợ lý AI thông minh. Nhiệm vụ của bạn là thêm dấu câu phù hợp vào văn bản tiếng Việt. "
                "Hãy thêm dấu chấm, dấu phẩy, dấu chấm hỏi, dấu chấm than, dấu hai chấm, dấu chấm phẩy, "
                "và các dấu ngoặc đơn hoặc kép nếu cần thiết. Đảm bảo văn bản trở nên dễ đọc và có ý nghĩa."),
            "summary_system_prompt": cfg.get("postprocess", {}).get("summary_system_prompt",
                "Bạn là một trợ lý AI thông minh. Nhiệm vụ của bạn là tóm tắt văn bản tiếng Việt một cách ngắn gọn và chính xác. "
                "Hãy tạo ra một bản tóm tắt có cấu trúc rõ ràng với các điểm chính và một đoạn tóm tắt ngắn gọn."),
        }

        # Initialize vLLM model
        try:
            from vllm import LLM, SamplingParams  # type: ignore

            self.llm = LLM(
                model=self.config["model_id"],
                quantization=self.config["quantization"],
                max_model_len=self.config["max_model_len"],
                gpu_memory_utilization=self.config["gpu_memory_utilization"],
                max_num_seqs=self.config["max_num_seqs"],
                max_num_batched_tokens=self.config["max_num_batched_tokens"],
                trust_remote_code=self.config["trust_remote_code"],
            )
            print(f"Post-processing model initialized: {self.config['model_id']}")
        except Exception as e:
            print(f"Failed to initialize post-processing model: {e}")
            raise

    def process_asr_output(self, asr_output: Dict[str, Any]) -> Dict[str, Any]:
        """Process ASR output to add punctuation and generate summary."""
        if not self.llm:
            raise RuntimeError("Post-processing model not initialized")

        # Extract data from ASR output
        segments = asr_output.get("segments", [])
        transcript_raw = asr_output.get("transcript_raw", "")

        if not transcript_raw:
            return {
                "transcript_punct": "",
                "summary": {"bullets": [], "abstract": ""}
            }

        # Import functions from scripts/post.py
        sys.path.insert(0, str(Path("/home/cetech/omoai/scripts")))
        try:
            from post import (  # type: ignore
                punctuate_text_with_splitting,
                summarize_long_text_map_reduce,
                join_punctuated_segments,
                _segmentwise_punctuate_segments,
                _safe_distribute_punct_to_segments,
            )
        except ImportError:
            # Fallback implementation if import fails
            return self._fallback_process(transcript_raw, segments)

        # Add punctuation to transcript
        try:
            # First, punctuate individual segments
            punctuated_segments = _segmentwise_punctuate_segments(
                llm=self.llm,
                system_prompt=self.config["punctuation_system_prompt"],
                max_model_len=self.config["max_model_len"],
                segments=segments,
                temperature=self.config["temperature"],
            )

            # Join punctuated segments into coherent transcript
            transcript_punct = join_punctuated_segments(punctuated_segments)

        except Exception as e:
            print(f"Punctuation failed, using fallback: {e}")
            transcript_punct = self._add_basic_punctuation(transcript_raw)

        # Generate summary
        try:
            summary = summarize_long_text_map_reduce(
                llm=self.llm,
                text=transcript_punct or transcript_raw,
                system_prompt=self.config["summary_system_prompt"],
                temperature=self.config["temperature"],
                max_model_len=self.config["max_model_len"],
            )
        except Exception as e:
            print(f"Summarization failed, using fallback: {e}")
            summary = self._fallback_summarize(transcript_punct or transcript_raw)

        return {
            "transcript_punct": transcript_punct,
            "summary": summary
        }

    def _fallback_process(self, transcript_raw: str, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fallback processing when script import fails."""
        transcript_punct = self._add_basic_punctuation(transcript_raw)
        summary = self._fallback_summarize(transcript_punct)
        return {
            "transcript_punct": transcript_punct,
            "summary": summary
        }

    def _add_basic_punctuation(self, text: str) -> str:
        """Add basic punctuation as fallback."""
        if not text:
            return ""

        # Capitalize first letter
        text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()

        # Add period at end if no punctuation
        if not text[-1] in '.!?':
            text += '.'

        return text

    def _fallback_summarize(self, text: str) -> Dict[str, Any]:
        """Simple fallback summarization."""
        if not text:
            return {"bullets": [], "abstract": ""}

        # Simple word-based truncation for abstract
        words = text.split()
        abstract = " ".join(words[:50]) + "..." if len(words) > 50 else text

        # Create simple bullet points
        sentences = text.split('.')
        bullets = [s.strip() + '.' for s in sentences[:3] if s.strip()]

        return {
            "bullets": bullets,
            "abstract": abstract
        }


from omoai.api.services_enhanced import postprocess_service


class PostprocessController(Controller):
    path = "/postprocess"

    @post()
    async def postprocess(
        self,
        data: PostprocessRequest,
        state: State,
        output_params: Optional[OutputFormatParams] = None
    ) -> PostprocessResponse:
        """Process post-processing request with optional output formatting.

        Query Parameters (optional):
        - include: What to include (transcript_raw, transcript_punct, segments)
        - ts: Timestamp format (none, s, ms, clock)
        - summary: Summary type (bullets, abstract, both, none)
        - summary_bullets_max: Maximum number of bullet points
        - summary_lang: Summary language
        """
        # For now, just pass through - can be enhanced later to support output formatting
        return postprocess_service(data)