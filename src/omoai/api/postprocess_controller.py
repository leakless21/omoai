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
        # Prefer punctuation settings from top-level 'punctuation' section; fall back to 'postprocess' if missing
        punct_cfg = cfg.get("punctuation", {}) or {}
        base_punct_prompt = punct_cfg.get(
            "system_prompt",
            cfg.get("postprocess", {}).get(
                "punctuation_system_prompt",
                "Bạn là một trợ lý AI thông minh. Nhiệm vụ của bạn là thêm dấu câu phù hợp vào văn bản tiếng Việt. "
                "Hãy thêm dấu chấm, dấu phẩy, dấu chấm hỏi, dấu chấm than, dấu hai chấm, dấu chấm phẩy, "
                "và các dấu ngoặc đơn hoặc kép nếu cần thiết. Đảm bảo văn bản trở nên dễ đọc và có ý nghĩa."
            ),
        )
        # Use the configured system prompt as-is.
        # The default prompt in config.yaml contains the anti-deletion policy,
        # so do not append or manage a separate prevent_deletions_prompt flag here.

        self.config = {
            "model_id": cfg.get("postprocess", {}).get("model_id", "microsoft/DialoGPT-medium"),
            "quantization": cfg.get("postprocess", {}).get("quantization", None),
            "max_model_len": cfg.get("postprocess", {}).get("max_model_len", 2048),
            "gpu_memory_utilization": cfg.get("postprocess", {}).get("gpu_memory_utilization", 0.9),
            "max_num_seqs": cfg.get("postprocess", {}).get("max_num_seqs", 1),
            "max_num_batched_tokens": cfg.get("postprocess", {}).get("max_num_batched_tokens", None),
            "temperature": cfg.get("postprocess", {}).get("temperature", 0.0),
            "trust_remote_code": cfg.get("postprocess", {}).get("trust_remote_code", False),
            "punctuation_system_prompt": base_punct_prompt,
            "punctuation_keep_nonempty_segments": bool(punct_cfg.get("keep_nonempty_segments", False)),
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

        # Import functions from scripts/post.py with a project-root sys.path for static analyzers and runtime
        sys.path.insert(0, str(Path(__file__).parents[3]))
        from scripts.post import (
            punctuate_text_with_splitting,
            summarize_long_text_map_reduce,
            join_punctuated_segments,
            _segmentwise_punctuate_segments,
            _safe_distribute_punct_to_segments,
        )

        # Add punctuation to transcript using enhanced alignment
        # First, punctuate individual segments (segment-wise for throughput)
        punctuated_segments = _segmentwise_punctuate_segments(
            llm=self.llm,
            system_prompt=self.config["punctuation_system_prompt"],
            max_model_len=self.config["max_model_len"],
            segments=segments,
            temperature=self.config["temperature"],
        )

        # Join per-segment outputs into a single punctuated string
        punctuated_text = join_punctuated_segments(punctuated_segments)

        # Distribute the punctuated text back to the original segments with alignment
        final_segments = _safe_distribute_punct_to_segments(
            punctuated_text,
            segments,
            keep_nonempty_segments=self.config.get("punctuation_keep_nonempty_segments", False),
        )

        # Produce final coherent transcript from aligned segments
        transcript_punct = join_punctuated_segments(final_segments)

        # Generate summary
        summary = summarize_long_text_map_reduce(
            llm=self.llm,
            text=transcript_punct or transcript_raw,
            system_prompt=self.config["summary_system_prompt"],
            temperature=self.config["temperature"],
            max_model_len=self.config["max_model_len"],
        )

        return {
            "transcript_punct": transcript_punct,
            "summary": summary
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
        - include_quality_metrics: Include quality metrics in response
        - include_diffs: Include human-readable diffs in response
        """
        # Process with optional quality metrics and diffs
        # Convert OutputFormatParams to dict if provided
        output_params_dict = None
        if output_params:
            output_params_dict = {
                "include_quality_metrics": output_params.include_quality_metrics,
                "include_diffs": output_params.include_diffs,
                "include": output_params.include,
                "ts": output_params.ts,
                "summary": output_params.summary,
                "summary_bullets_max": output_params.summary_bullets_max,
                "summary_lang": output_params.summary_lang
            }
        
        result = await postprocess_service(data, output_params_dict)
        
        return result