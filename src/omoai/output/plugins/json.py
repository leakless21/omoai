"""
JSON formatter for transcripts and summaries.
"""

import json
from typing import Any, Dict, List, Optional

from ..formatter import OutputFormatter, register_formatter, extract_segment_data
from ...config.schemas import TranscriptOutputConfig, SummaryOutputConfig


@register_formatter
class JsonFormatter(OutputFormatter):
    """JSON formatter for structured data output."""
    
    format_name = "json"
    file_extension = ".json"
    
    def format_transcript(
        self,
        segments: List[Dict[str, Any]],
        transcript_raw: str,
        transcript_punct: str,
        config: TranscriptOutputConfig,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Format transcript as JSON."""
        result = {}
        
        if config.include_raw:
            result["transcript_raw"] = transcript_raw
        
        if config.include_punct:
            result["transcript_punct"] = transcript_punct
        
        if config.include_segments:
            result["segments"] = extract_segment_data(segments)
        
        if metadata:
            result["metadata"] = metadata
        
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    def format_summary(
        self,
        summary: Dict[str, Any],
        config: SummaryOutputConfig,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Format summary as JSON."""
        if config.mode == "none":
            return json.dumps({}, ensure_ascii=False, indent=2)
        
        result = {}
        
        # Add bullets if requested
        if config.mode in ["bullets", "both"]:
            bullets = summary.get("bullets", [])
            result["bullets"] = bullets[:config.bullets_max]
        
        # Add abstract if requested
        if config.mode in ["abstract", "both"]:
            abstract = summary.get("abstract", "")
            if len(abstract) > config.abstract_max_chars:
                abstract = abstract[:config.abstract_max_chars].rsplit(" ", 1)[0] + "..."
            result["abstract"] = abstract
        
        # Add configuration metadata
        result["config"] = {
            "mode": config.mode,
            "language": config.language,
            "bullets_max": config.bullets_max,
            "abstract_max_chars": config.abstract_max_chars,
        }
        
        if metadata:
            result["metadata"] = metadata
        
        return json.dumps(result, ensure_ascii=False, indent=2)
