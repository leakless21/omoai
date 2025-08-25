"""
Text formatter for transcripts and summaries.
"""

from typing import Any, Dict, List, Optional

from ..formatter import OutputFormatter, register_formatter, format_timestamp, wrap_text, extract_segment_data
from ...config.schemas import TranscriptOutputConfig, SummaryOutputConfig


@register_formatter
class TextFormatter(OutputFormatter):
    """Plain text formatter for transcripts and summaries."""
    
    format_name = "text"
    file_extension = ".txt"
    
    def format_transcript(
        self,
        segments: List[Dict[str, Any]],
        transcript_raw: str,
        transcript_punct: str,
        config: TranscriptOutputConfig,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Format transcript as plain text."""
        if not config.include_segments and not config.include_raw and not config.include_punct:
            return ""
        
        lines = []
        
        # If we want segments with timestamps
        if config.include_segments and config.timestamps != "none":
            normalized_segments = extract_segment_data(segments)
            
            for seg in normalized_segments:
                # Choose text based on configuration
                if config.include_punct and seg["text_punct"]:
                    text = seg["text_punct"]
                elif config.include_raw and seg["text_raw"]:
                    text = seg["text_raw"]
                else:
                    continue
                
                # Format timestamp
                start_ts = format_timestamp(seg["start"], config.timestamps)
                end_ts = format_timestamp(seg["end"], config.timestamps)
                
                if start_ts and end_ts:
                    lines.append(f"[{start_ts} - {end_ts}] {text}")
                else:
                    lines.append(text)
        
        # If we want full transcript without individual timestamps
        elif config.include_punct and transcript_punct:
            text = transcript_punct
            if config.wrap_width > 0:
                text = wrap_text(text, config.wrap_width)
            lines.append(text)
        
        elif config.include_raw and transcript_raw:
            text = transcript_raw  
            if config.wrap_width > 0:
                text = wrap_text(text, config.wrap_width)
            lines.append(text)
        
        return "\n".join(lines)
    
    def format_summary(
        self,
        summary: Dict[str, Any],
        config: SummaryOutputConfig,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Format summary as plain text."""
        if config.mode == "none":
            return ""
        
        lines = []
        
        # Add bullets if requested
        if config.mode in ["bullets", "both"]:
            bullets = summary.get("bullets", [])
            if bullets:
                lines.append("# Summary Points")
                lines.append("")
                for bullet in bullets[:config.bullets_max]:
                    lines.append(f"- {bullet}")
                lines.append("")
        
        # Add abstract if requested  
        if config.mode in ["abstract", "both"]:
            abstract = summary.get("abstract", "")
            if abstract:
                if lines:  # Add separator if we already have bullets
                    lines.append("# Abstract")
                    lines.append("")
                
                # Truncate abstract if needed
                if len(abstract) > config.abstract_max_chars:
                    abstract = abstract[:config.abstract_max_chars].rsplit(" ", 1)[0] + "..."
                
                lines.append(abstract)
        
        return "\n".join(lines)
