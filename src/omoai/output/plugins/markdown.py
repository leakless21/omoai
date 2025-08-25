"""
Markdown formatter for transcripts and summaries.
"""

from typing import Any, Dict, List, Optional

from ..formatter import OutputFormatter, register_formatter, format_timestamp, extract_segment_data
from ...config.schemas import TranscriptOutputConfig, SummaryOutputConfig


@register_formatter
class MarkdownFormatter(OutputFormatter):
    """Markdown formatter for transcripts and summaries."""
    
    format_name = "md"
    file_extension = ".md"
    
    def format_transcript(
        self,
        segments: List[Dict[str, Any]],
        transcript_raw: str,
        transcript_punct: str,
        config: TranscriptOutputConfig,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Format transcript as Markdown."""
        lines = ["# Transcript", ""]
        
        # Add metadata if available
        if metadata:
            lines.append("## Metadata")
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    lines.append(f"- **{key}**: {value}")
            lines.append("")
        
        # If we want segments with timestamps
        if config.include_segments and config.timestamps != "none":
            lines.append("## Segments")
            lines.append("")
            
            normalized_segments = extract_segment_data(segments)
            
            for i, seg in enumerate(normalized_segments, 1):
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
                    lines.append(f"### Segment {i} ({start_ts} - {end_ts})")
                else:
                    lines.append(f"### Segment {i}")
                
                lines.append(text.strip())
                lines.append("")
        
        # If we want full transcript without individual timestamps
        elif config.include_punct and transcript_punct:
            lines.append("## Full Transcript")
            lines.append("")
            lines.append(transcript_punct)
            lines.append("")
        
        elif config.include_raw and transcript_raw:
            lines.append("## Raw Transcript")
            lines.append("")
            lines.append(transcript_raw)
            lines.append("")
        
        return "\n".join(lines)
    
    def format_summary(
        self,
        summary: Dict[str, Any],
        config: SummaryOutputConfig,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Format summary as Markdown."""
        if config.mode == "none":
            return ""
        
        lines = ["# Summary", ""]
        
        # Add metadata if available
        if metadata:
            lines.append("## Metadata")
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    lines.append(f"- **{key}**: {value}")
            lines.append("")
        
        # Add bullets if requested
        if config.mode in ["bullets", "both"]:
            bullets = summary.get("bullets", [])
            if bullets:
                lines.append("## Key Points")
                lines.append("")
                for bullet in bullets[:config.bullets_max]:
                    lines.append(f"- {bullet}")
                lines.append("")
        
        # Add abstract if requested  
        if config.mode in ["abstract", "both"]:
            abstract = summary.get("abstract", "")
            if abstract:
                if lines:  # Add separator if we already have bullets
                    lines.append("## Abstract")
                    lines.append("")
                
                # Truncate abstract if needed
                if len(abstract) > config.abstract_max_chars:
                    abstract = abstract[:config.abstract_max_chars].rsplit(" ", 1)[0] + "..."
                
                lines.append(abstract)
                lines.append("")
        
        # Add configuration info
        lines.append("## Configuration")
        lines.append("")
        lines.append(f"- **Mode**: {config.mode}")
        lines.append(f"- **Language**: {config.language}")
        lines.append(f"- **Max Bullets**: {config.bullets_max}")
        lines.append(f"- **Max Abstract Length**: {config.abstract_max_chars} characters")
        
        return "\n".join(lines)
