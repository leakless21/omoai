"""
SRT subtitle formatter for transcripts.
"""

from typing import Any, Dict, List, Optional

from ..formatter import OutputFormatter, register_formatter, format_timestamp, extract_segment_data
from ...config.schemas import TranscriptOutputConfig, SummaryOutputConfig


@register_formatter
class SRTFormatter(OutputFormatter):
    """SRT subtitle formatter for transcripts."""
    
    format_name = "srt"
    file_extension = ".srt"
    
    def format_transcript(
        self,
        segments: List[Dict[str, Any]],
        transcript_raw: str,
        transcript_punct: str,
        config: TranscriptOutputConfig,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Format transcript as SRT subtitles."""
        if not config.include_segments:
            return ""
        
        normalized_segments = extract_segment_data(segments)
        if not normalized_segments:
            return ""
        
        lines = []
        
        for i, seg in enumerate(normalized_segments, 1):
            # Choose text based on configuration
            if config.include_punct and seg["text_punct"]:
                text = seg["text_punct"]
            elif config.include_raw and seg["text_raw"]:
                text = seg["text_raw"]
            else:
                continue
            
            # Skip empty segments
            if not text.strip():
                continue
            
            # SRT format: sequence number, timestamp, text, blank line
            lines.append(str(i))
            
            # Format timestamps as HH:MM:SS,mmm
            start_time = self._format_srt_time(seg["start"])
            end_time = self._format_srt_time(seg["end"])
            lines.append(f"{start_time} --> {end_time}")
            
            # Add text (can be multiple lines)
            lines.append(text.strip())
            lines.append("")  # Blank line between entries
        
        return "\n".join(lines)
    
    def format_summary(
        self,
        summary: Dict[str, Any],
        config: SummaryOutputConfig,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """SRT format doesn't support summaries - return empty string."""
        return ""
    
    def _format_srt_time(self, seconds: float) -> str:
        """Format seconds as SRT timestamp (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
