"""
Output formatter interfaces and registry for OMOAI.

This module provides the base interfaces for output formatting and a registry system
for pluggable formatters.
"""

import textwrap
from abc import ABC, abstractmethod
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from ..config.schemas import OutputConfig, TranscriptOutputConfig, SummaryOutputConfig


class OutputFormatter(ABC):
    """Base class for output formatters."""
    
    format_name: str = ""  # Must be overridden by subclasses
    file_extension: str = ""  # Must be overridden by subclasses
    
    @abstractmethod
    def format_transcript(
        self,
        segments: List[Dict[str, Any]],
        transcript_raw: str,
        transcript_punct: str,
        config: TranscriptOutputConfig,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Format transcript data."""
        pass
    
    @abstractmethod
    def format_summary(
        self,
        summary: Dict[str, Any],
        config: SummaryOutputConfig,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Format summary data."""
        pass
    
    def format_final_json(
        self,
        segments: List[Dict[str, Any]],
        transcript_raw: str,
        transcript_punct: str,
        summary: Dict[str, Any],
        metadata: Dict[str, Any],
        config: OutputConfig,
    ) -> str:
        """Format complete pipeline result as JSON. Default implementation."""
        import json
        
        result = {
            "transcript": {
                "raw": transcript_raw if config.transcript.include_raw else None,
                "punct": transcript_punct if config.transcript.include_punct else None,
                "segments": segments if config.transcript.include_segments else None,
            },
            "summary": summary,
            "metadata": metadata,
        }
        
        # Remove None values
        result = {k: v for k, v in result.items() if v is not None}
        if result["transcript"]:
            result["transcript"] = {k: v for k, v in result["transcript"].items() if v is not None}
        
        return json.dumps(result, ensure_ascii=False, indent=2)


# Global formatter registry
_FORMATTERS: Dict[str, Type[OutputFormatter]] = {}


def register_formatter(formatter_class: Type[OutputFormatter]) -> Type[OutputFormatter]:
    """Register an output formatter."""
    if not issubclass(formatter_class, OutputFormatter):
        raise ValueError(f"Formatter must inherit from OutputFormatter")
    
    if not formatter_class.format_name:
        raise ValueError(f"Formatter must have a format_name")
    
    _FORMATTERS[formatter_class.format_name] = formatter_class
    return formatter_class


def get_formatter(format_name: str) -> Type[OutputFormatter]:
    """Get a registered formatter by name."""
    if format_name not in _FORMATTERS:
        raise ValueError(f"Unknown format: {format_name}. Available: {list(_FORMATTERS.keys())}")
    
    return _FORMATTERS[format_name]


def list_formatters() -> List[str]:
    """List all registered formatter names."""
    return list(_FORMATTERS.keys())


# Utility functions for common formatting tasks

def format_timestamp(seconds: float, format_type: str) -> str:
    """Format timestamp according to the specified format."""
    if format_type == "none":
        return ""
    elif format_type == "s":
        return f"{seconds:.2f}s"
    elif format_type == "ms":
        return f"{int(seconds * 1000)}ms"
    elif format_type == "clock":
        # Convert to HH:MM:SS.mmm format
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        milliseconds = int((seconds - total_seconds) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"
    else:
        raise ValueError(f"Unknown timestamp format: {format_type}")


def wrap_text(text: str, width: int) -> str:
    """Wrap text to specified width, preserving paragraphs."""
    if width <= 0:
        return text
    
    paragraphs = text.split("\n\n")
    wrapped_paragraphs = []
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            wrapped_paragraphs.append("")
            continue
        
        wrapped = textwrap.fill(
            para,
            width=width,
            replace_whitespace=False,
            break_long_words=False,
            break_on_hyphens=False,
        )
        wrapped_paragraphs.append(wrapped)
    
    return "\n\n".join(wrapped_paragraphs)


def extract_segment_data(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract and normalize segment data for formatting."""
    normalized = []
    
    for seg in segments:
        # Handle different segment formats
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)
        
        # Extract text (try different field names)
        text_raw = seg.get("text_raw", seg.get("text", ""))
        text_punct = seg.get("text_punct", seg.get("text", text_raw))
        
        confidence = seg.get("confidence")
        
        normalized.append({
            "start": float(start),
            "end": float(end),
            "text_raw": str(text_raw).strip(),
            "text_punct": str(text_punct).strip(), 
            "confidence": confidence,
        })
    
    return normalized
