"""
Output writer orchestration for OMOAI.

This module coordinates the writing of multiple output formats based on configuration.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config.schemas import OutputConfig
from .formatter import get_formatter, list_formatters
# Import plugins to ensure they're registered
from . import plugins


class OutputWriter:
    """Orchestrates writing of multiple output formats."""
    
    def __init__(self, config: OutputConfig):
        self.config = config
    
    def write_outputs(
        self,
        output_dir: Path,
        segments: List[Dict[str, Any]],
        transcript_raw: str,
        transcript_punct: str,
        summary: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> Dict[str, Path]:
        """
        Write all configured output formats to the specified directory.
        
        Returns:
            Dictionary mapping format names to output file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        written_files = {}
        
        # Write final JSON (always written)
        final_json_path = output_dir / self.config.final_json
        final_json = self._build_final_json(
            segments, transcript_raw, transcript_punct, summary, metadata
        )
        with open(final_json_path, "w", encoding="utf-8") as f:
            json.dump(final_json, f, ensure_ascii=False, indent=2)
        written_files["final_json"] = final_json_path
        
        # Write configured formats
        for format_name in self.config.formats:
            try:
                formatter_class = get_formatter(format_name)
                formatter = formatter_class()
                
                # Write transcript if configured
                if (self.config.transcript.include_raw or 
                    self.config.transcript.include_punct or 
                    self.config.transcript.include_segments):
                    
                    transcript_content = formatter.format_transcript(
                        segments=segments,
                        transcript_raw=transcript_raw,
                        transcript_punct=transcript_punct,
                        config=self.config.transcript,
                        metadata=metadata,
                    )
                    
                    if transcript_content:
                        # Determine filename based on format and content type
                        if format_name == "json":
                            filename = self.config.transcript.file_segments
                        elif format_name == "text":
                            if self.config.transcript.include_punct:
                                filename = self.config.transcript.file_punct
                            elif self.config.transcript.include_raw:
                                filename = self.config.transcript.file_raw
                            else:
                                filename = f"transcript{formatter.file_extension}"
                        else:
                            filename = f"transcript{formatter.file_extension}"
                        
                        transcript_path = output_dir / filename
                        with open(transcript_path, "w", encoding="utf-8") as f:
                            f.write(transcript_content)
                        
                        written_files[f"transcript_{format_name}"] = transcript_path
                
                # Write summary if configured
                if self.config.summary.mode != "none":
                    summary_content = formatter.format_summary(
                        summary=summary,
                        config=self.config.summary,
                        metadata=metadata,
                    )
                    
                    if summary_content:
                        if format_name == "json":
                            filename = f"summary{formatter.file_extension}"
                        elif format_name == "text":
                            filename = self.config.summary.file
                        else:
                            filename = f"summary{formatter.file_extension}"
                        
                        summary_path = output_dir / filename
                        with open(summary_path, "w", encoding="utf-8") as f:
                            f.write(summary_content)
                        
                        written_files[f"summary_{format_name}"] = summary_path
                
            except Exception as e:
                # Log error but continue with other formats
                print(f"Warning: Failed to write {format_name} format: {e}")
                continue
        
        return written_files
    
    def _build_final_json(
        self,
        segments: List[Dict[str, Any]],
        transcript_raw: str,
        transcript_punct: str,
        summary: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build the final JSON output structure."""
        result = {
            "transcript": {},
            "summary": summary,
            "metadata": metadata,
        }
        
        # Add transcript data based on configuration
        if self.config.transcript.include_raw:
            result["transcript"]["raw"] = transcript_raw
        
        if self.config.transcript.include_punct:
            result["transcript"]["punct"] = transcript_punct
        
        if self.config.transcript.include_segments:
            result["transcript"]["segments"] = segments
        
        # Remove empty transcript if no content
        if not result["transcript"]:
            del result["transcript"]
        
        return result


def write_outputs(
    output_dir: Path,
    segments: List[Dict[str, Any]],
    transcript_raw: str,
    transcript_punct: str,
    summary: Dict[str, Any],
    metadata: Dict[str, Any],
    config: OutputConfig,
) -> Dict[str, Path]:
    """
    Convenience function to write outputs with default writer.
    
    Args:
        output_dir: Directory to write outputs to
        segments: List of transcript segments
        transcript_raw: Raw transcript text
        transcript_punct: Punctuated transcript text
        summary: Summary dictionary
        metadata: Additional metadata
        config: Output configuration
        
    Returns:
        Dictionary mapping format names to output file paths
    """
    writer = OutputWriter(config)
    return writer.write_outputs(
        output_dir=output_dir,
        segments=segments,
        transcript_raw=transcript_raw,
        transcript_punct=transcript_punct,
        summary=summary,
        metadata=metadata,
    )
