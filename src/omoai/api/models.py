from pydantic import BaseModel, ConfigDict
from litestar.datastructures import UploadFile
from typing import List, Optional, Literal


# Request Models
class PipelineRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    audio_file: UploadFile


class PreprocessRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    audio_file: UploadFile


class ASRRequest(BaseModel):
    preprocessed_path: str


class PostprocessRequest(BaseModel):
    asr_output: dict


# Enhanced Output Parameters (Query Parameters)
class OutputFormatParams(BaseModel):
    """Query parameters for controlling output formats and options."""

    # Format selection
    formats: Optional[List[Literal["json", "text", "srt", "vtt", "md"]]] = None

    # Transcript options
    include: Optional[List[Literal["transcript_raw", "transcript_punct", "segments"]]] = None
    ts: Optional[Literal["none", "s", "ms", "clock"]] = None

    # Summary options
    summary: Optional[Literal["bullets", "abstract", "both", "none"]] = None
    summary_bullets_max: Optional[int] = None
    summary_lang: Optional[str] = None

    # Quality metrics and diff options
    include_quality_metrics: Optional[bool] = None
    include_diffs: Optional[bool] = None
    # Simple raw summary option
    return_summary_raw: Optional[bool] = None


# Response Models
class QualityMetrics(BaseModel):
    """Quality metrics for punctuation alignment."""
    wer: Optional[float] = None
    cer: Optional[float] = None
    per: Optional[float] = None
    uwer: Optional[float] = None
    fwer: Optional[float] = None
    alignment_confidence: Optional[float] = None


class HumanReadableDiff(BaseModel):
    """Human-readable diff for quality assurance."""
    original_text: Optional[str] = None
    punctuated_text: Optional[str] = None
    diff_output: Optional[str] = None
    alignment_summary: Optional[str] = None


class PipelineResponse(BaseModel):
    # Summary is now a structured dict with the shape:
    # {
    #   "title": str,
    #   "summary": str,    # abstract / main paragraph
    #   "points": List[str]    # bullet points
    # }
    summary: dict
    segments: list
    # Punctuated transcript text for convenience (used for default text/plain responses)
    # Note: Raw transcript is excluded by default for privacy and data minimization
    transcript_punct: str | None = None
    quality_metrics: Optional[QualityMetrics] = None
    diffs: Optional[HumanReadableDiff] = None
    # Optional raw LLM summary text (unparsed), included only on request
    summary_raw_text: Optional[str] = None


class PreprocessResponse(BaseModel):
    output_path: str


class ASRResponse(BaseModel):
    segments: list


class PostprocessResponse(BaseModel):
    # Postprocess now returns a structured summary dict (see PipelineResponse.summary)
    summary: dict
    segments: list  # Include segments with punctuated text
    quality_metrics: Optional[QualityMetrics] = None
    diffs: Optional[HumanReadableDiff] = None
    # Optional raw LLM summary text (unparsed)
    summary_raw_text: Optional[str] = None
