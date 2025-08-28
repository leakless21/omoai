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


# Response Models
class PipelineResponse(BaseModel):
    summary: dict
    segments: list
    # Punctuated transcript text for convenience (used for default text/plain responses)
    # Note: Raw transcript is excluded by default for privacy and data minimization
    transcript_punct: str | None = None


class PreprocessResponse(BaseModel):
    output_path: str


class ASRResponse(BaseModel):
    segments: list


class PostprocessResponse(BaseModel):
    summary: dict
    segments: list  # Include segments with punctuated text