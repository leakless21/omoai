from pydantic import BaseModel, ConfigDict
from litestar.datastructures import UploadFile


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


# Response Models
class PipelineResponse(BaseModel):
    summary: dict
    segments: list


class PreprocessResponse(BaseModel):
    output_path: str


class ASRResponse(BaseModel):
    segments: list


class PostprocessResponse(BaseModel):
    summary: dict
    segments: list  # Include segments with punctuated text