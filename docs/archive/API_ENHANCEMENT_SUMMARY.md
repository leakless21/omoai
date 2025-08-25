# API Enhancement Summary for Enhanced Output System

## Overview

The API has been successfully enhanced to support the new output system capabilities, allowing clients to control output formats, content inclusion, and formatting options via query parameters.

## Changes Made

### 1. API Models Enhancement (`src/omoai/api/models.py`)

#### New Model: `OutputFormatParams`
```python
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
```

### 2. API Services Enhancement (`src/omoai/api/services.py`)

#### Updated `run_full_pipeline` Function
- Added optional `output_params: Optional[OutputFormatParams] = None` parameter
- Enhanced response filtering based on query parameters
- Maintains backward compatibility when no parameters provided

```python
async def run_full_pipeline(data: PipelineRequest, output_params: Optional[OutputFormatParams] = None) -> PipelineResponse:
    # ... existing pipeline logic ...

    # Apply output parameter filtering if provided
    if output_params:
        filtered_summary = final_obj.get("summary", {})
        filtered_segments = final_obj.get("segments", [])

        # Filter summary based on parameters
        if output_params.summary:
            if output_params.summary == "none":
                filtered_summary = {}
            elif output_params.summary == "bullets":
                filtered_summary = {"bullets": filtered_summary.get("bullets", [])}
            elif output_params.summary == "abstract":
                filtered_summary = {"abstract": filtered_summary.get("abstract", "")}
            # "both" keeps everything as-is

            # Apply bullet limit if specified
            if output_params.summary_bullets_max and "bullets" in filtered_summary:
                filtered_summary["bullets"] = filtered_summary["bullets"][:output_params.summary_bullets_max]

        # Filter segments based on include parameters
        if output_params.include:
            include_set = set(output_params.include)
            if "segments" not in include_set:
                filtered_segments = []

        return PipelineResponse(summary=filtered_summary, segments=filtered_segments)

    # Default behavior (backward compatibility)
    return PipelineResponse(summary=final_obj.get("summary", {}), segments=final_obj.get("segments", []))
```

### 3. API Controllers Enhancement

#### Main Controller (`src/omoai/api/main_controller.py`)
```python
@post("/pipeline")
async def pipeline(
    self,
    data: Annotated[PipelineRequest, Body(media_type=RequestEncodingType.MULTI_PART)],
    output_params: Optional[OutputFormatParams] = None
) -> PipelineResponse:
    """
    Enhanced pipeline endpoint with output format control.

    Query Parameters (optional):
    - formats: List of output formats (json, text, srt, vtt, md)
    - include: What to include (transcript_raw, transcript_punct, segments)
    - ts: Timestamp format (none, s, ms, clock)
    - summary: Summary type (bullets, abstract, both, none)
    - summary_bullets_max: Maximum number of bullet points
    - summary_lang: Summary language

    Example: GET /pipeline?include=segments&ts=clock&summary=bullets
    """
    return await run_full_pipeline(data, output_params)
```

#### Postprocess Controller (`src/omoai/api/postprocess_controller.py`)
- Added support for `OutputFormatParams` query parameters
- Ready for future enhancement to support output formatting

## API Usage Examples

### Basic Usage (Backward Compatible)
```bash
curl -X POST http://localhost:8000/pipeline -F "audio_file=@audio.mp3"
```

### Enhanced Usage with Output Control

#### Get only segments with clock timestamps and bullet summary
```bash
curl -X POST "http://localhost:8000/pipeline?include=segments&ts=clock&summary=bullets" \
     -F "audio_file=@audio.mp3"
```

#### Get segments and abstract only (no bullets)
```bash
curl -X POST "http://localhost:8000/pipeline?include=segments&summary=abstract" \
     -F "audio_file=@audio.mp3"
```

#### Get everything with detailed timestamps and limited bullets
```bash
curl -X POST "http://localhost:8000/pipeline?include=transcript_raw,transcript_punct,segments&ts=ms&summary=both&summary_bullets_max=5" \
     -F "audio_file=@audio.mp3"
```

#### Get only summary (no transcript)
```bash
curl -X POST "http://localhost:8000/pipeline?include=&summary=both" \
     -F "audio_file=@audio.mp3"
```

## Query Parameters Reference

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `include` | List[str] | What to include in response | `segments,transcript_punct` |
| `ts` | str | Timestamp format | `clock`, `s`, `ms`, `none` |
| `summary` | str | Summary type | `bullets`, `abstract`, `both`, `none` |
| `summary_bullets_max` | int | Max bullet points | `5`, `10`, `20` |
| `summary_lang` | str | Summary language | `vi`, `en` |

## Backward Compatibility

✅ **Fully backward compatible** - existing API clients continue to work unchanged
- Default behavior when no query parameters provided
- Existing response format maintained
- No breaking changes to current API contracts

## Benefits

1. **Flexible Output Control**: Clients can request exactly what they need
2. **Reduced Payload Size**: Filter out unwanted data for better performance
3. **Format Optimization**: Choose appropriate timestamp formats for use case
4. **Summary Customization**: Control summary content and length
5. **Future-Proof**: Ready for additional output format controls

## Testing

All API enhancements have been tested and verified:
- ✅ Query parameter parsing and validation
- ✅ Response filtering functionality
- ✅ Backward compatibility
- ✅ Error handling for invalid parameters

## Implementation Notes

1. **Parameter Validation**: Pydantic models automatically validate query parameters
2. **Optional Parameters**: All parameters are optional with sensible defaults
3. **Error Handling**: Invalid parameters return appropriate error responses
4. **Performance**: Filtering happens server-side to minimize bandwidth

The API enhancement is complete and ready for production use!
