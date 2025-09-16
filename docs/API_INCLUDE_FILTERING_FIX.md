# API Include Filtering Fix

## Summary

This document describes the fix implemented to ensure that both `summary` and `timestamped_summary` fields are properly controlled by the `output.api_defaults.include` setting in `config.yaml`.

## Problem

Previously, the `summary` field was being included in API responses even when it was not explicitly listed in the `output.api_defaults.include` setting. The filtering logic in [`src/omoai/api/services.py`](src/omoai/api/services.py) only filtered `segments` and `transcript_punct` based on the include list, but did not filter `summary` or `timestamped_summary`.

## Solution

### 1. Updated Filtering Logic in Services

Modified the filtering logic in [`_run_full_pipeline_script`](src/omoai/api/services.py:783-789) to include checks for `summary` and `timestamped_summary`:

```python
if output_params.include:
    include_set = set(output_params.include)
    if "segments" not in include_set:
        filtered_segments = []
    if "transcript_punct" not in include_set:
        filtered_transcript_punct = ""
    if "summary" not in include_set:
        filtered_summary = {}
    if "timestamped_summary" not in include_set:
        final_obj["timestamped_summary"] = None
```

Also updated the fallback case (lines 883-900) to apply the same filtering logic when no `output_params` are provided.

### 2. Updated Model Definitions

Added `"summary"` as a valid literal in the `include` parameter for:

- [`OutputFormatParams`](src/omoai/api/models.py:34) model
- [`MainController.pipeline`](src/omoai/api/main_controller.py:42) method signature

### 3. Fixed PostprocessResponse Model

Removed the incorrect `timestamped_summary` parameter from [`PostprocessResponse`](src/omoai/api/services.py:345) since that model doesn't support it.

## Configuration

The fix respects the existing configuration in [`config.yaml`](config.yaml:260):

```yaml
output:
  api_defaults:
    include: ["transcript_punct", "timestamped_summary"] # Default fields to include
```

## Usage

### Default Behavior (from config.yaml)

```bash
# Uses config.yaml defaults: ["transcript_punct", "timestamped_summary"]
curl -X POST http://localhost:8000/pipeline \
  -F "audio_file=@test.mp3"
```

### Query Parameter Override

```bash
# Include only specific fields
curl -X POST "http://localhost:8000/pipeline?include=transcript_punct,segments" \
  -F "audio_file=@test.mp3"

# Include summary and timestamped_summary
curl -X POST "http://localhost:8000/pipeline?include=transcript_punct,summary,timestamped_summary" \
  -F "audio_file=@test.mp3"

# Exclude both summary fields
curl -X POST "http://localhost:8000/pipeline?include=transcript_punct,segments" \
  -F "audio_file=@test.mp3"
```

## Valid Include Values

- `transcript_raw` - Raw ASR transcript
- `transcript_punct` - Punctuated transcript
- `segments` - Timestamped segments
- `summary` - Structured summary (title, abstract, bullets)
- `timestamped_summary` - Timestamped summary with [HH:MM:SS] format

## Testing

Comprehensive tests have been added to verify the fix:

1. **Unit Tests**: [`tests/test_api_include_filtering.py`](tests/test_api_include_filtering.py) - Tests the core filtering logic
2. **Integration Tests**: [`tests/test_api_include_integration.py`](tests/test_api_include_integration.py) - Tests configuration and model integration

Run tests with:

```bash
uv run pytest tests/test_api_include_filtering.py tests/test_api_include_integration.py -v
```

## Backward Compatibility

The fix maintains full backward compatibility:

- Existing API calls without `include` parameter continue to work as before
- The default behavior from `config.yaml` is preserved
- All existing functionality remains intact

## Files Modified

1. [`src/omoai/api/services.py`](src/omoai/api/services.py) - Core filtering logic
2. [`src/omoai/api/models.py`](src/omoai/api/models.py) - Added "summary" literal
3. [`src/omoai/api/main_controller.py`](src/omoai/api/main_controller.py) - Updated method signature
4. [`tests/test_api_include_filtering.py`](tests/test_api_include_filtering.py) - New unit tests
5. [`tests/test_api_include_integration.py`](tests/test_api_include_integration.py) - New integration tests

## Result

The API now correctly respects the `include` parameter for both `summary` and `timestamped_summary` fields, allowing fine-grained control over which fields are returned in the response.
