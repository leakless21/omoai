# Migration Guide: Legacy Scripts to New Pipeline Modules

## Overview

This guide explains how to migrate from using the legacy standalone scripts (`scripts/asr.py`, `scripts/post.py`, `scripts/preprocess.py`) to the new integrated pipeline modules in `src/omoai/pipeline/`.

## Key Changes

1. **Removal of Standalone Scripts**: The legacy scripts in the `scripts/` directory have been archived and are no longer maintained.
2. **Unified Pipeline Modules**: All functionality is now available through the modules in `src/omoai/pipeline/`.
3. **Improved API**: The new modules provide a more consistent and pythonic interface.

## Migration Path

### 1. Preprocessing

**Legacy Script Usage:**
```bash
python scripts/preprocess.py --input input.mp3 --output output.wav
```

**New Module Usage:**
```python
from omoai.pipeline.preprocess import preprocess_file_to_wav_bytes

# Convert file to WAV bytes
wav_bytes = preprocess_file_to_wav_bytes("input.mp3")
with open("output.wav", "wb") as f:
    f.write(wav_bytes)
```

### 2. ASR (Automatic Speech Recognition)

**Legacy Script Usage:**
```bash
python scripts/asr.py --config config.yaml --audio input.wav --out output.json
```

**New Module Usage:**
```python
from omoai.pipeline.asr import run_asr_inference

# Using the modern API with configuration
result = run_asr_inference(
    audio_input="input.wav",
    config=your_config  # OmoAIConfig object or dict
)
```

### 3. Postprocessing

**Legacy Script Usage:**
```bash
python scripts/post.py --config config.yaml --asr-json asr.json --out final.json
```

**New Module Usage:**
```python
from omoai.pipeline.postprocess import postprocess_transcript

# Using the modern API
result = postprocess_transcript(
    asr_result=asr_result,   # ASRResult object
    config=your_config       # OmoAIConfig object or dict
)
```

## Benefits of Migration

1. **Better Performance**: The new pipeline modules are optimized for in-memory processing, reducing I/O overhead.
2. **Consistent API**: All functionality follows a consistent interface pattern.
3. **Improved Error Handling**: Better error messages and handling throughout the pipeline.
4. **Enhanced Configuration**: Centralized configuration management with validation.
5. **Easier Maintenance**: Single codebase to maintain instead of separate scripts.

## Support

If you encounter any issues during migration, please:
1. Check the updated documentation in `docs/`
2. Review the example usage in the test files
3. Open an issue on the project repository if you find a bug

## Migration Status

✅ **COMPLETED**: The legacy scripts have been fully removed from the active codebase and replaced with the new pipeline modules.

### What Was Done

1. **Legacy Scripts Removed**: All legacy scripts (`scripts/asr.py`, `scripts/post.py`, `scripts/preprocess.py`) have been archived to `archive/legacy_scripts/` and the active `scripts/` directory has been removed.

2. **API Migration Complete**: The API now uses `service_mode: "memory"` by default, which automatically routes all requests through the new pipeline modules (`services_v2.py`) instead of legacy script subprocess calls.

3. **Automatic Fallback**: The `services_enhanced.py` module provides automatic fallback between the new pipeline and legacy scripts if needed, but with the current configuration, it uses the pipeline exclusively.

### Current Architecture

- **Primary API**: Uses `services_enhanced.py` which routes to `services_v2.py` (pipeline-based)
- **Legacy Compatibility**: `services.py` is still available but deprecated and will raise errors if used
- **Performance**: All processing now happens in-memory without subprocess overhead

## Timeline

The legacy scripts were archived on **September 8, 2025** and have been completely removed from active use. The API now runs exclusively on the new pipeline modules, providing better performance and maintainability.