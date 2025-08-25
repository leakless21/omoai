# Gap Analysis Report

This document tracks bugs, missing features, and issues identified during development and testing. Resolved items are kept for historical reference and to prevent regression through their associated unit tests.

## Active Issues

_No active issues at this time._

## Resolved Issues

### Query Parameter Parsing Bug in Pipeline Endpoint (RESOLVED)

- **Issue**: Query parameters like `summary=bullets&summary_bullets_max=3&include=segments` were not being parsed correctly in the `/pipeline` endpoint
- **Root Cause**: The endpoint parameter `output_params: Optional[OutputFormatParams] = None` was not configured for Litestar query parameter binding. Litestar doesn't automatically parse query parameters into Pydantic models without proper configuration.
- **Symptoms**:
  - Tests showing `output_params: None` in logs even when query parameters were provided
  - Output filtering not working (e.g., `summary_bullets_max=3` was ignored)
- **Resolution**: Modified the endpoint signature to accept individual query parameters explicitly, then construct the `OutputFormatParams` object manually
- **Files Modified**: `src/omoai/api/main_controller.py`
- **Test Coverage**: `tests/test_api_integration_real.py::TestAPIIntegrationWithRealAudio::test_full_pipeline_with_output_parameters`
- **Date Resolved**: 2025-08-22

### ASR Script Module Import Error (RESOLVED)

- **Issue**: `ModuleNotFoundError: No module named 'omoai'` when running ASR script as subprocess from API
- **Root Cause**: When the ASR script is executed as `python -m scripts.asr`, the `src` directory containing the `omoai` package is not automatically in the Python path. The `ensure_chunkformer_on_path` function was only adding the `chunkformer` directory to sys.path, but not the `src` directory which contains the `omoai` package.
- **Symptoms**:
  - API endpoint failing with `AudioProcessingException` when trying to run ASR
  - Subprocess error showing `ModuleNotFoundError: No module named 'omoai'` in stderr
  - Script failing at line `from omoai.chunkformer import decode as cfdecode`
- **Resolution**: Modified the `ensure_chunkformer_on_path` function in `scripts/asr.py` to also add the `src` directory to sys.path, allowing Python to find and import the `omoai` package.
- **Files Modified**: `scripts/asr.py`
- **Test Coverage**: `tests/test_asr_import_fix.py`
- **Date Resolved**: 2025-08-25

### pyproject.toml UV Scripts Configuration Error (RESOLVED)

- **Issue**: UV was failing to parse `pyproject.toml` with the error "unknown field `scripts`, expected one of `required-version`, `native-tls`, `offline`, `no-cache`, `cache-dir`, `preview`, `python-preference`, `python-downloads`, `concurrent-downloads`, `concurrent-builds`, `concurrent-installs`, `index`, `index-url`, `extra-index-url`, `no-index`, `find-links`, `index-strategy`, `keyring-provider`, `allow-insecure-host`, `resolution`, `prerelease`, `fork-strategy`, `dependency-metadata`, `config-settings`, `no-build-isolation`, `no-build-isolation-package`, `exclude-newer`, `link-mode`, `compile-bytecode`, `no-sources`, `upgrade`, `upgrade-package`, `reinstall`, `reinstall-package`, `no-build`, `no-build-package`, `no-binary`, `no-binary-package`, `python-install-mirror`, `pypy-install-mirror`, `python-downloads-json-url`, `publish-url`, `trusted-publishing`, `check-url`, `pip`, `cache-keys`, `override-dependencies`, `constraint-dependencies`, `build-constraint-dependencies`, `environments`, `required-environments`, `conflicts`, `workspace`, `sources`, `managed`, `package`, `default-groups`, `dev-dependencies`, `build-backend`"
- **Root Cause**: The `[tool.uv.scripts]` section is not a valid configuration in UV. UV uses `[project.scripts]` for defining CLI scripts, not `[tool.uv.scripts]`.
- **Symptoms**:
  - Warning message appearing when running any UV command
  - "Failed to parse `pyproject.toml` during settings discovery" warning
- **Resolution**: Moved the `api = "uvicorn omoai.api.app:app --reload"` script from `[tool.uv.scripts]` to `[project.scripts]` section, which is the correct way to define scripts in UV.
- **Files Modified**: `pyproject.toml`
- **Date Resolved**: 2025-08-25

## Future Considerations

- Add configuration validation on application startup
- Implement health check endpoint that validates model accessibility
- Consider environment-specific configuration files for development/production
