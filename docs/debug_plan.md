# Comprehensive Debug Plan – OMOAI Test Failures (2025-09-12)

## 1. Failure Matrix

| Test Class                                                                     | Failure                                                                 | Root Cause Bucket                                                                                                                                                   |
| ------------------------------------------------------------------------------ | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `test_structured_logging_to_file`                                              | Assertion `obj.get("message") == "test-message"` fails (got `None`)     | **A-1** Loguru JSON sink serializes `{"record": {...}, "text": "..."}` instead of flat `{"message": ..., "level": ..., "timestamp": ...}` expected by test.         |
| `test_full_pipeline_endpoint_with_real_audio`                                  | Assertion `'points' not found in {'bullets': [...], 'abstract': ...}`   | **B-1** `_normalize_summary()` only invoked in non-filtered branch; filtered branch (lines 457-505 in `services.py`) returns raw dict still containing `"bullets"`. |
| API-services tests (`test_postprocess_service_success`, etc.)                  | AttributeError: `'coroutine' object has no attribute 'summary'`         | **C-1** `postprocess_service()` is async-only; test calls it synchronously.                                                                                         |
| `test_api_singletons.py`                                                       | NameError: `get_service_status` / `warmup_services` not defined         | **D-1** Tests removed these imports during refactor; functions exist in `services.py` but not re-exported to test namespace.                                        |
| `test_complete_logging_setup`                                                  | Assertion `'timestamp' not found in {'text': ..., 'record': {...}}`     | **A-1** Same as first; schema mismatch.                                                                                                                             |
| `test_postprocess_service_script_failure`                                      | `DID NOT RAISE <class 'omoai.api.exceptions.AudioProcessingException'>` | **C-1** Coroutine handling hides actual exception; test expects sync exception.                                                                                     |
| `test_run_full_pipeline_success` & `test_run_full_pipeline_with_output_params` | RuntimeError: `coroutine raised StopIteration`                          | **C-1** Async/sync mismatch in test mocks.                                                                                                                          |
| `test_benchmark_pipeline`                                                      | 500 Internal Server Error from post-process script                      | **E-1** vLLM GPU memory exhaustion (0.78/11.63 GiB free vs 0.9 utilization) causes subprocess failure, unrelated to interface.                                      |

## 2. Root-Cause Buckets

**A. Logging Serialization Contract Drift**  
The project migrated to Loguru with `serialize=True` for file sink.  
Default Loguru JSON structure:

```json
{"record": {"elapsed": ..., "message": "...", "extra": {...}}, "text": "2025-09-12 22:14:46.640 | INFO     | logging:callHandlers:1706 - test-message\n"}
```

Tests expect flat schema:

```json
{
  "timestamp": "...",
  "level": "INFO",
  "message": "test-message",
  "extra": { "component": "test" }
}
```

→ Need deterministic custom serializer.

**B. Summary Normalization Branch Skipped**  
In `_run_full_pipeline_script()` (services.py:360) when `output_params` is provided, the code enters filtered branch (lines 457-505) and returns early; `_normalize_summary()` is only called in the later branch (lines 523-546).  
→ Both branches must canonicalize keys (`bullets` → `points`).

**C. Service Execution Model Inconsistency**  
`asr_service()` implements dual-mode (sync/async) but `postprocess_service()` is async-only. Tests run synchronously and receive a coroutine instead of a response object.  
→ Harmonize both services to dual-mode pattern.

**D. Missing Re-exports for Legacy Tests**  
`test_api_singletons.py` references `get_service_status` / `warmup_services` that were removed from its imports during earlier refactor. Functions still exist in `services.py`.  
→ Re-export or update import paths.

**E. GPU Resource Exhaustion in CI/Local**  
vLLM spawns workers requesting 90 % of GPU memory; only 0.78 GiB free → initialization fails. This is orthogonal to interface correctness but blocks test completion.  
→ Provide opt-out / lower-utilization mode for tests.

## 3. Debug Steps (instrumentation first)

1. Add temporary probe logs:
   - Inside `configure_python_logging()` after adding JSON sink, emit one dummy log and print raw line to stdout for inspection.
   - In `_run_full_pipeline_script`, log which branch is taken and the keys of `summary_data` before return.
   - In `postprocess_service`, log whether running inside event loop.
2. Run single failing test per bucket to confirm hypothesis:
   - `pytest tests/logging/test_logging_smoke.py::test_structured_logging_to_file -v -s`
   - `pytest tests/test_api_integration_real.py::TestAPIIntegrationWithRealAudio::test_full_pipeline_endpoint_with_real_audio -v -s`
   - `pytest tests/test_api_services.py::TestPostprocessService -v -s`
3. Capture actual JSON line and branch log; verify schema and branch selection.

## 4. Ground-Up Remediation Plan

We will implement minimal, deterministic fixes without renaming/mapping hacks:

### 4.1 Provide Deterministic JSON Sink Schema (A)

- Introduce `src/omoai/logging_system/serializers.py` with `flat_json_serializer(record)` → returns dict with `timestamp, level, message, logger, extra`.
- Patch Loguru sink call inside `configure_python_logging()` to use this serializer instead of `serialize=True`.
- Keep rest of Loguru config untouched (rotation, retention, color, etc.).

### 4.2 Summary Normalization Pipeline Object (B)

- Extract `_normalize_summary()` into a pure helper in `src/omoai/api/summary_normalizer.py`.
- Ensure **both** early-filtered branch and late branch call this helper before returning.
- Guarantee output keys: `title, summary, abstract, points` (never `bullets`).

### 4.3 Dual-Mode Service Execution (C)

- Refactor `postprocess_service()` to mirror `asr_service()` pattern:
  - Detect running event loop.
  - If loop present → return `asyncio.to_thread(...)`
  - If no loop → run `_postprocess_script()` synchronously and return result.
- Keep `_postprocess_script()` unchanged (still sync).

### 4.4 Re-export Service Functions (D)

- Add at bottom of `src/omoai/api/services.py`:
  ```python
  # Re-exports for legacy test compatibility
  __all__ += ["get_service_status", "warmup_services"]  # already listed but ensure importable
  ```
- Update `tests/test_api_singletons.py` imports to:
  ```python
  from omoai.api.services import get_service_status, warmup_services
  ```

### 4.5 Optional GPU-Light Mode (E)

- Add environment variable `OMOAI_VLLM_GPU_MEMORY_UTILIZATION` (default `0.9`).
- In `scripts/post.py` → `build_llm()` read this env var and pass `gpu_memory_utilization=float(os.environ.get(..., 0.9))` to vLLM.
- For CI/tests set `export OMOAI_VLLM_GPU_MEMORY_UTILIZATION=0.4` to avoid OOM.

## 5. Implementation Order

1. Add probe logs (temporary).
2. Implement 4.1 (serializer) → verify logging test passes.
3. Implement 4.2 (normalizer) → verify API integration test passes.
4. Implement 4.3 (dual-mode) → verify postprocess_service tests pass.
5. Implement 4.4 (re-exports) → verify singletons tests pass.
6. Implement 4.5 (GPU flag) → reduce flaky GPU failures.
7. Remove probe logs.
8. Run full test suite; capture any new regressions.

## 6. Success Criteria

- All previously failing tests now pass.
- No new test failures introduced.
- Logging JSON lines match expected flat schema.
- API responses always contain `"points"` key, never `"bullets"`.
- Services work both in async controllers and sync scripts/tests.
- GPU-light mode prevents memory exhaustion in constrained environments.

## 7. Documentation Updates (to be delegated)

- Update `ARCHITECTURE.md` → add "Logging Serialization" and "Service Execution Modes" subsections.
- Update `GAP_ANALYSIS.md` → mark resolved issues, add prevention items.
- Create `COMPONENT_LOGGING_DOCS.md` and `COMPONENT_API_SERVICES_DOCS.md` with class listings and file locations.

This plan ensures deterministic, minimal, and maintainable fixes without renaming or fragile remapping.
