# OMOAI Program Analysis Report

## 1. Executive Summary

The OMOAI pipeline remains functional and modular. Since the initial analysis, an API layer was added, but it still shells out to the same scripts and exchanges data via temporary files. Key opportunities—to move to in‑memory handoffs, cache models in a long‑lived process, tighten GPU memory handling, reduce duplication, and validate configuration—remain applicable. This update confirms current behavior and refines recommendations with concrete file references.

## 2. Architecture Overview

The system implements a three‑stage pipeline:

1.  **Preprocessing (ffmpeg)**: Converts input audio to 16kHz mono PCM16 WAV.
    - Orchestrated in `src/omoai/main.py` and `src/omoai/interactive_cli.py`
    - Implemented in `scripts/preprocess.py`
2.  **ASR (ChunkFormer)**: Transcribes preprocessed audio.
    - Orchestrated in `src/omoai/main.py` and API services
    - Implemented in `scripts/asr.py` (ChunkFormer wrapper)
3.  **Post‑processing (LLM via vLLM)**: Punctuates transcript and generates summaries.
    - Implemented in `scripts/post.py` using `vllm.LLM`

Execution paths:
- Command‑line orchestrator: `src/omoai/main.py` calls `ffmpeg` then runs `scripts/asr.py` and `scripts/post.py` via `subprocess.run()`.
- Interactive CLI: `src/omoai/interactive_cli.py` wraps the same scripts.
- API layer: `src/omoai/api/services.py` exposes endpoints but still writes temp files and shells out to the scripts via wrappers.

Data is currently passed between stages via on‑disk files (e.g., `preprocessed.wav`, `asr.json`, `final.json`).

## 3. Optimization Opportunities

### 3.1. In‑Memory Data Transfer

The pipeline relies on intermediate files for handoff across stages in both CLI and API paths.

**Recommendation**: Add function‑level APIs that exchange in‑memory objects (e.g., NumPy/Tensor audio buffers and Python dicts for ASR/LLM outputs). Use these in the API to avoid disk I/O; optionally keep a CLI flag to persist artifacts for debugging.

### 3.2. Model Loading and Caching

`scripts/asr.py` initializes ChunkFormer per run and `scripts/post.py` initializes a vLLM engine per run. API singletons exist (`src/omoai/api/asr_controller.py::ASRModel`, `src/omoai/api/postprocess_controller.py::PostprocessModel`), but the service layer (`src/omoai/api/services.py`) does not use them and instead shells out to the scripts.

**Recommendation**: Initialize and cache models in a long‑lived process (the API) and route requests to in‑process functions. This removes repeated model loads and enables reuse (including punctuation/summarization model reuse already present in `scripts/post.py`).

### 3.3. GPU Utilization

`torch.cuda.empty_cache()` is still called:
- In `scripts/asr.py` inside the chunk loop (per‑chunk)
- In `scripts/post.py` between/after LLM stages

**Recommendation**: Avoid per‑chunk cache clears; allow PyTorch/vLLM to manage memory. If needed, clear at stage boundaries only, behind a debug/flag for troubleshooting.

### 3.4. Disk I/O and Temp Artifacts

The API writes to `api.temp_dir` and accumulates `asr_*.json` / `final_*.json` files.

**Recommendation**: When moving to in‑process calls, emit files only when explicitly requested. If temp files remain, add a periodic cleanup policy.

## 4. Simplification and Refactoring Suggestions

### 4.1. Reduce Code Duplication

There is duplication in orchestration and config handling between `src/omoai/main.py` and `src/omoai/interactive_cli.py`.

**Recommendation**: Extract a reusable `pipeline` module exposing `preprocess()`, `run_asr()`, and `postprocess()` functions (pure Python, in‑memory). Let CLI, API, and any batch runner import and call these.

### 4.2. Configuration Management

`src/omoai/api/config.py` provides a dataclass loader for `config.yaml`, but there is no schema validation. `config.yaml` also includes `llm.trust_remote_code: true` by default.

**Recommendation**:
- Validate configuration with a schema (e.g., Pydantic or pydantic‑settings). Enforce required fields like `paths.chunkformer_checkpoint` and types/ranges.
- Centralize config loading so scripts/CLI/API consume the same validated config object.
- Default `trust_remote_code` to false and require an explicit opt‑in.

### 4.3. Script‑based to Function‑based Pipeline

The pipeline currently shells out to `scripts/*.py` via `subprocess.run()`, which complicates error handling and in‑memory transfer.

**Recommendation**: Move core logic from `scripts/` into importable library functions under `src/omoai/` and have both CLI and API call them directly. Keep thin script entrypoints for backward compatibility.

## 5. Potential Errors and Loopholes

### 5.1. Error Handling

`src/omoai/main.py` uses print+return‑code handling for failures, which provides limited diagnostics.

**Recommendation**: Introduce structured logging (with stderr capture for failed subprocesses), error categories, and clear remediation hints. When refactored to in‑process calls, raise typed exceptions surfaced to callers.

### 5.2. Security

`config.yaml` sets `llm.trust_remote_code: true`, which enables arbitrary code execution from model repos.

**Recommendation**: Default to `false`, document risks, and add CLI/API flags to opt‑in explicitly.

### 5.3. Input Validation

Preprocessing assumes valid audio inputs and delegates to `ffmpeg`.

**Recommendation**: Validate inputs early (existence, readable file, size limits, known container/codec) and present actionable messages before invoking heavy work.

### 5.4. Temporary Files Lifecycle

Multiple temp artifacts are written under `api.temp_dir` and `paths.out_dir`.

**Recommendation**: Ensure automatic cleanup (time‑based reaper or on‑success removal) and make retention configurable for debugging.

## 6. Conclusion

The original recommendations remain valid. The added API layer is a natural home for in‑process execution and model caching; adopting it, together with in‑memory handoffs, judicious GPU memory management, config validation, and a reduction in duplication, will substantially improve performance, robustness, and maintainability.
