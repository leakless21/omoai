## OMOAI Codebase Review

This document summarizes repository structure, core components, execution flows, notable design choices, and prioritized recommendations (performance, reliability, and DX).

### At-a-glance
- Language/runtime: Python 3.11, PyTorch 2.7.x
- Stack: ChunkFormer ASR, optional VAD, optional wav2vec2 alignment, vLLM for punctuation/summarization, Litestar API
- Entry points:
  - CLI: `src/omoai/main.py` (`uv run start`), scripts in `scripts/`
  - API: `src/omoai/api/app.py` (`uv run api`)
  - ASR: `scripts/asr.py` (script-based), model in `src/chunkformer/`

---

## Repository structure

```
.
├─ config.yaml                    # Single source of truth (validated via Pydantic)
├─ pyproject.toml                 # Project + groups (asr, llm), ruff config
├─ src/
│  ├─ omoai/
│  │  ├─ api/                     # Litestar app, controllers, services, wrappers
│  │  ├─ config/                  # Pydantic schemas, singleton loader
│  │  ├─ integrations/            # vad.py, alignment.py
│  │  ├─ logging_system/          # Loguru-backed structured logging
│  │  ├─ pipeline/                # postprocess helpers
│  │  ├─ main.py                  # Orchestrated pipeline (preprocess→ASR→post)
│  │  └─ interactive_cli.py
│  └─ chunkformer/                # Model code (encoder, attention, decode helpers)
│     ├─ model/                   # Conformer/ChunkFormer encoder & utilities
│     └─ decode.py                # Reference script (close to scripts/asr.py)
├─ scripts/
│  ├─ preprocess.py               # ffmpeg to 16kHz mono PCM16
│  ├─ asr.py                      # ChunkFormer decoding + optional VAD + alignment
│  └─ post.py                     # Punctuation + summarization (vLLM)
├─ docs/                          # Architecture & component docs
└─ tests/                         # API, pipeline, config, quality metrics, etc.
```

## Core execution paths

### CLI pipeline
1) `src/omoai/main.py` orchestrates:
   - Preprocess (ffmpeg) → `scripts/preprocess.py`
   - ASR (ChunkFormer) → `scripts/asr.py`
   - Post-process (punctuation/summarization) → `scripts/post.py`
2) Outputs written under `data/output/<stem-YYYYMMDD-HHMMSS>/final.json` (+timed text)

### API
- Litestar app in `src/omoai/api/app.py` exposes:
  - `/v1/preprocess` → `preprocess_audio_service`
  - `/v1/asr` → `asr_service`
  - `/v1/postprocess` → `postprocess_service`
  - `/v1/pipeline` → `run_full_pipeline`
- Services run script-based steps via wrappers (`src/omoai/api/scripts/*.py`) and support streaming/verbose output via `config.yaml`.

## Configuration & logging
- `config.yaml` validated by Pydantic models in `src/omoai/config/schemas.py` (strict, env overrides supported via `OMOAI_*`).
- Logging via `src/omoai/logging_system/` with Loguru sinks, structured records, and noise controls for third-party loggers.

## ASR specifics (scripts/asr.py)
- Loads ChunkFormer from `models/chunkformer/...`, derives subsampling, contexts.
- Feature extraction uses `torchaudio.compliance.kaldi.fbank` (CPU) per window.
- Chunked encoder inference with CTC decoding and timestamp recovery from CTC.
- Optional VAD pre-segmentation (`webrtcvad` or `silero-vad`) to reduce decode time.
- Optional alignment step (`src/omoai/integrations/alignment.py`) providing word-level timestamps using torchaudio/HF wav2vec2.

## Tests & docs
- Tests cover API controllers, pipeline orchestration, config validation, logging, quality metrics, and integration flows.
- Documentation: high-level architecture and component docs under `docs/` align with the implementation.

---

## Notable strengths
- Clear separation between script-based pipeline and API orchestration.
- Centralized config with robust validation and convenient env overrides.
- Comprehensive logging with consistent adapters and performance logging hooks.
- Good test coverage across modules and end-to-end flows.

## Areas to watch
- Duplication: ASR logic appears both in `scripts/asr.py` and `src/chunkformer/decode.py` (drift risk). Consider consolidating shared decode utilities.
- CPU-bound fbank: `torchaudio.compliance.kaldi.fbank` runs on CPU; becomes a bottleneck for GPU inference pipelines.
- Occasional `torch.cuda.empty_cache()` in loops (in reference implementation) can hurt throughput; your script guards this behind an env flag for safety.
- `scripts/post.py` is large/complex; ensure error paths and timeouts are consistently surfaced via the API wrappers.

---

## Performance recommendations (prioritized)

1) Move fbank feature extraction to GPU
   - Replace Kaldi fbank with `kaldifeat.Fbank` on GPU inside `scripts/asr.py` decode path to avoid CPU hot-spot and repeated H2D copies.
   - If keeping Kaldi fbank, at least move the full features tensor to GPU once before the chunk loop.

2) Enable fast matmul and cuDNN heuristics
   - Before model use, set:
     - `torch.backends.cuda.matmul.allow_tf32 = True`
     - `torch.backends.cudnn.allow_tf32 = True`
     - `torch.backends.cudnn.benchmark = True` (inputs relatively stable)
     - `torch.set_float32_matmul_precision("high")`

3) Compile the encoder (PyTorch 2.3+)
   - `model.encoder = torch.compile(model.encoder, mode="reduce-overhead", dynamic=True)` can yield 10–25% on transformer-like inference.

4) Faster audio I/O
   - Prefer `torchaudio.load` + `torchaudio.functional.resample` over `pydub` to avoid Python-level copies.

5) ChunkFormer parameter tuning (throughput vs. quality)
   - `chunk_size`: start with 96–128 for better GPU utilization.
   - `left_context_size`: 64–96; `right_context_size`: ~max(conv_lorder, 32–64).
   - `total_batch_duration_s`: increase to 3600–7200 for long files if memory allows.
   - Keep autocast fp16 on CUDA; try bf16 on Ampere/Hopper if needed.

6) VAD & alignment
   - VAD: increase `chunk_size` (45–60s) and reduce `overlap_s` (0.2–0.3) to cut decode calls; prefer `webrtc` for speed.
   - Alignment: ensure GPU device; disable char-level alignments unless required; keep beam widths small.

7) Micro-optimizations
   - Avoid frequent cache clears; keep `torch.inference_mode()` and autocast (already in place).
   - Consider `channels_last` on inputs/encoder if compatible.

Implementation notes
- Introduce guarded flags and CLI switches to toggle the above without breaking defaults.
- Stage changes: first adopt GPU fbank + cudnn/T F32; then opt-in `torch.compile` once stable.

---

## Reliability & security
- `trust_remote_code` is disabled by default in LLM configs (good).
- API wrappers stream/verbose child logs based on config and enforce timeouts via config.
- Preprocess and postprocess shell out to `ffmpeg`/scripts—inputs are user-provided; ensure temp directories are non-executable and filenames are sanitized (current usage writes to request-scoped temp paths; continue to avoid shell interpolations).

---

## Developer experience (DX)
- The `config.yaml` + Pydantic schema provides a solid UX; keep docstrings in schemas authoritative.
- Consider consolidating decode utilities to reduce duplication between `scripts/asr.py` and `chunkformer/decode.py`.
- Add a small `--fast` preset in `scripts/asr.py` mapping to recommended performance flags/params.
- Provide a `CONTRIBUTING.md` with test matrix and perf benchmark guidance.

---

## Quick checklist (suggested next steps)
- [ ] Add optional GPU fbank (`kaldifeat`) path and CLI switch; default to CPU fbank for portability.
- [ ] Add runtime flags for TF32/cudnn benchmark and opt-in `torch.compile`.
- [ ] Switch audio loading to `torchaudio` in ASR script.
- [ ] Tune default ASR params in `config.yaml` for higher throughput (e.g., chunk_size=96, left=64, right≈64, total_batch_duration_s=3600).
- [ ] Consolidate shared ASR decode code to a single module to avoid drift.
- [ ] Add a small perf doc in `docs/` with A/B results and recommended presets.

---

## References (internal)
- Pipeline orchestrator: `src/omoai/main.py`
- API app & routes: `src/omoai/api/app.py`
- Services and subprocess wrappers: `src/omoai/api/services.py`, `src/omoai/api/scripts/`
- Config schemas: `src/omoai/config/schemas.py`
- ASR script: `scripts/asr.py` (ChunkFormer), model code under `src/chunkformer/`
- VAD: `src/omoai/integrations/vad.py`
- Alignment: `src/omoai/integrations/alignment.py`
- Logging: `src/omoai/logging_system/`
- Docs: `docs/`
- Tests: `tests/`


