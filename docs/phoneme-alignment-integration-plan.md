# Phonetic/Character‑Level Timestamp Integration Plan (WhisperX‑Referenced)

This document describes how to add optional phonetic/character‑level timestamps to OmoAI
by replicating WhisperX’s forced alignment approach (via wav2vec2 CTC) and adapting it to
our current architecture. The reference implementation is vendored under `ref/whisperx`.

Goals
- Accurate word‑level and optional character‑level timestamps ("phonetic‑level" granularity).
- Preserve current outputs and behavior by default; add non‑breaking enrichments.
- Integrate cleanly with existing VAD pre‑segmentation, ASR, and post‑processing.
- Ship behind a configuration flag with safe fallbacks.

Scope & Current State
- ASR flow: `scripts/asr.py` decodes with ChunkFormer into segments with string timestamps
  (`"HH:MM:SS:MS"`) and unpunctuated text (`text_raw`). Optional VAD is already integrated (see
  `docs/vad-integration-plan.md`, `src/omoai/integrations/vad.py`).
- Post‑process: `scripts/post.py` handles punctuation/summarization and parses flexible timestamp
  formats using `_parse_time_to_seconds`.
- Config: Pydantic schemas in `src/omoai/config/schemas.py`; VAD schema exists. No alignment schema
  yet.
- Reference: WhisperX alignment is vendored under `ref/whisperx/alignment.py` and related modules.

WhisperX Model & APIs To Replicate (from `ref/whisperx`)
- Load alignment model:
  - `whisperx.alignment.load_align_model(language_code, device, model_name=None)`
  - Supports torchaudio pipelines for `{en, fr, de, es, it}` and many languages via Hugging Face.
  - Vietnamese default in the reference list: `"nguyenvulebinh/wav2vec2-base-vi"`.
- Run alignment:
  - `whisperx.alignment.align(segments, model, metadata, audio, device, *,
     interpolate_method="nearest", return_char_alignments=False, print_progress=False)`
  - Returns an object with `segments` enriched with `words` and optional `chars`, and a flattened
    `word_segments` list.
- Internals (for context): trellis building, beam backtrack, `merge_repeats`/`merge_words`.
  We will use the API above rather than re‑implementing internals.

High‑Level Integration Strategy
1) Map our ASR segments to WhisperX’s expected input (float seconds and `text`).
2) Call WhisperX alignment using the vendored code in `ref/whisperx`.
3) Merge the aligned results back into our segment objects without breaking existing fields.
4) Expose configuration to enable/disable alignment and character‑level detail.
5) Keep failure‑proof fallbacks: on any error, skip alignment and retain current outputs.

Output Schema (Non‑Breaking Additions)
- Keep existing fields:
  - `segments[*].start` and `segments[*].end`: unchanged (string timestamps).
  - `segments[*].text_raw`: unchanged.
- Additive enrichments:
  - `segments[*].words`: list of `{ word: str, start: float, end: float, score: float }`.
  - `segments[*].chars` (optional when enabled): list of
    `{ char: str, start: float, end: float, score: float }`.
  - Top‑level: `word_segments`: flattened list of all words.
  - `metadata.alignment`: `{ enabled, language, model, type, interpolate_method,
     return_char_alignments }`.

Configuration Additions
- `config.yaml` (example):

```yaml
alignment:
  enabled: false
  # auto → derive from known language or default; else explicit code (e.g., "vi")
  language: auto
  # cpu|cuda|auto; we recommend defaulting to cpu to avoid GPU contention
  device: cpu
  # Optional HF/torchaudio model override; when null use WhisperX defaults per language
  align_model: null
  return_char_alignments: false
  # whisperx alignment: nearest|linear|ignore
  interpolate_method: nearest
  print_progress: false
```

- `src/omoai/config/schemas.py` (new Pydantic model):

```python
class AlignmentConfig(BaseModel):
    enabled: bool = Field(default=False)
    language: str | Literal["auto"] = Field(default="auto")
    device: Literal["cpu", "cuda", "auto"] = Field(default="cpu")
    align_model: str | None = Field(default=None)
    return_char_alignments: bool = Field(default=False)
    interpolate_method: Literal["nearest", "linear", "ignore"] = Field(default="nearest")
    print_progress: bool = Field(default=False)

    @field_validator("device")
    @classmethod
    def resolve_device(cls, v: str) -> str:
        if v == "auto":
            try:
                import torch  # noqa: WPS433
                return "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                return "cpu"
        return v
```

- Add `alignment: AlignmentConfig = Field(default_factory=AlignmentConfig)` at the appropriate
  top‑level config (next to `vad`, `asr`, etc.).

Dependencies
- Required for WhisperX alignment (already vendored, but runtime deps needed):
  - `transformers` (Wav2Vec2), `torchaudio` (if using torchaudio pipeline), `nltk` (Punkt tokenizer).
- Our `pyproject.toml` already includes `torchaudio` in the `asr` group. Add:
  - `transformers>=4.43`, `nltk>=3.8` (either core deps or the `asr` group).
- NLTK `punkt` model:
  - If the Punkt data is not present, WhisperX’s tokenizer may fail. We will:
    - Catch and degrade to a simple sentence splitter, or
    - Document a one‑time `nltk.download('punkt')` step for environments that need it.

Module Additions (Adapter Layer)
- New: `src/omoai/integrations/alignment.py`
  - Purpose: Isolate mapping, vendored imports, error handling, and merging logic.
  - Ensure `ref` is importable before importing `whisperx`:

```python
# src/omoai/integrations/alignment.py (sketch)
from __future__ import annotations
import sys
from pathlib import Path
from typing import Any

# Ensure vendored whisperx under ref/ is importable
_ref = Path(__file__).resolve().parents[2] / "ref"
if str(_ref) not in sys.path:
    sys.path.insert(0, str(_ref))

from whisperx.alignment import load_align_model as _wx_load_model  # type: ignore
from whisperx.alignment import align as _wx_align  # type: ignore

from omoai.pipeline.postprocess_core_utils import (
    _parse_time_to_seconds as _p2s,
)

# 1) Map our segments → WhisperX SingleSegment (seconds + text)

def to_whisperx_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in segments:
        text = (s.get("text_raw") or s.get("text") or "").strip()
        if not text:
            continue
        start = _p2s(s.get("start")) or 0.0
        end = _p2s(s.get("end")) or start
        out.append({"start": float(start), "end": float(end), "text": text})
    return out

# 2) Load model via vendored whisperx

def load_alignment_model(language: str, device: str, model_name: str | None = None):
    return _wx_load_model(language_code=language, device=device, model_name=model_name)

# 3) Align and return WhisperX shaped result

def align_segments(
    wx_segments: list[dict[str, Any]],
    audio_path_or_array: Any,
    model: Any,
    metadata: dict,
    device: str,
    *,
    return_char_alignments: bool,
    interpolate_method: str,
    print_progress: bool,
) -> dict:
    return _wx_align(
        wx_segments,
        model,
        metadata,
        audio_path_or_array,
        device,
        interpolate_method=interpolate_method,
        return_char_alignments=return_char_alignments,
        print_progress=print_progress,
    )

# 4) Merge aligned words/chars back to our original segments by index order

def merge_alignment_back(
    original_segments: list[dict[str, Any]],
    aligned_result: dict,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    aligned_segments = aligned_result.get("segments", []) or []
    word_segments = aligned_result.get("word_segments", []) or []

    out = [dict(s) for s in original_segments]
    # Attach words/chars to corresponding non‑empty text segments in order
    j = 0
    for i, s in enumerate(out):
        text = (s.get("text_raw") or s.get("text") or "").strip()
        if not text:
            continue
        if j >= len(aligned_segments):
            break
        al = aligned_segments[j] or {}
        # Preserve existing fields and add enrichments
        if al.get("words"):
            s["words"] = al["words"]
        if al.get("chars") is not None:
            s["chars"] = al["chars"]  # may be [] when return_char_alignments=True
        out[i] = s
        j += 1
    return out, list(word_segments)
```

Wiring in `scripts/asr.py`
- After segments are collected (VAD windows merged) and before writing JSON:

```python
from omoai.config.schemas import get_config
cfg = get_config()

# ... after computing `segments`, `transcript_raw`, `audio_duration_s`
output: dict[str, Any] = {
    "audio": {"sr": 16000, "path": str(audio_path.resolve()), "duration_s": audio_duration_s},
    "segments": segments,
    "transcript_raw": transcript_raw,
    "metadata": {"asr_model": str(model_checkpoint), "params": {...}},
}

# Optional VAD metadata already set above (unchanged)

# NEW: Optional alignment enrichment
try:
    if getattr(cfg, "alignment", None) and bool(cfg.alignment.enabled):
        from omoai.integrations.alignment import (
            to_whisperx_segments,
            load_alignment_model,
            align_segments as _align_segments,
            merge_alignment_back,
        )
        language = cfg.alignment.language
        if language == "auto":
            # Heuristic/default: use a configured language or fall back to vi
            language = "vi"
        device = cfg.alignment.device
        model, meta = load_alignment_model(language, device, cfg.alignment.align_model)

        wx_segments = to_whisperx_segments(segments)
        if wx_segments:
            aligned = _align_segments(
                wx_segments,
                audio_path,
                model,
                meta,
                device,
                return_char_alignments=bool(cfg.alignment.return_char_alignments),
                interpolate_method=str(cfg.alignment.interpolate_method),
                print_progress=bool(cfg.alignment.print_progress),
            )
            segments_enriched, flat_words = merge_alignment_back(segments, aligned)
            output["segments"] = segments_enriched
            output["word_segments"] = flat_words
            output.setdefault("metadata", {}).setdefault("alignment", {})
            output["metadata"]["alignment"] = {
                "enabled": True,
                "language": language,
                "model": meta.get("dictionary") and cfg.alignment.align_model or "default",
                "type": meta.get("type", "unknown"),
                "interpolate_method": cfg.alignment.interpolate_method,
                "return_char_alignments": bool(cfg.alignment.return_char_alignments),
            }
        # Cleanup GPU memory if used
        try:
            if device == "cuda":
                import torch
                torch.cuda.empty_cache()
        except Exception:
            pass
except Exception as e:
    # Log best‑effort and continue without alignment
    try:
        from omoai.logging_system.logger import get_logger
        get_logger(__name__).warning(f"Alignment skipped due to error: {e!s}")
    except Exception:
        pass
```

Timestamp Handling
- Input to alignment: convert each segment’s `start`/`end` (string) to float seconds using
  `_parse_time_to_seconds`.
- Output from alignment: per‑word and per‑char timestamps are stored as float seconds in the
  enrichment fields; existing `segments[*].start`/`end` remain strings for compatibility.

Interplay With VAD
- VAD is performed before ASR decode; segments already reflect any VAD offsetting/merging.
- Forced alignment refines timing within segments and does not change the segment boundaries.
- Existing overlap de‑duplication (post‑processing) remains compatible and unchanged.

Logging & Metadata
- Add `metadata.alignment` when alignment runs, including language, model type (torchaudio vs
  huggingface), interpolation method, and whether char alignments were requested.
- Respect `api.stream_subprocess_output` for printing progress if `print_progress` is enabled.

Error Handling & Fallbacks
- If model loading fails (e.g., language not supported and no `align_model` override) or alignment
  raises, log a warning and proceed with the original segments.
- If an individual segment cannot be aligned (as in WhisperX’s fallback branches), keep its
  original content and continue.

Testing Plan
- Unit (no downloads):
  - `to_whisperx_segments` mapping correctness (string → seconds; text_raw selection).
  - `merge_alignment_back` with synthetic aligned result (attach words/chars to the right segments;
    ensure stability when some segments are empty text).
  - Disabled path: with `alignment.enabled=false`, output is unchanged.
  - Adapter resilience: exceptions from alignment are caught and skipped.
- Optional integration (requires models/caches): small audio clip to verify non‑empty
  `word_segments` when enabled.

Rollout Strategy
1) Land adapter module, config schema, and gated wiring in `scripts/asr.py` with defaults off.
2) Add deps (`transformers`, `nltk`) and document the optional `punkt` download.
3) Validate on representative Vietnamese audio; measure runtime overhead on CPU vs CUDA.
4) Later enhancements (optional): word‑highlighted SRT/VTT rendering using `segments[*].words`.

Appendix: Example Output (abridged)

```json
{
  "audio": {"sr": 16000, "path": "/abs/path.wav", "duration_s": 123.4},
  "segments": [
    {
      "start": "00:00:01:250",
      "end": "00:00:03:100",
      "text_raw": "xin chao the gioi",
      "words": [
        {"word": "xin", "start": 1.26, "end": 1.55, "score": 0.91},
        {"word": "chao", "start": 1.55, "end": 1.92, "score": 0.88}
      ],
      "chars": null
    }
  ],
  "word_segments": [
    {"word": "xin", "start": 1.26, "end": 1.55, "score": 0.91}
  ],
  "metadata": {
    "asr_model": "./models/chunkformer/chunkformer-large-vie",
    "params": {"chunk_size": 64, "left_context_size": 128, "right_context_size": 128},
    "alignment": {
      "enabled": true,
      "language": "vi",
      "model": "default",
      "type": "huggingface",
      "interpolate_method": "nearest",
      "return_char_alignments": false
    }
  }
}
```

References Within This Repo
- Vendored WhisperX (reference implementation):
  - `ref/whisperx/alignment.py` (alignment core), `ref/whisperx/transcribe.py` (end‑to‑end usage),
    `ref/whisperx/audio.py` (I/O helpers).
- OmoAI ASR and VAD:
  - `scripts/asr.py` (decode + VAD integration + JSON writer).
  - `src/omoai/integrations/vad.py` and `docs/vad-integration-plan.md` (VAD design/impl).
- Config and post‑processing:
  - `src/omoai/config/schemas.py` (add `AlignmentConfig`).
  - `scripts/post.py` (unchanged; may later consume `words` for advanced subtitle rendering).

Notes
- Default to CPU for alignment to minimize GPU contention with ChunkFormer. Expose device control.
- Vietnamese default alignment model is provided by WhisperX via HF. For other languages not on the
  default lists, require explicit `alignment.align_model` override.
- Character‑level alignment can be expensive; keep `return_char_alignments=false` by default.

