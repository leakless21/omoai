# WhisperX-Inspired ASR Architecture for OmoAI

This document summarizes key learnings from the reference program WhisperX and outlines a
clean, inspired architecture for OmoAI. It explains how to incorporate the concepts without
copying WhisperX code, preserving OmoAI’s current strengths while adding optional, high‑value
capabilities.

> Note: WhisperX under `ref/` is a reference only. We do not copy or embed its code. The
> implementation details below describe original modules and behaviors we will build ourselves
> using public libraries (e.g., torchaudio, Hugging Face, pyannote) where applicable.

## Goals

- Improve timestamp precision (segment and word level).
- Optionally add speaker diarization with per‑segment/word labels.
- Keep OmoAI’s current ASR and LLM post‑processing pipeline intact.
- Normalize timestamps across the stack and maintain backwards compatibility.
- Make additions strictly optional via configuration toggles.

## Current OmoAI Flow (Baseline)

1. Preprocess: ffmpeg converts audio to 16kHz mono PCM16 WAV.
2. ASR: ChunkFormer via `scripts/asr.py` produces `asr.json` with segments and raw transcript.
3. Post‑process: `scripts/post.py` punctuates and summarizes via vLLM, then formats outputs.
4. API and Config: Orchestrated by `src/omoai/api/services.py` and `config.yaml` with solid
   logging and tests.

## Reference Program (WhisperX) — What We Learn

WhisperX pipeline (high level):
- VAD segmentation first, to skip silence and bound chunk lengths.
- ASR with faster‑whisper.
- Forced alignment (wav2vec2) to refine timestamps down to words/characters with confidence.
- Optional diarization (pyannote) to assign speaker labels to segments/words.
- Consistent float‑second timestamps and flexible writers (VTT/SRT/TSV/JSON).

We adopt the ideas, not the code.

## Inspired OmoAI Architecture (Additive, Optional)

The existing flow remains. We add optional stages and normalize times:

1) Timestamp Normalization (always on)
- Immediately convert any time strings (e.g., `"HH:MM:SS:MS"`) to float seconds after ASR JSON
  is loaded in services. Store and return floats going forward.
- Keep readers tolerant of strings for backwards compatibility.

2) Alignment Stage (optional)
- Use a wav2vec2 alignment model (via torchaudio pipelines or Hugging Face) to align the known
  transcript to the audio and recover precise word/char timestamps and scores.
- Output `words` arrays on segments and compute a simple `alignment_confidence` metric (e.g., mean
  word score). On failure, fall back to original segments.

3) Diarization Stage (optional)
- Run a pyannote diarization pipeline to assign speaker labels to segments; if words exist, assign
  speakers at the word level too. Optionally return speaker embeddings.

4) VAD Pre‑Segmentation (optional; later)
- Use VAD (e.g., Silero or pyannote) to skip long silences and cap decode windows before ASR.
- Start simple: exclude silence regions or bound maximum window duration; preserve existing ASR
  batching model.

### Data Flow Summary

```
Audio → Preprocess → ASR (ChunkFormer) → Normalize times (float seconds)
    → [optional] Alignment → [optional] Diarization → Post‑process (LLM)
    → Writers / Persistence / API Response
```

## Component Responsibilities

### Timestamp Normalization
- Convert all segment `start`/`end` values to floats as early as possible in `services.py`.
- Serialize floats in `final.json` and API responses; keep tolerant parsers to accept legacy
  string times on input where necessary.

### Alignment Module (new, original)
- Location: `src/omoai/integrations/alignment.py` (to be created).
- Interface (example):

```python
def align_segments(
    audio_path: str | Path,
    segments: list[dict],
    *,
    language: str = "vi",
    model_name: str | None = None,
    device: str = "auto",
    return_char_alignments: bool = False,
) -> tuple[list[dict], float | None]:
    """Return (aligned_segments, alignment_confidence). Do not raise on failure; fall back."""
```

- Behavior:
  - Load a wav2vec2 CTC model for the language (torchaudio pipeline preferred; fallback to a HF
    checkpoint, both cacheable offline).
  - For each transcript segment, run emissions on the corresponding audio slice and align to text.
  - Produce `words: [{word, start, end, score}]` and optionally `chars: [...]`.
  - Compute `alignment_confidence` as the mean of word scores (ignore missing/NaN).
  - On error, log and return original segments with `alignment_confidence=None`.

### Diarization Module (new, original)
- Location: `src/omoai/integrations/diarization.py` (to be created).
- Interface (example):

```python
def diarize_segments(
    audio_path: str | Path,
    segments: list[dict],
    *,
    model: str = "pyannote/speaker-diarization-3.1",
    device: str = "auto",
    hf_token: str | None = None,
    return_embeddings: bool = False,
) -> tuple[list[dict], dict | None]:
    """Return (segments_with_speakers, embeddings_or_none)."""
```

- Behavior:
  - Run diarization to get speaker time spans; assign `speaker` to segments by overlap. If word
    timings exist, assign speakers to words similarly.
  - Optionally return embeddings keyed by speaker.
  - No‑op cleanly when not configured or token missing.

### VAD (optional, later)
- Thin wrapper around silero/pyannote VAD to generate speech windows. Use to exclude silence and
  bound maximum window length before ASR. Merge text safely across boundaries (avoid duplication).

## Configuration Additions

New optional sections in `config.yaml` (all default to disabled):

```yaml
alignment:
  enabled: false
  device: auto
  language: vi
  model: null          # choose sensible default per language if null
  interpolate_method: nearest
  return_char_alignments: false

diarization:
  enabled: false
  model: pyannote/speaker-diarization-3.1
  device: auto
  hf_token_env: HUGGINGFACE_TOKEN
  return_embeddings: false

vad:
  enabled: false
  method: silero       # or pyannote
  chunk_size: 30
  vad_onset: 0.5
  vad_offset: 0.363
```

## API Behavior and Outputs

- Defaults unchanged unless toggles are enabled.
- When alignment is enabled:
  - `segments` may include `words` arrays and `quality_metrics.alignment_confidence`.
- When diarization is enabled:
  - `segments[i].speaker` and `segments[i].words[j].speaker` may be present.
- Timestamps are returned as float seconds in `final.json` and API responses. Writers continue to
  support SRT/VTT; prefer word‑level spans when available for tighter cues.

## Data Schema (additions)

- Segment base:
  - `start: float`, `end: float`, `text: str`
  - Optional: `text_punct: str`, `speaker: str`
  - Optional: `words: [{word: str, start: float, end: float, score: float, speaker?: str}]`
- Quality metrics:
  - Optional: `quality_metrics.alignment_confidence: float`

## Performance & Reliability

- Lazy‑load models only when the corresponding feature is enabled.
- Use `device: auto` to prefer GPU when available; run on CPU otherwise.
- For long audio, consider enabling VAD to constrain compute; measure before making default.
- On any alignment/diarization failure: log and fall back silently to keep the pipeline robust.

## Dependency Strategy

- No WhisperX dependency. We build new modules ourselves.
- Alignment: prefer `torchaudio.pipelines` models; support Hugging Face Wav2Vec2 checkpoints when
  needed. Encourage local caching for offline runs.
- Diarization: `pyannote.audio` only when enabled; get token from `hf_token_env`. Avoid network at
  runtime when possible by using local caches.

## Security & Privacy

- Do not send audio or transcripts to any external services.
- Keep tokens in environment variables (e.g., `HUGGINGFACE_TOKEN`).
- Consider excluding `transcript_raw` from default API responses for privacy; include only on
  explicit request via query params.

## Testing Plan

- Unit tests
  - Timestamp normalization (string → float) and serialization correctness.
  - Alignment adapter (mock emissions): produces words, handles failure fallback without raising.
  - Diarization assignment: synthetic overlaps for segments/words; optional embeddings shape.
  - Writers prefer word timings when present for SRT/VTT.
- Integration tests
  - Pipeline with alignment enabled/disabled.
  - API response filtering via `include`, `summary` modes, and formats.
- Performance smoke
  - Large audio with VAD off/on to ensure no severe regressions.

## Phased Rollout

1. Normalize timestamps → floats; update docs; validate with tests.
2. Add alignment adapter + toggle; compute `alignment_confidence`; prefer words for SRT/VTT.
3. Add diarization + toggle; add `speaker` to segments/words; support embeddings (optional).
4. Add optional VAD pre‑segmentation; keep default off; validate throughput gains.

## Implementation Pointers

- Wire alignment/diarization inside `src/omoai/api/services.py` after loading the ASR JSON and
  before invoking post‑processing. Keep changes guarded by config flags.
- Normalize times centrally in services so all downstream code sees float seconds.
- Keep module boundaries: new code in `src/omoai/integrations/` with small, testable interfaces.

## Summary

This plan brings three practical improvements inspired by WhisperX—precise timing, diarization,
and (optionally) VAD—without adopting WhisperX code. The design preserves OmoAI’s existing ASR and
LLM strengths, adds value behind safe configuration toggles, and keeps the system robust and
maintainable.

