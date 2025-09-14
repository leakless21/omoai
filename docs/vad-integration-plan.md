# VAD Integration Plan (WhisperX-Referenced) for OmoAI

This document outlines a practical plan to add optional Voice Activity Detection (VAD)
pre‑segmentation to the OmoAI pipeline, referencing implementation patterns from the
WhisperX project stored under `ref/whisperx` in this repository.

Goals
- Reduce compute and latency by skipping long silent regions before ASR.
- Improve robustness of segmenting long audio into bounded windows.
- Preserve transcription quality with careful window overlaps and text de‑duplication.
- Keep the feature optional (default off) and safe: on failure, fall back to current path.

Scope & Assumptions
- Use in the current offline, file‑based ASR flow (no streaming mic input).
- VAD options: `silero` (default, PyPI) and `webrtc` (lightweight). Pyannote VAD can be
  considered later behind a stricter optional flag due to heavier deps.
- Maintain existing ChunkFormer ASR code and outputs. Add VAD as a pre‑ASR stage.

References (WhisperX)
- VAD integration points: `ref/whisperx/asr.py` (see VAD usage and window loop).
- VAD methods: `ref/whisperx/vads/pyannote.py`, `ref/whisperx/vads/silero.py`, and merge logic
  `ref/whisperx/vads/vad.py::merge_chunks`.
- CLI options: `ref/whisperx/__main__.py` (flags `--vad_method`, `--vad_onset`, `--vad_offset`,
  `--chunk_size`).

High‑Level Design
1) Preprocess
   - Keep current `ffmpeg` normalization to 16kHz mono PCM16 WAV.

2) Optional VAD stage (new)
   - Detect speech intervals on the preprocessed waveform.
   - Merge/split intervals to enforce maximum window duration `chunk_size` (seconds), using a
     merge strategy similar to WhisperX (`Vad.merge_chunks`).
   - Add a small overlap (e.g., 0.4 s) between adjacent windows to protect context at edges.

3) ASR per window
   - For each speech window, run the existing ChunkFormer decode path from `scripts/asr.py` over
     that window only (caches reset per window), then offset emitted segment timestamps by the
     window start time and collect all segments.
   - De‑duplicate text at window boundaries (existing `_dedup_overlap` logic in post step can be
     reused if needed; a light variant may be placed near ASR merge if we want earlier cleanup).

4) Post‑process
   - Continue with punctuation and summarization unchanged.
   - Maintain the current output schema; add optional VAD metrics in `metadata`.

Configuration Additions (config.yaml)
```yaml
vad:
  enabled: false
  method: silero        # silero | webrtc | pyannote
  chunk_size: 30        # max speech window length in seconds
  overlap_s: 0.4        # window overlap to protect context
  vad_onset: 0.50       # speech activation threshold (method‑specific)
  vad_offset: 0.363     # speech deactivation threshold (method‑specific)
  min_speech_s: 0.30    # ignore speech shorter than this (smoothing)
  min_silence_s: 0.30   # fill gaps shorter than this (smoothing)
  device: auto          # for Torch‑based methods (silero/pyannote)
  hf_token_env: HUGGINGFACE_TOKEN  # for pyannote if enabled later
  webrtc:
    mode: 2
    frame_ms: 20
    start_hangover_frames: 3
    end_hangover_frames: 5
  silero:
    threshold: 0.50
    min_speech_duration_ms: 250
    min_silence_duration_ms: 100
    max_speech_duration_s: 30
    speech_pad_ms: 30
    window_size_samples: 512
```

New Modules
- `src/omoai/integrations/vad.py`
  - Purpose: Method‑agnostic VAD adapter and interval merging.
  - Interface:
    ```python
    from pathlib import Path
    from typing import Iterable, Tuple, Literal, Optional

    VADMethod = Literal["webrtc", "silero", "pyannote"]

    def detect_speech(
        audio_path: str | Path,
        *,
        method: VADMethod = "silero",
        sample_rate: int = 16000,
        vad_onset: float = 0.5,
        vad_offset: float = 0.363,
        min_speech_s: float = 0.30,
        min_silence_s: float = 0.30,
        chunk_size: float = 30.0,
        webrtc_mode: int = 2,
        frame_ms: int = 20,
        speech_pad_ms: int = 30,
        window_size_samples: int = 512,
        device: str = "auto",
    ) -> list[tuple[float, float]]:
        """Return (start_s, end_s) speech intervals. Never raise; return [] on failure."""

    def merge_chunks(
        segments: Iterable[tuple[float, float]],
        *,
        chunk_size: float,
    ) -> list[dict]:
        """WhisperX‑style merge into bounded windows; returns dicts with "start","end","segments"."""
    ```
  - Methods:
    - `webrtc`: implement with `webrtcvad` over 20 ms frames, with simple hysteresis and smoothing.
    - `silero`: prefer PyPI `silero-vad`; fallback to torch.hub loader if needed.
    - `pyannote`: later; would require local model files or HF token.
  - Guarantees: results are monotonic, within `[0, duration]`, gaps and micro‑segments smoothed
    per config.

Integration Points
- `scripts/asr.py`
  - Read VAD config via `get_config()`.
  - If `vad.enabled`:
    1. Compute `speech_intervals = detect_speech(preprocessed_audio, ...)`.
    2. If empty, log and fall back to single full‑audio window.
    3. Expand neighbors by `overlap_s` (clip to bounds).
    4. Apply `merge_chunks(..., chunk_size)` to bound window lengths.
    5. For each window:
       - Slice audio; run the existing feature extraction and ChunkFormer inference loop.
       - Call `get_output_with_timestamps` as today.
       - Convert each segment’s local HH:MM:SS:MS timestamps to seconds and add `window_start`.
    6. Merge all window segments into the final `segments` list.
  - Keep the non‑VAD path unchanged.

Timestamp Handling
- Current CTC utility emits `"HH:MM:SS:MS"` strings. The post‑processing already parses flexible
  timestamps; we will convert to float seconds at the ASR boundary when VAD is enabled to simplify
  offset addition. For the non‑VAD path, retain current behavior to minimize diffs.

Window Boundary Quality
- Use `overlap_s` (default 0.4 s). At merge, if neighboring segments produce duplicated tokens,
  rely on the existing `_dedup_overlap` logic in post processors; optionally a lightweight de‑duper
  can run right after ASR merge to minimize downstream confusion.

Error Handling & Fallbacks
- If VAD import fails or raises, log a warning and run the current single‑window ASR.
- If VAD yields no intervals, treat entire audio as one window.
- Never fail the request solely because of VAD.

Metrics & Logging
- Add to ASR JSON `metadata.vad`: `{ enabled, method, windows, speech_ratio, params }`.
- Log at INFO: number of windows, speech seconds, percent of audio kept, and per‑window decode
  durations. This supports before/after perf checks.

Testing Plan
- Unit tests (new under `tests/integrations/test_vad.py`):
  - `detect_speech` on synthetic audio with inserted silences; verify intervals and smoothing.
  - `merge_chunks` behavior: chunk bounds, monotonicity, coverage.
  - Timestamp offsetting correctness and monotonic segment times.
- Integration tests:
  - Pipeline with `vad.enabled=false/true` on a long clip; assert roughly similar transcript and
    strictly monotonic times; compare total runtime for a smoke perf delta.
- Edge cases:
  - Very short audios (< 1 s), all‑silence, and speech‑dense clips.
  - Non‑16 kHz inputs (covered by existing preprocess stage).

Performance Expectations
- Speedups scale with silence ratio. On lecture/meeting content, expect noticeable reduction in
  ASR compute time. For speech‑dense audio, overhead should be minimal if VAD is kept lightweight
  (WebRTC) and window merging is simple.

Rollout Strategy
1. Implement `webrtc` VAD adapter and merge logic; integrate behind `vad.enabled` flag.
2. Add metrics/logs; ship with default off.
3. Validate on representative datasets; document expected improvements and caveats.
4. Optionally add `silero` adapter (Torch hub), guarded by availability and a config switch.
5. Consider `pyannote` VAD subsequently if local models/tokens are available.

Risks & Mitigations
- Boundary quality: mitigate with `overlap_s` and text de‑duplication.
- False negatives on soft speech: tune `vad_onset/offset`, min durations; allow method choice.
- Dependency weight: default to `webrtc` to avoid heavy installs; keep others optional.

Work Breakdown (Implementation Tasks)
1. Config: add `vad` section (default off) and Pydantic schema additions if required.
2. Module: create `src/omoai/integrations/vad.py` with `detect_speech` (webrtc) and `merge_chunks`.
3. ASR: wire VAD into `scripts/asr.py` behind flag; implement window slicing and timestamp offset.
4. Metadata: add VAD metrics to ASR JSON `metadata.vad`.
5. Tests: unit tests for VAD + merge + offsets; one integration smoke test.
6. Docs: update `docs/whisperx-inspired-architecture.md` to cross‑reference this plan.

Appendix: WhisperX Artifacts Referenced
- `whisperx/vads/vad.py::merge_chunks` — bounded window merging strategy.
- `whisperx/vads/silero.py` — Silero VAD adapter; onset threshold, chunk interaction.
- `whisperx/vads/pyannote.py` — Pyannote VAD adapter and custom `Binarize` with max‑duration split.
- `whisperx/asr.py` — VAD → windowed ASR loop and aggregation.


Proven Settings (WhisperX‑Style) Adapted to ChunkFormer
- Recommended defaults
  - VAD
    - method: webrtc (default) or silero
    - chunk_size: 30        # seconds, max speech window length
    - overlap_s: 0.4        # seconds, protect context at seams
    - vad_onset: 0.50       # speech activation threshold
    - vad_offset: 0.363     # speech deactivation threshold
    - min_speech_s: 0.30    # drop very short blips
    - min_silence_s: 0.30   # fill tiny gaps between speech
    - device: auto          # for torch‑based methods
  - WebRTC specifics
    - frame_ms: 20
    - mode: 2               # 0..3; 2 balances FP/FN well
    - start_hangover: 3 frames (≈60 ms)
    - end_hangover: 5 frames (≈100 ms)
  - Silero specifics
    - sample_rate: 16000
    - max_speech_duration_s: 30 (match chunk_size)
    - threshold: same as `vad_onset`
  - ChunkFormer parameters (unchanged)
    - asr.chunk_size: 64; left_context_size: 128; right_context_size: 128
    - total_batch_duration_s: 3600
    - Rationale: These are model‑internal frame settings; VAD operates at audio window level.

- Config mapping (config.yaml)
  - Use the existing `vad` section in this plan; set defaults to the values above when enabled.
  - Keep `enabled: false` by default. Allow runtime overrides via environment or CLI where applicable.

- Windowing and merging (WhisperX mapping)
  - Detect raw speech intervals: list of `(start_s, end_s)`.
  - Smooth intervals:
    - Remove segments shorter than `min_speech_s`.
    - Merge gaps shorter than `min_silence_s`.
  - Bound duration: merge/split to ensure each window length ≤ `chunk_size`.
  - Overlap: expand each window by `overlap_s` on both sides, clipped to `[0, audio_duration]`.

- ChunkFormer integration sketch (scripts/asr.py)
  - After audio is normalized to 16 kHz mono, if `vad.enabled`:
    1) `intervals = detect_speech(preprocessed.wav, method, params)`
    2) If empty: `intervals = [(0.0, duration)]`
    3) `windows = merge_chunks(intervals, chunk_size=cfg.vad.chunk_size)`
    4) For each window `w`:
       - Slice waveform: `[int(w.start*16000):int(w.end*16000)]`
       - Compute features and run the existing ChunkFormer inference loop on the slice
       - Decode segments with `get_output_with_timestamps`
       - Convert each local `start/end` to seconds (see Timestamp handling), then add `w.start`
    5) Concatenate all segments in order; optionally de‑duplicate text across boundaries
  - Else: run the current single‑pass path unchanged.

- Timestamp handling (robust)
  - Convert `"HH:MM:SS:MS"` to float seconds using `omoai.pipeline.postprocess_core_utils._parse_time_to_seconds`.
  - When VAD is on, convert immediately after decode, then offset by the window start, so times remain float seconds thereafter.
  - Writers already support SRT/VTT clock formatting from float seconds in the API layer.

- ASR JSON metadata additions
  - `metadata.vad`: `{ enabled, method, chunk_size, overlap_s, windows, speech_ratio, params }`
  - `speech_ratio = total_speech_seconds / total_audio_seconds`

- Tuning guidelines
  - Missed soft speech: decrease `vad_onset` (e.g., 0.45) and/or increase `overlap_s`.
  - False positives/noise: increase `vad_onset` (e.g., 0.55) and/or increase `min_speech_s`.
  - Long monologues truncation: increase `chunk_size` to 45–60; keep `overlap_s` at 0.4.

VAD Tuning (practical)
- Clean talks / lectures
  - silero: threshold ≈ 0.50; min_speech_duration_ms ≈ 250; min_silence_duration_ms ≈ 100
  - webrtc: mode 2; frame_ms 20
  - chunk_size: 30–45; overlap_s: 0.4
- Soft speech / distant mic
  - silero: threshold 0.45; min_speech_duration_ms 200–250
  - webrtc: mode 1–2
  - overlap_s: 0.5
- Noisy calls / background noise
  - silero: threshold 0.55; min_silence_duration_ms 150–200
  - webrtc: mode 3
  - chunk_size: 20–30 to limit drift on difficult audio
Tips
- Watch ASR logs: `[VAD] enabled method=... windows=... speech_ratio=...` to assess impact.
- If boundary truncation appears, raise `overlap_s` (e.g., 0.5) and rely on de‑duplication.

- Why this maps cleanly to ChunkFormer
  - The ASR loop already handles incremental processing with caches; per‑window execution simply resets them at safe boundaries.
  - The existing CTC silence logic continues to refine segmentation inside each window; VAD prevents wasted compute on silence and caps window lengths.
