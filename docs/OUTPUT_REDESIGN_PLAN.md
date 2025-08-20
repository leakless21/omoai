## Output Options & Formatting Redesign Plan

### Goal

- **Clarity**: Distinguish outputs by option (transcript vs summary) and by format (timestamps, raw/punctuated, SRT/VTT, text, JSON, MD).
- **Consistency**: Single source of truth for output data; writers render to multiple formats.
- **Configurability**: Control via config, API params, and CLI flags with safe defaults.

### Plan at a Glance

- **Unify output building** into one final JSON object; drive all file writers from it.
- **Extend `OutputConfig`** to toggle transcript parts, timestamp style, summary mode, formats, and filenames.
- **API/CLI params** to select includes, timestamps, and formats per request.
- **Centralize formatters/writers** in `src/omoai/output/` to remove duplication across script and pipeline code.

### Output Schema (Single Source of Truth)

Final JSON returned by pipeline and used by writers:

```json
{
  "transcript": {
    "raw": "...",                
    "punct": "...",              
    "segments": [
      {
        "start_s": 0.00,
        "end_s": 3.45,
        "start_ms": 0,
        "end_ms": 3450,
        "start_clock": "00:00:00.000",
        "end_clock": "00:00:03.450",
        "text_raw": "...",
        "text_punct": "...",
        "confidence": 0.97
      }
    ]
  },
  "summary": {
    "bullets": ["..."],
    "abstract": "..."
  },
  "metadata": { "model": { }, "timing": { }, "audio": { } }
}
```

Notes:
- Segments carry seconds, milliseconds, and clock timestamps. Display chosen by config/API.
- API can omit `transcript.raw`/`transcript.punct` by default to reduce payload size.

### Configuration: Extend `OutputConfig`

Add fields under `output:`:

- **transcript**:
  - `include_raw: bool`
  - `include_punct: bool`
  - `include_segments: bool`
  - `timestamps: enum [none, s, ms, clock]`
  - `wrap_width: int` (keep existing)
  - `file_raw: str` (default `transcript.raw.txt`)
  - `file_punct: str` (default `transcript.punct.txt`)
  - `file_srt: str` (default `transcript.srt`)
  - `file_vtt: str` (default `transcript.vtt`)
  - `file_segments: str` (default `segments.json`)
- **summary**:
  - `mode: enum [bullets, abstract, both, none]`
  - `bullets_max: int`
  - `abstract_max_chars: int`
  - `language: str`
  - `file: str` (default `summary.md`)
- **formats: [json, text, srt, vtt, md]`** (controls which files are written)
- **final_json: str** (default `final.json`)
- Keep `write_separate_files: bool` and `wrap_width` (move under `transcript` for clarity if desired).

### API Additions (Backward-Compatible)

- Endpoints: `POST /pipeline`, `POST /postprocess`
- Optional query params:
  - `include=transcript_raw,transcript_punct,segments`
  - `ts=none|s|ms|clock`
  - `summary=bullets|abstract|both|none`
  - `fmt=json|srt|vtt|text|md` (for secondary artifacts; API may return URLs or a multipart response)
- Defaults remain minimal: segments + summary only.
- Segments always include numeric and clock fields; clients choose what to render.

### CLI / Interactive CLI

- New flags:
  - `--include transcript_raw,transcript_punct,segments`
  - `--timestamps none|s|ms|clock`
  - `--formats json,text,srt,vtt,md`
  - `--summary bullets|abstract|both|none`
  - `--summary-bullets N` `--summary-lang vi|en`
- Interactive: toggles for these with a one-screen preview before writing files.

### Centralized Writers & Formatters

- `src/omoai/output/formatter.py`:
  - `render_transcript_text(segments, use_punct: bool, timestamps: str, wrap_width: int) -> str`
  - `render_srt(segments) -> str`
  - `render_vtt(segments) -> str`
  - `render_summary_md(summary, lang: str, bullets_max: int) -> str`
  - `build_final_json(asr, punct, summary, metadata, options) -> dict`
- `src/omoai/output/writer.py`:
  - `write_all(final_json: dict, output_dir: Path, cfg: OutputConfig) -> Dict[str, Path]`
- Update `src/omoai/pipeline/pipeline.py` and `scripts/post.py` to call writer instead of ad-hoc file writes.

### Timestamps & Segment Clarity

- Store all forms once per segment: `start_s`, `end_s`, `start_ms`, `end_ms`, `start_clock`, `end_clock`.
- Choose display purely at render-time (no recomputation drift).

### Summarization Controls

- Respect `summary.mode` and `bullets_max` when rendering outputs and when returning API responses.
- Writer emits `summary.md` with bullets, abstract, and optional metadata footer in verbose mode.

### File Layout per Run

```
out/
  final.json
  transcript.raw.txt        (optional)
  transcript.punct.txt      (optional)
  transcript.srt            (optional)
  transcript.vtt            (optional)
  segments.json             (optional)
  summary.md                (optional)
```

### Defaults

- API responses: segments + summary; `ts=clock`; `summary=both`; `bullets_max=7`.
- CLI write: `final.json`, `summary.md`, `transcript.punct.txt`; `timestamps=clock`.

### Example Config Snippet

```yaml
output:
  write_separate_files: true
  formats: [json, text, srt, vtt, md]
  transcript:
    include_raw: true
    include_punct: true
    include_segments: true
    timestamps: clock        # none|s|ms|clock
    wrap_width: 100
    file_raw: "transcript.raw.txt"
    file_punct: "transcript.punct.txt"
    file_srt: "transcript.srt"
    file_vtt: "transcript.vtt"
    file_segments: "segments.json"
  summary:
    mode: both               # bullets|abstract|both|none
    bullets_max: 7
    abstract_max_chars: 1000
    language: "vi"
    file: "summary.md"
  final_json: "final.json"
```

### Example API Usage

- `POST /pipeline?include=transcript_punct,segments&ts=clock&summary=both`
- `POST /postprocess?include=transcript_raw,transcript_punct&ts=ms&summary=bullets`

### QA & Migration

- Unit tests for formatters (golden outputs for SRT, VTT, MD, TEXT) and writer path logic.
- Schema tests for new `OutputConfig`; retain old keys with deprecation mapping.
- Update README and docs with before/after examples and parameter reference.

### Implementation Hotspots

- `src/omoai/config/schemas.py`: extend `OutputConfig` and related models.
- `src/omoai/pipeline/pipeline.py`: replace ad-hoc writing with `output.writer.write_all`.
- `scripts/post.py`: same as above.
- `src/omoai/api/models.py`: add optional transcript fields and request params.
- `src/omoai/api/services.py` and `services_v2.py`: plumb include/ts/summary/format params.
- `src/omoai/interactive_cli.py`: add toggles and preview.

### Rollout Steps

1) Add schema changes and create `formatter.py` / `writer.py` with tests.
2) Wire pipeline and scripts to writer; maintain current defaults.
3) Expose API/CLI parameters; keep response defaults backward compatible.
4) Add docs and examples; deprecate old output fields with mapping.
5) Optional: add URLs/multipart for downloadable artifacts in API.


