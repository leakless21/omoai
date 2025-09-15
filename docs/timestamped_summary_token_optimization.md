# Timestamped Summary: Token Reduction and Chunked Processing

This document explains how the current `timestamped_summary` pipeline works, why it can be token-heavy, and provides a concrete, code-aligned plan to reduce input tokens and/or split the work into chunks. It includes examples and implementation guidance tailored to the existing codebase.

## 1) Current behavior in this repo

The timestamped summary is generated in `scripts/post.py` only when invoked with `--timestamped_summary`:

```1979:2011:/home/cetech/omoai/scripts/post.py
    if args.timestamped_summary:
        if not VLLM_AVAILABLE or args.dry_run:
            logger.warning("[post] Timestamped summary requested but vLLM not available or dry-run mode; skipping")
        else:
            try:
                print("Generating timestamped summary...")
                timestamped_transcript = prepare_timestamp_context(final["segments"])
                system_prompt = cfg_get(["timestamped_summary", "system_prompt"], "You are a helpful assistant that processes transcripts to create summaries...")

                # Build a dedicated LLM instance for timestamped_summary using its own config
                ts_model_id = cfg_get(["timestamped_summary", "llm", "model_id"], model_id_default) or model_id_default
                ts_quant = cfg_get(["timestamped_summary", "llm", "quantization"], quant_default) or quant_default
                ts_mml = int(cfg_get(["timestamped_summary", "llm", "max_model_len"], None) or mml_default)
                ts_gmu = float(cfg_get(["timestamped_summary", "llm", "gpu_memory_utilization"], None) or gmu_default)
                ts_mns = int(cfg_get(["timestamped_summary", "llm", "max_num_seqs"], None) or mns_default)
                ts_mbt = int(cfg_get(["timestamped_summary", "llm", "max_num_batched_tokens"], None) or mbt_default)

                llm_ts = build_llm(ts_model_id, ts_quant, ts_mml, ts_gmu, ts_mns, ts_mbt)

                timestamped_summary_raw = generate_timestamped_summary(llm_ts, timestamped_transcript, system_prompt)
                include_raw = cfg_get(["timestamped_summary", "return_raw"], False)
                timestamped_summary = parse_llm_response(timestamped_summary_raw, include_raw=include_raw)
                final["timestamped_summary"] = timestamped_summary
```

The prompt payload comes from `prepare_timestamp_context`, which serializes every word with a timestamp:

```422:456:/home/cetech/omoai/scripts/post.py
def prepare_timestamp_context(segments: list[dict]) -> str:
    """
    Create a numbered and timestamped transcript string from ASR segments.
    """
    def _format_timestamp(seconds: float) -> str:
        """Format seconds into [MM:SS] string for LLM prompt."""
        ...
        return f"[{minutes:02d}:{seconds:02d}]"

    timestamped_transcript = []
    word_counter = 1
    for segment in segments:
        words = segment.get("word_segments") or segment.get("words")
        if not words:
            continue
        for word in words:
            if "start" in word and "end" in word:
                start_time = _format_timestamp(word['start'])
                timestamped_transcript.append(
                    f"{word_counter}. {start_time} {word['word']}"
                )
                word_counter += 1
    return "\n".join(timestamped_transcript)
```

Parsing supports both `[MM:SS]` and `[HH:MM:SS]` from the model output:

```468:527:/home/cetech/omoai/scripts/post.py
def parse_llm_response(response_text: str, include_raw: bool = False) -> dict:
    ...
    for match in re.finditer(r"\[(\d{2}):(\d{2})(?::(\d{2}))?\]", response_text):
        ...
```

### Why tokens explode

- One line per word means thousands of lines for long audio, with brackets, digits and numbering that tokenize poorly.
- All words are sent in a single prompt by default.

## 2) Goal

Cut input tokens dramatically while keeping sufficient temporal context for LLM-generated “chapter” or “topic” markers, and add a safe chunked mode for extremely long inputs.

## 3) Proposed strategies (compatible with current code)

There are two complementary strategies. You can adopt either or both.

### A) Switch to sentence/segment-level lines (major token cut)

Instead of emitting one line per word, emit one line per sentence (or per ASR segment) with a single start timestamp. That alone reduces tokens by an order of magnitude.

Implementation sketch:

1) Build sentence/segment lines

```python
def prepare_sentence_timestamp_lines(segments: list[dict]) -> str:
    """Emit one line per sentence or segment: [HH:MM:SS] sentence_text.
    Use first word start if available; otherwise segment start."""
    from itertools import chain

    lines: list[str] = []
    for seg in segments:
        words = seg.get("word_segments") or seg.get("words") or []
        if words:
            first = next((w for w in words if "start" in w), None)
            start_s = float(first["start"]) if first else float(seg.get("start", 0.0) or 0.0)
        else:
            start_s = float(seg.get("start", 0.0) or 0.0)

        # Prefer punctuated if available, else raw
        text = (seg.get("text_punct") or seg.get("text_raw") or seg.get("text") or "").strip()
        if not text:
            continue

        hh = int(start_s // 3600)
        mm = int((start_s % 3600) // 60)
        ss = int(start_s % 60)
        ts = f"[{hh:02d}:{mm:02d}:{ss:02d}]"
        lines.append(f"{ts} {text}")
    return "\n".join(lines)
```

2) Use this in place of `prepare_timestamp_context` when length is large (see auto-switch, below).

3) Keep the existing `parse_llm_response` as-is; it already understands `[HH:MM:SS]`.

Trade-offs: You lose exact word-level granularity in the prompt, but gain a huge token reduction while preserving the time anchor for each sentence/segment.

### B) Add chunked (map-reduce) mode for timestamped_summary

Mirror your summarization’s auto-switch logic for `timestamped_summary`. You already have all building blocks: `_compute_input_token_limit`, tokenizers, and segment-based chunkers.

High-level flow:

1) Estimate token count of the prepared prompt text (prefer sentence/segment lines above).
2) If it exceeds a threshold, split into chunks by token budget.
3) For each chunk, prompt the model to emit 0–N topic lines in the format `[HH:MM:SS] Topic` using the earliest relevant timestamp in that chunk.
4) Merge all chunk outputs, sort by time, and deduplicate similar topics.

Chunking helpers (already in code):

```1163:1211:/home/cetech/omoai/scripts/post.py
def _build_segment_batches_by_token_budget(llm: Any, segments: list[dict[str, Any]], token_budget: int, safety_margin: int = 64) -> list[dict[str, Any]]:
    """Group adjacent segments into batches that fit within a token budget."""
    ...
```

For sentence/segment-line prompts you can either:
- Reuse `_build_segment_batches_by_token_budget` and build lines per batch; or
- Use `_compute_input_token_limit` + tokenizer length on the prepared text and split with `_split_text_by_token_budget`.

Config keys already exist under `timestamped_summary` that we can wire up:

```146:186:/home/cetech/omoai/config.yaml
timestamped_summary:
  llm:
    <<: *base_llm
  map_reduce: false
  auto_switch_ratio: 0.98
  auto_margin_tokens: 128
  return_raw: false
  system_prompt: |
    ...
```

We simply need to apply the same auto-switch logic as summarization:

- Compute `token_limit = int(auto_switch_ratio * ts_mml) - auto_margin_tokens`.
- If input tokens > token_limit or `map_reduce: true`, go to chunked flow; else single-pass.

## 4) Detailed implementation steps

All changes are localized to `scripts/post.py` and `config.yaml`. No API surface changes required.

1) Prepare sentence/segment-level context (new helper)

- Add `prepare_sentence_timestamp_lines(segments)` as shown above.

2) Auto-switch logic for `timestamped_summary`

- Right before calling `generate_timestamped_summary`, compute tokens for your chosen prompt (prefer sentence-level when long). Use:

```166:169:/home/cetech/omoai/scripts/post.py
def _compute_input_token_limit(max_model_len: int, prompt_overhead_tokens: int = 128, output_margin_tokens: int = 128) -> int:
    return max(256, int(max_model_len) - int(prompt_overhead_tokens) - int(output_margin_tokens))
```

- Read `timestamped_summary.auto_switch_ratio`, `auto_margin_tokens`, and `map_reduce` from config.
- If over budget, do chunked flow.

3) Chunked flow for `timestamped_summary`

- Option A (segment-native): use `_build_segment_batches_by_token_budget` to group adjacent segments under budget. For each batch, build sentence/segment lines only for the segments in the batch and prompt the model to emit 0–N `[HH:MM:SS] Topic` lines. Keep `max_tokens` small (e.g., 128–256) for cost control.

- Option B (text splitter): prepare a single big sentence-line string and split by tokenizer budget using `_split_text_by_token_budget`; then prompt per chunk.

4) Merge and deduplicate chunk outputs

Pseudocode:

```python
from difflib import SequenceMatcher

def merge_chunk_topics(topics: list[tuple[str, float]]):
    """topics: list of (topic_text, start_seconds) from all chunks"""
    # sort by time
    topics.sort(key=lambda x: x[1])

    merged: list[tuple[str, float]] = []
    for text, ts in topics:
        if not merged:
            merged.append((text, ts))
            continue
        prev_text, prev_ts = merged[-1]
        # simple fuzzy dedup: drop near-duplicates
        sim = SequenceMatcher(None, prev_text.lower(), text.lower()).ratio()
        if sim >= 0.85:
            continue
        merged.append((text, ts))
    return merged
```

- Convert back to final schema using existing `parse_llm_response` (or keep chunk parsing and assemble directly into `{text, start, end}` records, with a fixed small duration like 5–10s as today).

5) Minimal UI/config changes

- Optionally set in `config.yaml`:

```yaml
timestamped_summary:
  map_reduce: true            # enable chunked mode by default for long inputs
  auto_switch_ratio: 0.95     # be a bit more conservative
  auto_margin_tokens: 256     # larger margin to avoid edge cases
```

No API changes are necessary; the result remains under `final["timestamped_summary"]`.

## 5) Prompting guidance per chunk

Use a tight user prompt to keep output compact, e.g.:

```
Given these sentences with start times, output 0–3 key topics as lines in the exact format:
[HH:MM:SS] Topic
Use the earliest relevant time in this chunk and avoid repeating previously seen topics.

<input>
{CHUNK_SENTENCE_LINES}
</input>
```

This pairs well with the existing `timestamped_summary.system_prompt` (which already enforces the `[HH:MM:SS] Topic` shape).

## 6) Examples

### Example A: Single-pass with sentence lines

Input (excerpt):

```
[00:00:16] Chào mừng đến với podcast của chúng tôi hôm nay.
[00:00:45] Chủ đề chính là trí tuệ nhân tạo trong giáo dục.
[00:01:20] Chúng ta sẽ thảo luận về lợi ích và thách thức.
```

Expected model output:

```
[00:00:45] Trí tuệ nhân tạo trong giáo dục
[00:01:20] Lợi ích và thách thức
```

Parsed final JSON fragment:

```json
{
  "timestamped_summary": {
    "summary_text": "[00:00:45] Trí tuệ nhân tạo trong giáo dục [00:01:20] Lợi ích và thách thức",
    "timestamps": [
      {"text": "Trí tuệ nhân tạo trong giáo dục", "start": 45.0, "end": 50.0},
      {"text": "Lợi ích và thách thức", "start": 80.0, "end": 85.0}
    ]
  }
}
```

Token effect: reduces from per-word lines (thousands of tokens) to ~a few hundred tokens for several minutes of audio.

### Example B: Chunked mode for a long file

Assume 60 minutes of content. We set `auto_switch_ratio=0.95`, `auto_margin_tokens=256`, and `ts_mml=50_000` so `token_limit ≈ 47,244`.

1) Prepare sentence lines; tokenizer count gives 120,000 tokens → exceeds limit.
2) Build segment batches under budget. Suppose we get 4 chunks.
3) For each chunk, prompt to emit up to 3 `[HH:MM:SS] Topic` lines.
4) Merge outputs chronologically and deduplicate near-duplicates.

Result: ~8–12 topic lines overall, produced cheaply and safely within context limits.

## 7) Why this is correct for this codebase

- The proposed helpers and flow reuse existing primitives in `scripts/post.py` (token limit estimation, segment-based batching, batch generation utilities), minimizing new surface area and risk.
- `parse_llm_response` already understands both `[MM:SS]` and `[HH:MM:SS]`, so standardized sentence/segment timestamps integrate cleanly.
- The `timestamped_summary` config section already contains `map_reduce`, `auto_switch_ratio`, `auto_margin_tokens`, and `return_raw`, which we can wire up exactly like existing summarization logic.

## 8) Optional enhancements

- Sliding window overlap: add a small overlap (e.g., last 1–2 sentences) between batches to avoid boundary misses; dedupe after merge.
- Topic cap per chunk: cap at 2–3 lines to bound cost and stabilize results.
- Adaptive time granularity: if audio < 1 hour, print `[MM:SS]`; else `[HH:MM:SS]`.

## 9) Quick checklist (implementation)

- [ ] Add `prepare_sentence_timestamp_lines(segments)`.
- [ ] Compute token count of chosen prompt; auto-switch based on `timestamped_summary` config.
- [ ] Implement chunked path using segment batches or tokenizer-based splits.
- [ ] Prompt per chunk and merge outputs with simple fuzzy dedupe.
- [ ] Keep final schema identical; optionally expose `return_raw` (already supported).

---

References and prior work in this repo:

- Summarization map-reduce logic and token budgeting in `scripts/post.py` (`summarize_long_text_map_reduce`, `_compute_input_token_limit`).
- Existing timestamp summary end-to-end integration in `scripts/post.py` (`prepare_timestamp_context`, `generate_timestamped_summary`, `parse_llm_response`).


## 10) Sentence-level timestamp extraction (details)

### Do chunkformer/wav2vec2 punctuate?

- No. Chunkformer and wav2vec2 (alignment) do not add punctuation; punctuation is performed downstream by the LLM punctuation stage. In this repo, punctuation runs first and, by default, preserves word order via `_force_preserve_with_alignment(...)`, so `segments[*].text_punct` aligns with the original words.

### Recommended approach

1. Ensure alignment is enabled so you have word-level times: `segments[*].word_segments` or `segments[*].words` with `start`/`end`.
2. Use punctuated text (`text_punct`) to split into sentences reliably.
3. Map each sentence to the corresponding span of words (same order), and derive the sentence start/end from the first/last word timestamps.

### Code example: build sentence-level timestamps

```python
def sentence_timestamps_from_segments(segments: list[dict]) -> list[dict]:
    """Return a list of {"text", "start", "end"} for each sentence.

    - Uses punctuated text when available for accurate sentence splits.
    - Maps sentences to word-level timestamps assuming preserved word order.
    - Carries tail fragments across segments when a sentence spans boundaries.
    """
    import re

    sentence_end = re.compile(r"[\.!\?…][”\)\"»\]]*\s+")
    out: list[dict] = []
    carry_text = ""
    carry_times: list[tuple[float, float]] = []

    def words_from_segment(seg: dict) -> tuple[str, list[tuple[str, float, float]]]:
        text = (seg.get("text_punct") or seg.get("text_raw") or seg.get("text") or "").strip()
        words_times: list[tuple[str, float, float]] = []
        for w in (seg.get("word_segments") or seg.get("words") or []):
            if "word" in w and "start" in w and "end" in w:
                words_times.append((str(w["word"]), float(w["start"]), float(w["end"])) )
        return text, words_times

    def flush_sentence(buf_tokens: list[str], buf_times: list[tuple[float, float]]):
        nonlocal carry_text, carry_times, out
        if not buf_times and not carry_times:
            return
        text = (" ".join(buf_tokens)).strip()
        if carry_text:
            text = f"{carry_text} {text}".strip()
        start = carry_times[0][0] if carry_times else buf_times[0][0]
        end = (buf_times[-1][1] if buf_times else carry_times[-1][1])
        out.append({"text": text, "start": start, "end": end})
        carry_text, carry_times[:] = "", []

    def emit_sentences(punct_text: str, words_times: list[tuple[str, float, float]]):
        nonlocal carry_text, carry_times
        if not punct_text or not words_times:
            return
        tokens = [t for t in re.split(r"\s+", punct_text) if t]
        # Only advance word index for tokens that contain letters/numbers (ignore pure punctuation)
        wi = 0
        buf_tokens: list[str] = []
        buf_times: list[tuple[float, float]] = []

        for t in tokens:
            bare = re.sub(r"[^\w]+", "", t, flags=re.UNICODE)
            if bare and wi < len(words_times):
                _, s, e = words_times[wi]
                wi += 1
                buf_times.append((s, e))
            buf_tokens.append(t)
            # Sentence boundary reached?
            if sentence_end.search("".join(buf_tokens)):
                flush_sentence(buf_tokens, buf_times)
                buf_tokens, buf_times = [], []

        # Carry tail (no terminal punctuation) to the next segment
        carry_text = (" ".join(buf_tokens)).strip()
        carry_times = buf_times[:]

    for seg in segments:
        text, wt = words_from_segment(seg)
        if not text or not wt:
            continue
        emit_sentences(text, wt)

    if carry_times:
        out.append({
            "text": carry_text.strip(),
            "start": carry_times[0][0],
            "end": carry_times[-1][1],
        })
    return out
```

This assumes your punctuation stage preserves word order (default). If a model occasionally replaces words, add a lightweight alignment (e.g., `SequenceMatcher`) to map punctuated tokens back to `word_segments` words.

### Without punctuation (fallback)

If you cannot rely on punctuation, approximate sentence/phrase boundaries using VAD gaps: treat silences over a threshold as boundaries. Relevant knobs are under `vad.*` in `config.yaml` (e.g., `min_silence_s`, `chunk_size`, `overlap_s`). This is language-agnostic but less precise than punctuation.

### Where to integrate in the pipeline

- Run after punctuation is complete and before `timestamped_summary`. You can add the sentence spans to the final output, or use them to build token-efficient `[HH:MM:SS] sentence` lines for the `timestamped_summary` prompt.
- This complements Section 3A’s strategy of replacing per-word lines with per-sentence lines to reduce tokens.


## 11) Refined, actionable plan

This section provides precise, low-risk edits aligned with the current code to deliver sentence-level prompts and chunked processing for `timestamped_summary`.

### 11.1 Integration steps (code anchors)

- Add a new helper next to other prompt builders in `scripts/post.py`:
  - Function: `prepare_sentence_timestamp_lines(segments: list[dict]) -> str` (see Section 3A or 10 for implementation sketch)

- In the `timestamped_summary` block (anchor shown below), auto-select single-pass vs. chunked:

```1979:2011:/home/cetech/omoai/scripts/post.py
    if args.timestamped_summary:
        ...
        llm_ts = build_llm(...)  # existing

        # 1) Build sentence-level prompt text to cut tokens
        sentence_lines = prepare_sentence_timestamp_lines(final["segments"])  # NEW

        # 2) Compute token counts and budget
        tokenizer_ts = get_tokenizer(llm_ts)
        t_in = len(tokenizer_ts.encode(sentence_lines)) if sentence_lines else 0
        ts_auto_ratio = float(cfg_get(["timestamped_summary", "auto_switch_ratio"], 0.98))
        ts_margin = int(cfg_get(["timestamped_summary", "auto_margin_tokens"], 128))
        ts_mml = int(cfg_get(["timestamped_summary", "llm", "max_model_len"], mml_default))
        token_limit = int(max(0.5, min(1.0, ts_auto_ratio)) * ts_mml) - ts_margin

        use_map_reduce = bool(cfg_get(["timestamped_summary", "map_reduce"], False)) or (t_in and token_limit > 0 and t_in > token_limit)

        if not use_map_reduce:
            # 3) Single-pass on sentence lines
            timestamped_summary_raw = generate_timestamped_summary(llm_ts, sentence_lines, system_prompt)
            include_raw = cfg_get(["timestamped_summary", "return_raw"], False)
            timestamped_summary = parse_llm_response(timestamped_summary_raw, include_raw=include_raw)
            final["timestamped_summary"] = timestamped_summary
        else:
            # 4) Chunked map-reduce: group segments by token budget, prompt per chunk
            batches = _build_segment_batches_by_token_budget(llm_ts, final["segments"], token_limit, safety_margin=64)
            chunk_msgs: list[list[dict[str, str]]] = []
            for b in batches:
                if not b["text"]:
                    continue
                chunk_lines = prepare_sentence_timestamp_lines(final["segments"][b["start_idx"]:b["end_idx"]])
                chunk_msgs.append([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": chunk_lines},
                ])
            # Keep chunk outputs tight
            params_max = 192
            outs = generate_chat_batch(llm_ts, chunk_msgs, temperature=0.2, max_tokens=params_max)

            # 5) Parse each chunk, merge timestamps
            include_raw = cfg_get(["timestamped_summary", "return_raw"], False)
            all_ts = []  # list[(text, start, end)]
            raw_concat: list[str] = []
            for o in outs:
                if not (o or "").strip():
                    continue
                parsed = parse_llm_response(o, include_raw=False)
                raw_concat.append(o)
                for item in parsed.get("timestamps", []) or []:
                    all_ts.append((str(item.get("text", "")).strip(), float(item.get("start", 0.0)), float(item.get("end", 0.0))))

            # 6) Deduplicate and sort by time
            from difflib import SequenceMatcher as _SM
            all_ts.sort(key=lambda x: x[1])
            merged: list[tuple[str, float, float]] = []
            for text, s, e in all_ts:
                if not text:
                    continue
                if merged:
                    prev_text, ps, pe = merged[-1]
                    # near-duplicate textual similarity or very close in time window
                    sim = _SM(None, prev_text.lower(), text.lower()).ratio()
                    if sim >= 0.85 or abs(s - ps) <= 2.0:
                        continue
                merged.append((text, s, e if e > s else s + 5.0))

            # 7) Assemble final object
            summary_text = " ".join([f"[{int(t//3600):02d}:{int((t%3600)//60):02d}:{int(t%60):02d}] {txt}" for (txt, t, _) in merged])
            final_obj = {"summary_text": summary_text, "timestamps": [{"text": txt, "start": s, "end": e} for (txt, s, e) in merged]}
            if include_raw:
                final_obj["raw"] = "\n\n".join([r for r in raw_concat if r.strip()])
            final["timestamped_summary"] = final_obj
```

Notes:
- The snippet uses sentence-level lines exclusively; if you must retain per-word mode for very short prompts, compute and compare token counts of both and pick the smaller.
- `_build_segment_batches_by_token_budget` already exists and is used for punctuation; reusing it keeps behavior consistent.

### 11.2 Prompt details per chunk

- System prompt: keep your existing `timestamped_summary.system_prompt` (it already prescribes `[HH:MM:SS] Topic`).
- User content: pass only the sentence lines for the batch. Optionally prepend a short instruction: “Output 0–3 lines in the format `[HH:MM:SS] Topic`. Avoid repeats.”
- Set `max_tokens` low (128–256) to bound cost and encourage terse outputs.

### 11.3 Merge and deduplicate

- Sort all parsed items by `start`.
- Drop items that are near-duplicates by text (`ratio >= 0.85`) or that are within a tight time window (e.g., <= 2s) to avoid repeats at chunk boundaries.
- Ensure each item has `end >= start`; when missing, default to `start + 5.0` (consistent with current parser semantics).

### 11.4 Configuration guidance

Recommended defaults in `config.yaml`:

```yaml
timestamped_summary:
  map_reduce: true
  auto_switch_ratio: 0.95
  auto_margin_tokens: 256
  return_raw: true
```

- These encourage chunking on long inputs and preserve raw outputs for inspection.
- No new keys are required; sentence-lines are chosen automatically by the token budget logic.

### 11.5 Edge cases and fallbacks

- Missing alignment (no `word_segments`): use segment-level `start`/`end` and segment text; sentences will be coarser but still useful.
- No punctuation: fall back to VAD-based boundaries (`vad.min_silence_s`) to approximate sentences.
- Extremely short chunks: allow 0 outputs to reduce noise.

### 11.6 Testing and QA checklist

- Short clip (< 1 min): verify single-pass; ensure ≤ 2–3 topic lines; confirm bracketed times parse correctly.
- Medium clip (5–15 min): verify auto-switch triggers only when needed; confirm dedupe across boundaries.
- Long clip (≥ 60 min): verify chunking; check that merged topics are chronological and non-duplicated.
- Regression: ensure `parse_llm_response` continues to parse both `[MM:SS]` and `[HH:MM:SS]` forms.
- Measure token savings vs. per-word prompt; expect 5–20× reduction.


## 12) Map-reduce and maximizing token utilization

### 12.1 What “map-reduce” means here

- Map: Split the long input into chunks that each fit the model’s context window (after subtracting a small output buffer). Run the same prompt on each chunk to produce partial results (local topics/markers).
- Reduce: Combine the partial results into a single final output: sort by time, deduplicate near-duplicates, optionally re-summarize if needed. For very long inputs you can reduce hierarchically (chunk → group → final).

We already do this for `summarization` when needed; this section specifies the same approach for `timestamped_summary` with sentence-level inputs.

### 12.2 Fully utilizing the token limit (packing strategy)

Goal: maximize input tokens per request while keeping a minimal, safe buffer for outputs and prompt overhead.

- Budgeting formula (already in code):

```python
# Make the prompt as large as possible while reserving a small output buffer
def _compute_input_token_limit(max_model_len: int, prompt_overhead_tokens: int = 64, output_margin_tokens: int = 128) -> int:
    return max(256, int(max_model_len) - int(prompt_overhead_tokens) - int(output_margin_tokens))
```

- Recommended settings for “tight packing”:
  - In `config.yaml` (both `summarization` and `timestamped_summary`):
    - `auto_switch_ratio: 0.98–0.99` (closer to 1.00 increases risk of overflow)
    - `auto_margin_tokens: 96–128`
    - `map_reduce: true` for long inputs so each chunk fills the window
  - In code where you set per-chunk `max_tokens` for outputs, keep them small (e.g., 128–256) to free most of the window for inputs.

- Chunk building (tokenizer-driven):
  - Prefer sentence/segment lines to minimize token waste.
  - Use tokenizer counts to accumulate sentences until reaching `input_limit`.
  - Avoid or minimize overlap between chunks unless strictly needed.

Example chunk packer for sentence-lines:

```python
def pack_sentence_lines_to_budget(llm, lines_text: str, max_model_len: int, out_margin: int = 128, overhead: int = 64) -> list[str]:
    tokenizer = get_tokenizer(llm)
    input_limit = max(256, int(max_model_len) - overhead - out_margin)
    # Split by lines to preserve sentence boundaries
    lines = [ln for ln in lines_text.splitlines() if ln.strip()]
    chunks: list[str] = []
    cur: list[str] = []
    cur_tokens = 0
    for ln in lines:
        t = len(tokenizer.encode(ln))
        if cur_tokens and cur_tokens + t > input_limit:
            chunks.append("\n".join(cur))
            cur, cur_tokens = [], 0
        cur.append(ln)
        cur_tokens += t
    if cur:
        chunks.append("\n".join(cur))
    return chunks
```

Then for each chunk you can set `max_tokens = out_margin` (or a fixed small value like 192) and call `generate_chat`/`generate_chat_batch`.

### 12.3 Configuration templates

Balanced safety (recommended):

```yaml
summarization:
  map_reduce: true
  auto_switch_ratio: 0.98
  auto_margin_tokens: 96

timestamped_summary:
  map_reduce: true
  auto_switch_ratio: 0.98
  auto_margin_tokens: 96
```

Aggressive packing (use only if stable and monitored):

```yaml
summarization:
  map_reduce: true
  auto_switch_ratio: 0.99
  auto_margin_tokens: 80

timestamped_summary:
  map_reduce: true
  auto_switch_ratio: 0.99
  auto_margin_tokens: 80
```

Note: Pushing closer to 1.00 or shrinking margins increases risk of truncation or generation errors. Prefer 0.98 with ~96–128 margin in production.

### 12.4 Monitoring and safeguards

- Log per-chunk token counts: input tokens, reserved output margin, and `max_tokens` actually set.
- Detect truncation signals (e.g., unusually short outputs or model stop reasons) and auto-retry with a slightly larger margin when needed.
- Consider adding a runtime clamp: `input_limit = max(256, min(input_limit, max_model_len - 256))` as a last-resort guard.


