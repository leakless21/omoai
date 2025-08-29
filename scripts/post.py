import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import gc
import re
import textwrap
try:  # optional torch for CUDA cache clearing; not required in dry-run
    import torch  # type: ignore
except Exception:  # pragma: no cover - keep runnable without torch
    torch = None  # type: ignore

# Environment flag for debug GPU memory clearing
DEBUG_EMPTY_CACHE = os.environ.get("OMOAI_DEBUG_EMPTY_CACHE", "false").lower() == "true"
from difflib import SequenceMatcher

from contextlib import suppress

try:  # optional streaming JSON parser for very large ASR files
    import ijson  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    ijson = None  # type: ignore

try:  # optional progress bar
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover - keep runnable without tqdm
    tqdm = None  # type: ignore

try:  # optional import to allow GPU-free tests without vllm installed
    from vllm import LLM, SamplingParams  # type: ignore
except Exception:  # pragma: no cover - tests won't exercise vllm paths
    LLM = Any  # type: ignore
    class SamplingParams:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
            pass


def load_asr_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_asr_top_level(path: Path) -> Dict[str, Any]:
    """Read only top-level keys of an ASR JSON using ijson if available.

    Skips the heavy `segments` array to reduce memory footprint.
    Fallback to full load if ijson is not available or on error.
    """
    if ijson is None:
        return load_asr_json(path)
    top: Dict[str, Any] = {}
    try:
        with open(path, "rb") as f:
            for key, value in ijson.kvitems(f, ""):
                if key == "segments":
                    continue
                top[key] = value
    except Exception:
        return load_asr_json(path)
    return top


def iter_asr_segments(path: Path):
    """Yield segments one-by-one using ijson if available; otherwise load all."""
    if ijson is None:
        asr = load_asr_json(path)
        for seg in asr.get("segments", []) or []:
            yield seg
        return
    try:
        with open(path, "rb") as f:
            for seg in ijson.items(f, "segments.item"):
                yield seg
    except Exception:
        asr = load_asr_json(path)
        for seg in asr.get("segments", []) or []:
            yield seg


def get_tokenizer(llm: Any):
    return llm.get_tokenizer() if hasattr(llm, "get_tokenizer") else llm.tokenizer


def apply_chat_template(llm: Any, messages: List[Dict[str, str]]) -> str:
    """Build a prompt for chat-style models with a safe fallback.

    Tries tokenizer.apply_chat_template first. If unavailable or failing,
    constructs a simple role-tagged prompt that works for base models too.
    """
    tokenizer = get_tokenizer(llm)
    try:
        tmpl = getattr(tokenizer, "chat_template", None)
        if tmpl:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    except Exception:
        pass
    # Fallback: simple role-tagged format
    parts: List[str] = []
    for msg in messages:
        role = (msg.get("role") or "").strip().lower()
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        if role == "system":
            parts.append("[SYSTEM]\n" + content)
        elif role == "user":
            parts.append("[USER]\n" + content)
        elif role == "assistant":
            parts.append("[ASSISTANT]\n" + content)
        else:
            parts.append(content)
    parts.append("[ASSISTANT]\n")
    return "\n\n".join(parts)


def generate_chat(llm: Any, messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> str:
    prompt = apply_chat_template(llm, messages)
    params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    outputs = llm.generate([prompt], params)
    return outputs[0].outputs[0].text


def generate_chat_batch(llm: Any, list_of_messages: List[List[Dict[str, str]]], temperature: float, max_tokens: int) -> List[str]:
    """Batched chat generation for multiple prompts to improve throughput."""
    if not list_of_messages:
        return []
    prompts = [apply_chat_template(llm, m) for m in list_of_messages]
    params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    outputs = llm.generate(prompts, params)
    return [o.outputs[0].text if o.outputs else "" for o in outputs]


def _compute_input_token_limit(max_model_len: int, prompt_overhead_tokens: int = 128, output_margin_tokens: int = 128) -> int:
    """Conservative upper bound for input tokens so total prompt + output fits model context."""
    return max(256, int(max_model_len) - int(prompt_overhead_tokens) - int(output_margin_tokens))


def _split_text_by_token_budget(llm: Any, text: str, max_input_tokens: int) -> List[str]:
    """Split text into chunks whose tokenized length is <= max_input_tokens.

    Uses the model tokenizer to split exactly on token boundaries to guarantee
    reversibility when concatenated. Falls back to sentence/char-based splitting
    if tokenization fails for any reason.
    """
    if not text:
        return []
    try:
        tokenizer = get_tokenizer(llm)
        tokens = tokenizer.encode(text)
        if len(tokens) <= max_input_tokens:
            return [text]
        chunks: List[str] = []
        start = 0
        n = len(tokens)
        while start < n:
            end = min(start + max_input_tokens, n)
            piece = tokenizer.decode(tokens[start:end])
            chunks.append(piece)
            start = end
        return chunks
    except Exception:
        # Fallback on sentence/char splitter with an approximate size
        approx_chars = max(500, int(max_input_tokens * 4))
        return split_text_into_chunks(text, max_chars=approx_chars, overlap_sentences=1)


def _split_text_by_token_budget_with_counts(llm: Any, text: str, max_input_tokens: int) -> List[Tuple[str, int]]:
    """Return list of (chunk_text, token_count) within token budget to avoid re-encoding."""
    if not text:
        return []
    try:
        tokenizer = get_tokenizer(llm)
        tokens = tokenizer.encode(text)
        if len(tokens) <= max_input_tokens:
            return [(text, len(tokens))]
        out: List[Tuple[str, int]] = []
        start = 0
        n = len(tokens)
        while start < n:
            end = min(start + max_input_tokens, n)
            piece_tokens = tokens[start:end]
            piece = tokenizer.decode(piece_tokens)
            out.append((piece, len(piece_tokens)))
            start = end
        return out
    except Exception:
        approx_chars = max(500, int(max_input_tokens * 4))
        chunks = split_text_into_chunks(text, max_chars=approx_chars, overlap_sentences=1)
        return [(c, max(1, len(c) // 3)) for c in chunks]


def punctuate_text_with_splitting(
    llm: Any,
    text: str,
    system_prompt: str,
    max_model_len: int,
    temperature: float = 0.0,
    batch_prompts: int = 1,
    show_progress: bool = False,
) -> str:
    """Punctuate text safely by splitting into token-budgeted chunks if needed.

    Ensures each generation request keeps input length within model limits.
    """
    if not text:
        return ""
    try:
        tokenizer = get_tokenizer(llm)
    except Exception:
        tokenizer = None

    input_limit = _compute_input_token_limit(max_model_len)
    parts_counts = _split_text_by_token_budget_with_counts(llm, text, input_limit)
    if not parts_counts:
        return ""
    per_piece_max: List[int] = []
    for _, tokens_in in parts_counts:
        per_piece_max.append(max(64, min(int(max_model_len) - 64, int(tokens_in) + 128)))

    pieces_only = [p for p, _ in parts_counts]
    if batch_prompts <= 1:
        iterable = pieces_only
        if show_progress and tqdm is not None:
            iterable = tqdm(iterable, total=len(pieces_only), desc="Punctuating")
        out_pieces: List[str] = []
        for idx, piece in enumerate(iterable):
            if not piece.strip():
                continue
            out_pieces.append(
                punctuate_text(
                    llm,
                    piece,
                    system_prompt,
                    max_tokens=per_piece_max[idx],
                    temperature=temperature,
                )
            )
        return " ".join(p.strip() for p in out_pieces if p is not None).strip()

    # Batched path
    out_texts: List[str] = [""] * len(pieces_only)
    batched_indices: List[int] = list(range(len(pieces_only)))
    start = 0
    while start < len(batched_indices):
        end = min(start + int(max(1, batch_prompts)), len(batched_indices))
        idxs = batched_indices[start:end]
        group_max_tokens = max(per_piece_max[i] for i in idxs)
        list_of_messages: List[List[Dict[str, str]]] = []
        for i in idxs:
            list_of_messages.append([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": pieces_only[i]},
            ])
        group_outs = generate_chat_batch(
            llm,
            list_of_messages,
            temperature=temperature,
            max_tokens=int(group_max_tokens),
        )
        for local_i, gi in enumerate(idxs):
            out_texts[gi] = group_outs[local_i] or ""
        start = end
    return " ".join(s.strip() for s in out_texts if s is not None).strip()

def punctuate_text(llm: Any, text: str, system_prompt: str, max_tokens: Optional[int] = None, temperature: float = 0.0) -> str:
    if not text:
        return ""
    system = system_prompt
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": text},
    ]
    # Heuristic: allow output tokens ~= input tokens + margin
    if max_tokens is None:
        try:
            tokenizer = get_tokenizer(llm)
            max_tokens = min(4096, len(tokenizer.encode(text)) + 64)
        except Exception:
            max_tokens = 1024
    return generate_chat(llm, messages, temperature=temperature, max_tokens=int(max_tokens))


def summarize_text(llm: Any, text: str, system_prompt: str, temperature: float = 0.2) -> Dict[str, Any]:
    if not text:
        return {"bullets": [], "abstract": ""}
    system = system_prompt
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Hãy tóm tắt đoạn văn sau:\n\n{text}"},
    ]
    content = generate_chat(llm, messages, temperature=temperature, max_tokens=800)
    try:
        parsed = json.loads(content)
        bullets = parsed.get("bullets", [])
        abstract = parsed.get("abstract", "")
    except Exception:
        bullets = [line.strip("- ") for line in content.splitlines() if line.strip()]
        abstract = ""
    return {"bullets": bullets, "abstract": abstract}


def _dedup_overlap(prev: str, nxt: str, max_tokens: int = 8) -> str:
    """Remove duplicated token overlap between previous buffer tail and next string head."""
    pt = prev.strip().split()
    nt = nxt.strip().split()
    max_k = min(max_tokens, len(pt), len(nt))
    for k in range(max_k, 0, -1):
        if pt[-k:] == nt[:k]:
            return " ".join(nt[k:])
    return nxt


def _parse_time_to_seconds(value: Any) -> Optional[float]:
    """Parse a timestamp value (float, int, or string) into seconds.

    Supports:
    - numeric types (returned as float)
    - strings like "HH:MM:SS.mmm", "MM:SS.mmm", "HH:MM:SS", "MM:SS",
      and also "HH:MM:SS:ms" where the last field is milliseconds.
    Returns None if parsing fails or value is None.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except Exception:
            return None
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s:
        return None
    s = s.replace(",", ".")
    # Try plain float string first
    try:
        return float(s)
    except Exception:
        pass
    # Split by colon
    parts = s.split(":")
    try:
        if len(parts) == 4:
            # HH:MM:SS:ms (ms is milliseconds)
            hh, mm, ss, ms = [int(p) for p in parts]
            return float(hh) * 3600.0 + float(mm) * 60.0 + float(ss) + float(ms) / 1000.0
        if len(parts) == 3:
            # HH:MM:SS(.mmm)?
            hh = int(parts[0])
            mm = int(parts[1])
            ss = float(parts[2])
            return float(hh) * 3600.0 + float(mm) * 60.0 + ss
        if len(parts) == 2:
            # MM:SS(.mmm)?
            mm = int(parts[0])
            ss = float(parts[1])
            return float(mm) * 60.0 + ss
    except Exception:
        return None
    return None


def join_punctuated_segments(
    segments: List[Dict[str, Any]],
    join_separator: str = " ",
    paragraph_gap_seconds: float = 3.0,
    use_vi_sentence_segmenter: bool = False,
) -> str:
    """Join per-segment punctuated texts into a coherent transcript.

    Strategy:
    - Append segment by segment into a buffer, joining with a separator
    - Emit complete sentences when sentence-ending punctuation is detected
    - Keep incomplete tail to be continued by the next segment
    - Insert paragraph breaks if there is a long time gap between segments
    - Deduplicate small overlaps at joins
    """
    extractor = None
    if use_vi_sentence_segmenter:
        with suppress(Exception):
            from underthesea import sent_tokenize  # type: ignore

            def extractor(text: str) -> Tuple[List[str], str]:
                sents = [s.strip() for s in sent_tokenize(text) if s.strip()]
                if not sents:
                    return [], text
                ends_with_term = bool(re.search(r"[\.\!\?…][”\")»\]]*$", text.strip()))
                if ends_with_term:
                    return sents, ""
                return sents[:-1], sents[-1] if sents else ""
    if extractor is None:
        # Use a forward-search matcher to avoid variable-length lookbehind
        sentence_end_pattern = re.compile(r"[\.\!\?…][”\")»\]]*\s+")

        def extractor(text: str) -> Tuple[List[str], str]:
            sentences: List[str] = []
            last_cut = 0
            for m in sentence_end_pattern.finditer(text):
                end_idx = m.end()
                sent = text[last_cut:end_idx].strip()
                if sent:
                    sentences.append(sent)
                last_cut = end_idx
            tail = text[last_cut:]
            return sentences, tail

    out_sentences: List[str] = []
    buffer = ""
    last_end: Optional[float] = None

    for seg in segments:
        part = (seg.get("text_punct") or seg.get("text_raw") or "").strip()
        if not part:
            end_sec = _parse_time_to_seconds(seg.get("end"))
            last_end = end_sec if end_sec is not None else last_end
            continue

        if last_end is not None:
            start_sec = _parse_time_to_seconds(seg.get("start"))
            gap = (start_sec - last_end) if start_sec is not None else 0.0
            if gap >= paragraph_gap_seconds and buffer.strip():
                # Flush buffer fully as a paragraph
                sentences_pg, tail_pg = extractor(buffer)
                out_sentences.extend(s for s in sentences_pg if s.strip())
                if tail_pg.strip():
                    out_sentences.append(tail_pg.strip())
                out_sentences.append("")  # paragraph break marker
                buffer = ""

        # Deduplicate small overlaps
        part = _dedup_overlap(buffer, part, max_tokens=8)

        # Append to rolling buffer
        buffer = (buffer + (join_separator if buffer else "") + part).strip()
        end_sec2 = _parse_time_to_seconds(seg.get("end"))
        last_end = end_sec2 if end_sec2 is not None else last_end

        # Emit complete sentences from buffer; keep last tail incomplete
        sentences, tail = extractor(buffer)
        if sentences:
            out_sentences.extend(s for s in sentences if s.strip())
            buffer = tail

    # Flush remaining tail
    if buffer.strip():
        out_sentences.append(buffer.strip())

    # Recompose paragraphs: empty string => paragraph break
    pieces: List[str] = []
    for s in out_sentences:
        if s == "":
            if pieces and pieces[-1] != "\n\n":
                pieces.append("\n\n")
        else:
            if not pieces or pieces[-1] == "\n\n":
                pieces.append(s)
            else:
                pieces.append(" " + s)
    return "".join(pieces).strip()


def _count_tokens(llm: Any, text: str) -> int:
    try:
        tokenizer = get_tokenizer(llm)
        return int(len(tokenizer.encode(text)))
    except Exception:
        # Fallback heuristic to avoid zeros in decision logic
        return max(1, len(text) // 3)


def _split_words(text: str) -> List[str]:
    return text.strip().split()


# --- Token-preserving alignment helpers ---
_PUNCT_CHARS = set(list(".,!?…:;\"'()[]{}«»“”‘’—-"))


def _tokenize_words_and_punct(text: str) -> List[str]:
    tokens: List[str] = []
    buf: List[str] = []
    for ch in text:
        if ch.isspace():
            if buf:
                tokens.append("".join(buf))
                buf = []
            continue
        if ch in _PUNCT_CHARS:
            if buf:
                tokens.append("".join(buf))
                buf = []
            tokens.append(ch)
        else:
            buf.append(ch)
    if buf:
        tokens.append("".join(buf))
    return tokens


def _is_word(tok: str) -> bool:
    if not tok:
        return False
    if tok in _PUNCT_CHARS:
        return False
    return True


def _join_tokens_with_spacing(tokens: List[str]) -> str:
    out: List[str] = []
    for tok in tokens:
        if not out:
            out.append(tok)
            continue
        if tok in _PUNCT_CHARS:
            # Attach punctuation to previous token without extra space
            out[-1] = (out[-1] + tok)
        else:
            out.append(" " + tok)
    return "".join(out)


def _force_preserve_with_alignment(original_text: str, llm_text: str, adopt_case: bool = True) -> str:
    """Keep original words order/content; adopt punctuation and optionally case from LLM.

    Corrected behavior:
    - When LLM inserts or replaces words, prefer the LLM words (respecting adopt_case).
    - When LLM deletes words, remove those original words.
    - Always preserve punctuation coming from the LLM, mapped to the LLM word it follows.
    """
    if not original_text:
        return llm_text or ""

    orig_words = [t for t in _tokenize_words_and_punct(original_text) if _is_word(t)]
    llm_tokens = _tokenize_words_and_punct(llm_text or "")
    llm_words = [t for t in llm_tokens if _is_word(t)]

    if not orig_words:
        return llm_text or ""
    if not llm_words:
        return original_text

    # Map punctuation to the word it FOLLOWS.
    # Index -1: leading punctuation. Index i: punctuation after word i.
    punct_after: Dict[int, List[str]] = {-1: []}
    for i in range(len(llm_words)):
        punct_after[i] = []

    llw_idx = -1
    for tok in llm_tokens:
        if _is_word(tok):
            llw_idx += 1
        else:
            punct_after[llw_idx].append(tok)

    a = [w.lower() for w in orig_words]
    b = [w.lower() for w in llm_words]
    sm = SequenceMatcher(None, a, b, autojunk=False)

    result_tokens: List[str] = []
    # leading punctuation from LLM
    result_tokens.extend(punct_after.get(-1, []))

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            # words matched: preserve original content, optionally adopt LLM case
            for oi, lj in zip(range(i1, i2), range(j1, j2)):
                orig_word = orig_words[oi]
                llm_word = llm_words[lj]
                if adopt_case:
                    # use LLM casing for matched words
                    out_word = llm_word
                else:
                    out_word = orig_word
                result_tokens.append(out_word)
                result_tokens.extend(punct_after.get(lj, []))
        elif tag == "replace":
            # LLM replaced original words: prefer LLM words (respect adopt_case)
            for lj in range(j1, j2):
                llm_word = llm_words[lj]
                out_word = llm_word if adopt_case else llm_word.lower()
                result_tokens.append(out_word)
                result_tokens.extend(punct_after.get(lj, []))
        elif tag == "delete":
            # LLM deleted some original words: skip the original words (do not emit them)
            # No addition to result_tokens here.
            continue
        elif tag == "insert":
            # LLM inserted new words: emit them (respect adopt_case)
            for lj in range(j1, j2):
                llm_word = llm_words[lj]
                out_word = llm_word if adopt_case else llm_word.lower()
                result_tokens.append(out_word)
                result_tokens.extend(punct_after.get(lj, []))

    return _join_tokens_with_spacing(result_tokens).strip()


def _distribute_punct_to_segments(
    punctuated_text: str, segments: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Distribute a single punctuated text back to segments using fuzzy word alignment.

    Handles word count mismatches by using proportional distribution and fuzzy matching.
    """
    if not segments:
        return []
    if not punctuated_text.strip():
        return [{"text_punct": "", **s} for s in segments]
    
    # Get original segments with text
    text_segments = [(i, s) for i, s in enumerate(segments) if (s.get("text_raw") or "").strip()]
    if not text_segments:
        return [dict(s) for s in segments]
    
    # Create concatenated original text
    original_concat = " ".join(s.get("text_raw", "").strip() for _, s in text_segments)
    original_words = _split_words(original_concat)
    punct_words = _split_words(punctuated_text)
    
    # If word counts match exactly, use precise distribution
    if len(original_words) == len(punct_words):
        return _distribute_exact_match(punctuated_text, segments)
    
    # Handle word count mismatch with fuzzy alignment
    return _distribute_fuzzy_match(punctuated_text, segments, original_concat)


def _distribute_exact_match(punctuated_text: str, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Distribute when word counts match exactly."""
    seg_word_counts = [len(_split_words((s.get("text_raw") or "").strip())) for s in segments]
    words_punct = _split_words(punctuated_text)
    
    out_segments: List[Dict[str, Any]] = []
    cursor = 0
    for s, cnt in zip(segments, seg_word_counts):
        if cnt == 0:
            out_segments.append({**s, "text_punct": ""})
        else:
            piece_words = words_punct[cursor : cursor + cnt]
            cursor += cnt
            out_segments.append({**s, "text_punct": " ".join(piece_words).strip()})
    return out_segments


def _distribute_fuzzy_match(punctuated_text: str, segments: List[Dict[str, Any]], original_concat: str) -> List[Dict[str, Any]]:
    """Distribute using a word-level alignment between original_concat and punctuated_text.
    
    Strategy:
    - Tokenize both original_concat and punctuated_text into words and punctuation.
    - Use SequenceMatcher on lowercased word sequences to get opcodes.
    - Map LLM words (and their trailing punctuation) back to original word indices.
    - For insertions (LLM-only words), attach them to the nearest original word (prefer previous).
    - For replacements of unequal sizes, distribute LLM words across the original span proportionally.
    - Finally, assemble per-segment punctuated text by collecting mapped outputs for the original word indices that belong to each segment.
    """
    # Defensive checks
    if not original_concat:
        return [dict(s) for s in segments]
    if not punctuated_text:
        # Nothing to distribute, return empty punct fields
        out_segments = []
        for s in segments:
            out_segments.append({**s, "text_punct": ""})
        return out_segments

    # Build original word list and per-segment word counts
    orig_words = _split_words(original_concat)
    seg_word_counts: List[int] = [len(_split_words((s.get("text_raw") or "").strip())) for s in segments]

    if not orig_words:
        return [dict(s) for s in segments]

    # Tokenize LLM text into words and punctuation and map punctuation to following word index
    llm_tokens = _tokenize_words_and_punct(punctuated_text)
    llm_words = [t for t in llm_tokens if _is_word(t)]

    # Map punctuation that follows LLM words (index -1 stores leading punctuation)
    punct_after: Dict[int, List[str]] = {-1: []}
    for i in range(len(llm_words)):
        punct_after[i] = []
    llw_idx = -1
    for tok in llm_tokens:
        if _is_word(tok):
            llw_idx += 1
        else:
            # punctuation
            punct_after.setdefault(llw_idx, []).append(tok)

    # Sequence match between original words and llm words (lowercased)
    a = [w.lower() for w in orig_words]
    b = [w.lower() for w in llm_words]
    sm = SequenceMatcher(None, a, b, autojunk=False)

    # Prepare mapping: for each original word index, a list of emitted LLM word+punct strings
    mapped_per_orig: List[List[str]] = [[] for _ in range(len(orig_words))]

    # Helper to render an llm word with its punctuation
    def render_llm_word(idx: int) -> str:
        w = llm_words[idx]
        toks = [w] + punct_after.get(idx, [])
        # Join tokens with spacing rules: attach punctuation to word
        return _join_tokens_with_spacing(_tokenize_words_and_punct(" ".join(toks)))

    # Iterate opcodes and distribute LLM words
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            # one-to-one mapping
            for oi, lj in zip(range(i1, i2), range(j1, j2)):
                mapped_per_orig[oi].append(render_llm_word(lj))
        elif tag == "replace":
            # distribute LLM words across original span proportionally
            orig_span = i2 - i1
            llm_span = j2 - j1
            if orig_span <= 0:
                # treat as pure insert
                attach_idx = i1 - 1 if i1 > 0 else 0
                for lj in range(j1, j2):
                    tgt = attach_idx
                    if 0 <= tgt < len(mapped_per_orig):
                        mapped_per_orig[tgt].append(render_llm_word(lj))
            elif llm_span == 0:
                # deletion: nothing to emit for those original words
                continue
            elif orig_span == llm_span:
                for oi, lj in zip(range(i1, i2), range(j1, j2)):
                    mapped_per_orig[oi].append(render_llm_word(lj))
            else:
                # proportional distribution of ljs to oi indices
                for offset_l, lj in enumerate(range(j1, j2)):
                    # compute fractional position within llm span and map to orig index
                    frac = offset_l / max(1, llm_span)
                    oi_rel = int(frac * orig_span)
                    oi = i1 + min(orig_span - 1, oi_rel)
                    mapped_per_orig[oi].append(render_llm_word(lj))
        elif tag == "delete":
            # original words removed by LLM: leave their mapped list empty (deleted)
            continue
        elif tag == "insert":
            # LLM inserted words with no original counterpart: attach to previous original word if exists, else next
            attach_idx = i1 - 1 if i1 > 0 else i1
            # clamp attach_idx
            if attach_idx < 0:
                attach_idx = 0
            if attach_idx >= len(mapped_per_orig):
                attach_idx = len(mapped_per_orig) - 1
            for lj in range(j1, j2):
                mapped_per_orig[attach_idx].append(render_llm_word(lj))

    # Now assemble per-segment punctuated text based on original word indices
    out_segments: List[Dict[str, Any]] = []
    word_cursor = 0
    for seg_idx, s in enumerate(segments):
        raw_text = (s.get("text_raw") or "").strip()
        cnt = seg_word_counts[seg_idx]
        if cnt == 0:
            out_segments.append({**s, "text_punct": ""})
            continue
        seg_tokens: List[str] = []
        # collect mapped outputs for each original word in this segment
        for wi in range(word_cursor, min(word_cursor + cnt, len(mapped_per_orig))):
            seg_tokens.extend(mapped_per_orig[wi])
        word_cursor += cnt
        # Fallback: if nothing mapped (e.g., LLM deleted all words), leave empty
        if not seg_tokens:
            out_segments.append({**s, "text_punct": ""})
        else:
            # Join tokens ensuring proper spacing/punctuation
            # seg_tokens may already contain word+punct strings; simply join with space and normalize whitespace
            piece = " ".join(t.strip() for t in seg_tokens if t.strip()).strip()
            # Cleanup double spaces before punctuation introduced by joins
            piece = re.sub(r"\s+([,\.!\?:;…])", r"\1", piece)
            out_segments.append({**s, "text_punct": piece})
    return out_segments


def _add_basic_punctuation(text: str) -> str:
    """Add basic punctuation to text as fallback."""
    if not text:
        return ""
    
    # Capitalize first letter
    text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
    
    # Add period at end if no punctuation
    if not text[-1] in '.!?':
        text += '.'
    
    return text


# --- Character-level alignment and quality metrics ---
def _align_chars(orig_word: str, llm_word: str) -> Tuple[List[str], List[str], List[str]]:
    """Align characters between two words using SequenceMatcher.
    Returns: (orig_chars, llm_chars, tags) where tags are 'equal', 'replace', 'delete', 'insert'
    """
    sm = SequenceMatcher(None, orig_word, llm_word)
    orig_chars: List[str] = []
    llm_chars: List[str] = []
    tags: List[str] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        orig_chars.extend(list(orig_word[i1:i2]))
        llm_chars.extend(list(llm_word[j1:j2]))
        tags.extend([tag] * max(i2 - i1, j2 - j1))
    return orig_chars, llm_chars, tags


def _compute_wer(orig_words: List[str], llm_words: List[str]) -> float:
    """Compute Word Error Rate (WER) between two lists of words."""
    if not orig_words:
        return 0.0 if not llm_words else 1.0
    sm = SequenceMatcher(None, orig_words, llm_words)
    edits = 0
    insertions = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'replace':
            edits += max(i2 - i1, j2 - j1)
        elif tag == 'delete':
            edits += i2 - i1
        elif tag == 'insert':
            edits += j2 - j1
            insertions += j2 - j1
    # Use (number of edits) / (number of words in reference + number of insertions)
    # This is a common alternative WER calculation method
    denominator = len(orig_words) + insertions
    return edits / denominator if denominator > 0 else 0.0


def _compute_cer(orig_text: str, llm_text: str) -> float:
    """Compute Character Error Rate (CER) between two strings."""
    if not orig_text:
        return 0.0 if not llm_text else 1.0
    sm = SequenceMatcher(None, orig_text, llm_text)
    s = d = i = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'replace':
            s += max(i2 - i1, j2 - j1)
        elif tag == 'delete':
            d += i2 - i1
        elif tag == 'insert':
            i += j2 - j1
    n = len(orig_text)
    return (s + d + i) / n


def _compute_per(orig_text: str, llm_text: str) -> float:
    """Compute Punctuation Error Rate (PER) by comparing only punctuation marks."""
    orig_punct = [c for c in orig_text if c in _PUNCT_CHARS]
    llm_punct = [c for c in llm_text if c in _PUNCT_CHARS]
    if not orig_punct:
        return 0.0 if not llm_punct else 1.0
    sm = SequenceMatcher(None, orig_punct, llm_punct)
    s = d = i = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'replace':
            s += max(i2 - i1, j2 - j1)
        elif tag == 'delete':
            d += i2 - i1
        elif tag == 'insert':
            i += j2 - j1
    n = len(orig_punct)
    return (s + d + i) / n


def _compute_uwer_fwer(orig_text: str, llm_text: str) -> Tuple[float, float]:
    """Compute Unpunctuated WER (U-WER) and Formatted WER (F-WER)."""
    # Remove punctuation for U-WER
    orig_no_punct = ''.join(c for c in orig_text if c not in _PUNCT_CHARS)
    llm_no_punct = ''.join(c for c in llm_text if c not in _PUNCT_CHARS)
    orig_words_u = _split_words(orig_no_punct)
    llm_words_u = _split_words(llm_no_punct)
    uwer = _compute_wer(orig_words_u, llm_words_u)
    # F-WER is WER on the original (punctuated) text
    orig_words_f = _split_words(orig_text)
    llm_words_f = _split_words(llm_text)
    fwer = _compute_wer(orig_words_f, llm_words_f)
    return uwer, fwer


def _generate_human_readable_diff(orig_text: str, llm_text: str) -> str:
    """Generate a side-by-side human-readable diff with color codes."""
    sm = SequenceMatcher(None, orig_text, llm_text)
    diff_lines: List[str] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        orig_seg = orig_text[i1:i2]
        llm_seg = llm_text[j1:j2]
        if tag == 'equal':
            diff_lines.append(f"  {orig_seg}")
        elif tag == 'replace':
            diff_lines.append(f"- {orig_seg}")
            diff_lines.append(f"+ {llm_seg}")
        elif tag == 'delete':
            diff_lines.append(f"- {orig_seg}")
        elif tag == 'insert':
            diff_lines.append(f"+ {llm_seg}")
    return "\n".join(diff_lines)


def _segmentwise_punctuate_segments(
    llm: Any,
    system_prompt: str,
    max_model_len: int,
    segments: List[Dict[str, Any]],
    temperature: float = 0.0,
    batch_prompts: int = 1,
    show_progress: bool = False,
) -> List[Dict[str, Any]]:
    """Punctuate each segment independently using the LLM (batched for throughput)."""
    non_empty_indices: List[int] = [i for i, s in enumerate(segments) if (s.get("text_raw") or "").strip()]
    if not non_empty_indices:
        return [dict(s) for s in segments]
    messages: List[List[Dict[str, str]]] = []
    per_piece_max: List[int] = []
    try:
        tokenizer = get_tokenizer(llm)
    except Exception:
        tokenizer = None
    for i in non_empty_indices:
        raw = (segments[i].get("text_raw") or "").strip()
        messages.append([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": raw},
        ])
        if tokenizer is not None:
            with suppress(Exception):
                t_in = len(tokenizer.encode(raw))
                per_piece_max.append(max(64, min(int(max_model_len) - 64, int(t_in) + 128)))
                continue
        per_piece_max.append(512)
    # Run in groups
    outputs: List[str] = [""] * len(non_empty_indices)
    start = 0
    if show_progress and tqdm is not None:
        pbar = tqdm(total=len(non_empty_indices), desc="Punct segwise")
    else:
        pbar = None
    while start < len(non_empty_indices):
        end = min(start + int(max(1, batch_prompts)), len(non_empty_indices))
        group_msgs = messages[start:end]
        group_max = max(per_piece_max[start:end]) if per_piece_max else 512
        group_out = generate_chat_batch(llm, group_msgs, temperature=temperature, max_tokens=int(group_max))
        for j, text in enumerate(group_out):
            outputs[start + j] = text or ""
        if pbar is not None:
            pbar.update(len(group_msgs))
        start = end
    if pbar is not None:
        pbar.close()
    # Write back
    out_segments: List[Dict[str, Any]] = [dict(s) for s in segments]
    for k, idx in enumerate(non_empty_indices):
        out_segments[idx]["text_punct"] = (outputs[k] or "").strip()
    return out_segments


def _safe_distribute_punct_to_segments(
    punctuated_text: str,
    segments: List[Dict[str, Any]],
    *,
    llm: Optional[Any] = None,
    system_prompt: Optional[str] = None,
    max_model_len: Optional[int] = None,
    temperature: float = 0.0,
    batch_prompts: int = 1,
    show_progress: bool = False,
) -> List[Dict[str, Any]]:
    """Distribute punctuated text to segments using improved fuzzy matching.
    
    No fallback - always uses the improved distribution logic.
    """
    return _distribute_punct_to_segments(punctuated_text, segments)


def _build_segment_batches_by_token_budget(
    llm: Any,
    segments: List[Dict[str, Any]],
    token_budget: int,
    safety_margin: int = 64,
) -> List[Dict[str, Any]]:
    """Group adjacent segments into batches that fit within a token budget.

    Returns a list of dicts with keys: start_idx, end_idx (exclusive), text,
    word_counts (per original segment), start (time), end (time).
    """
    tokenizer = get_tokenizer(llm)
    batches: List[Dict[str, Any]] = []
    i = 0
    budget = max(1, int(token_budget) - int(safety_margin))
    while i < len(segments):
        j = i
        texts: List[str] = []
        word_counts: List[int] = []
        cur_tokens = 0
        while j < len(segments):
            raw = (segments[j].get("text_raw") or "").strip()
            if not raw:
                j += 1
                continue
            t = len(tokenizer.encode(raw))
            if cur_tokens and cur_tokens + t > budget:
                break
            texts.append(raw)
            word_counts.append(len(_split_words(raw)))
            cur_tokens += t
            j += 1
        if not texts:
            # Force progress to avoid infinite loop on empty texts
            j = max(j, i + 1)
        start_time = segments[i].get("start")
        end_time = segments[j - 1].get("end") if j - 1 < len(segments) else segments[-1].get("end")
        batches.append(
            {
                "start_idx": i,
                "end_idx": j,
                "text": " ".join(texts).strip(),
                "word_counts": word_counts,
                "start": start_time,
                "end": end_time,
            }
        )
        i = j
    return batches


def split_text_into_chunks(text: str, max_chars: int = 4000, overlap_sentences: int = 1) -> List[str]:
    import re
    if not text:
        return []
    # Simple sentence split on Vietnamese punctuation and newlines
    sentences = re.split(r"(?<=[\.\!\?\n])\s+", text)
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for sent in sentences:
        s = sent.strip()
        if not s:
            continue
        if current_len + len(s) + 1 > max_chars and current:
            chunks.append(" ".join(current).strip())
            # overlap last N sentences
            current = current[-overlap_sentences:] if overlap_sentences > 0 else []
            current_len = sum(len(x) + 1 for x in current)
        current.append(s)
        current_len += len(s) + 1
    if current:
        chunks.append(" ".join(current).strip())
    return chunks


def build_chunks_from_segments_by_token_budget(
    llm: Any,
    segments: List[Dict[str, Any]],
    token_budget: int,
    safety_margin: int = 64,
) -> List[str]:
    tokenizer = get_tokenizer(llm)
    chunks: List[str] = []
    current_texts: List[str] = []
    current_tokens = 0
    budget = max(1, token_budget - safety_margin)
    for seg in segments:
        raw = (seg.get("text_raw") or "").strip()
        if not raw:
            continue
        seg_tokens = len(tokenizer.encode(raw))
        if current_tokens + seg_tokens > budget and current_texts:
            chunks.append(" ".join(current_texts).strip())
            current_texts = []
            current_tokens = 0
        current_texts.append(raw)
        current_tokens += seg_tokens
    if current_texts:
        chunks.append(" ".join(current_texts).strip())
    return chunks


def summarize_long_text_map_reduce(
    llm: Any,
    text: str,
    system_prompt: str,
    temperature: float = 0.2,
    max_model_len: int = 2048,
    batch_prompts: int = 1,
    show_progress: bool = False,
) -> Dict[str, Any]:
    # Split into manageable chunks by tokenizer budget and summarize each, then reduce
    input_limit = _compute_input_token_limit(max_model_len, prompt_overhead_tokens=128, output_margin_tokens=256)
    chunks = _split_text_by_token_budget(llm, text, max_input_tokens=input_limit)
    if not chunks:
        return {"bullets": [], "abstract": ""}
    partial_bullets: List[str] = []
    partial_abstracts: List[str] = []
    if batch_prompts <= 1:
        iterable = chunks
        if show_progress and tqdm is not None:
            iterable = tqdm(iterable, total=len(chunks), desc="Summarize map")
        for chunk in iterable:
            part = summarize_text(llm, chunk, system_prompt, temperature=temperature)
            partial_bullets.extend(part.get("bullets", [])[:5])
            if part.get("abstract"):
                partial_abstracts.append(part["abstract"]) 
    else:
        list_of_messages: List[List[Dict[str, str]]] = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Hãy tóm tắt đoạn văn sau:\n\n{c}"},
            ]
            for c in chunks
        ]
        params_max = 800
        outs: List[str] = []
        start = 0
        while start < len(list_of_messages):
            end = min(start + int(max(1, batch_prompts)), len(list_of_messages))
            msgs = list_of_messages[start:end]
            outs.extend(generate_chat_batch(llm, msgs, temperature=temperature, max_tokens=params_max))
            start = end
        for content in outs:
            try:
                parsed = json.loads(content)
                pb = parsed.get("bullets", [])
                pa = parsed.get("abstract", "")
            except Exception:
                pb = [line.strip("- ") for line in content.splitlines() if line.strip()]
                pa = ""
            partial_bullets.extend(pb[:5])
            if pa:
                partial_abstracts.append(pa)
    # Reduce
    reduce_text = "\n".join([f"- {b}" for b in partial_bullets] + partial_abstracts)
    # Ensure the reduce step also respects token budget
    reduce_chunks = _split_text_by_token_budget(llm, reduce_text, max_input_tokens=input_limit)
    if len(reduce_chunks) == 1:
        reduced = summarize_text(llm, reduce_chunks[0], system_prompt, temperature=temperature)
    else:
        # Summarize each reduce chunk then merge (batched if enabled)
        merged_bullets: List[str] = []
        merged_abstracts: List[str] = []
        if batch_prompts <= 1:
            iterable = reduce_chunks
            if show_progress and tqdm is not None:
                iterable = tqdm(iterable, total=len(reduce_chunks), desc="Summarize reduce")
            for rc in iterable:
                rpart = summarize_text(llm, rc, system_prompt, temperature=temperature)
                merged_bullets.extend(rpart.get("bullets", [])[:5])
                if rpart.get("abstract"):
                    merged_abstracts.append(rpart["abstract"]) 
        else:
            list_of_messages = [
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Hãy tóm tắt đoạn văn sau:\n\n{rc}"},
                ]
                for rc in reduce_chunks
            ]
            params_max = 800
            outs2: List[str] = []
            start = 0
            while start < len(list_of_messages):
                end = min(start + int(max(1, batch_prompts)), len(list_of_messages))
                outs2.extend(generate_chat_batch(llm, list_of_messages[start:end], temperature=temperature, max_tokens=params_max))
                start = end
            for content in outs2:
                try:
                    parsed = json.loads(content)
                    mb = parsed.get("bullets", [])
                    ma = parsed.get("abstract", "")
                except Exception:
                    mb = [line.strip("- ") for line in content.splitlines() if line.strip()]
                    ma = ""
                merged_bullets.extend(mb[:5])
                if ma:
                    merged_abstracts.append(ma)
        reduced = {
            "bullets": merged_bullets[:7],
            "abstract": " ".join(merged_abstracts)[:1000],
        }
    return {
        "bullets": reduced.get("bullets", partial_bullets[:7]),
        "abstract": reduced.get("abstract", " ".join(partial_abstracts)[:1000]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-process ASR with vLLM (offline): punctuation + summary")
    parser.add_argument("--config", type=str, default="/home/cetech/omoai/config.yaml", help="Path to config.yaml")
    parser.add_argument("--asr-json", type=str, required=True, help="Path to ASR JSON")
    parser.add_argument("--model", type=str, default=None, help="HF model id for vLLM")
    parser.add_argument("--quantization", type=str, default=None, help="Quantization type (e.g., awq, gptq)")
    parser.add_argument("--max-model-len", type=int, default=None, help="Max model len (context)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=None, help="GPU memory util ratio")
    parser.add_argument("--out", type=str, required=True, help="Path to output final JSON")
    parser.add_argument("--max-num-seqs", type=int, default=None, help="Max concurrent sequences (memory)")
    parser.add_argument("--max-num-batched-tokens", type=int, default=None, help="Max batched tokens (memory)")
    parser.add_argument("--auto-outdir", action="store_true", help="Create per-input folder under paths.out_dir/{stem-YYYYMMDD-HHMMSS}; if ASR JSON already resides under paths.out_dir/<name>, reuse that folder")
    # Punctuation controls (segmented batching only)
    parser.add_argument("--punct-auto-ratio", type=float, default=None, help="Ratio of context length used to form per-batch token budget")
    parser.add_argument("--punct-auto-margin", type=int, default=None, help="Margin tokens reserved for punctuation output")
    parser.add_argument("--adopt-case", action="store_true", help="Adopt LLM capitalization when words match")
    parser.add_argument("--no-adopt-case", action="store_true", help="Disable case adoption")
    parser.add_argument("--enable-paragraphs", action="store_true", help="Enable paragraph breaks based on timing gaps")
    parser.add_argument("--no-paragraphs", action="store_true", help="Disable paragraph formatting")
    # Safety controls
    parser.add_argument("--trust-remote-code", action="store_true", help="Enable trust_remote_code for vLLM")
    parser.add_argument("--no-trust-remote-code", action="store_true", help="Disable trust_remote_code")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    # Usability/runtime controls
    parser.add_argument("--dry-run", action="store_true", help="Simulate planning; no LLM calls; still write output with metadata/decisions")
    parser.add_argument("--stream-asr", action="store_true", help="Stream ASR JSON (requires ijson) to reduce memory")
    parser.add_argument("--progress", action="store_true", help="Enable progress bars (tqdm)")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars")
    # Removed: --use-vi-sentence-seg (no longer used in segmented batching-only flow)
    parser.add_argument("--batch-prompts", type=int, default=None, help="Max prompts per batched generation call")
    args = parser.parse_args()

    # Load config
    cfg: Dict[str, Any] = {}
    try:
        import yaml  # type: ignore

        cfg_path = Path(args.config)
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
    except Exception:
        cfg = cfg or {}

    def cfg_get(path: List[str], default: Optional[Any] = None) -> Any:
        cur: Any = cfg
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                return default
            cur = cur[key]
        return cur

    # Global defaults
    model_id_from_cfg = cfg_get(["llm", "model_id"])
    if model_id_from_cfg is None:
        raise ValueError("model_id not found in configuration. Please ensure it is set in the config file under the 'llm' section.")
    model_id_default = args.model or model_id_from_cfg
    quant_default = args.quantization or cfg_get(["llm", "quantization"], None)
    mml_default = int(args.max_model_len or cfg_get(["llm", "max_model_len"], 2048))
    gmu_default = float(args.gpu_memory_utilization or cfg_get(["llm", "gpu_memory_utilization"], 0.90))
    mns_default = int(args.max_num_seqs or cfg_get(["llm", "max_num_seqs"], 1))
    mbt_default = int(args.max_num_batched_tokens or cfg_get(["llm", "max_num_batched_tokens"], 512))

    # Punctuation overrides (segmented batching only)
    p_model_id = cfg_get(["punctuation", "llm", "model_id"], model_id_default)
    p_quant = cfg_get(["punctuation", "llm", "quantization"], quant_default)
    p_mml = int(cfg_get(["punctuation", "llm", "max_model_len"], mml_default))
    p_gmu = float(cfg_get(["punctuation", "llm", "gpu_memory_utilization"], gmu_default))
    p_mns = int(cfg_get(["punctuation", "llm", "max_num_seqs"], mns_default))
    p_mbt = int(cfg_get(["punctuation", "llm", "max_num_batched_tokens"], mbt_default))
    p_temp = float(cfg_get(["punctuation", "sampling", "temperature"], 0.0))
    punct_auto_ratio = float(args.punct_auto_ratio or cfg_get(["punctuation", "auto_switch_ratio"], 0.98))
    punct_auto_margin = int(args.punct_auto_margin or cfg_get(["punctuation", "auto_margin_tokens"], 128))
    
    # Output quality controls
    adopt_case = cfg_get(["punctuation", "adopt_case"], True)
    if args.adopt_case:
        adopt_case = True
    elif args.no_adopt_case:
        adopt_case = False
    preserve_original_words = bool(cfg_get(["punctuation", "preserve_original_words"], True))
    
    enable_paragraphs = cfg_get(["punctuation", "enable_paragraphs"], True)
    if args.enable_paragraphs:
        enable_paragraphs = True
    elif args.no_paragraphs:
        enable_paragraphs = False
    
    join_sep = str(cfg_get(["punctuation", "join_separator"], " "))
    paragraph_gap = float(cfg_get(["punctuation", "paragraph_gap_seconds"], 3.0))
    use_vi_sentence_seg = bool(cfg_get(["punctuation", "use_vi_sentence_segmenter"], False))
    
    # Safety controls
    trust_remote_code = cfg_get(["llm", "trust_remote_code"], True)
    if args.trust_remote_code:
        trust_remote_code = True
    elif args.no_trust_remote_code:
        trust_remote_code = False

    # Summarization overrides (optional)
    s_model_id = cfg_get(["summarization", "llm", "model_id"], model_id_default)
    s_quant = cfg_get(["summarization", "llm", "quantization"], quant_default)
    s_mml = int(cfg_get(["summarization", "llm", "max_model_len"], mml_default))
    s_gmu = float(cfg_get(["summarization", "llm", "gpu_memory_utilization"], gmu_default))
    s_mns = int(cfg_get(["summarization", "llm", "max_num_seqs"], mns_default))
    s_mbt = int(cfg_get(["summarization", "llm", "max_num_batched_tokens"], mbt_default))
    s_temp = float(cfg_get(["summarization", "sampling", "temperature"], 0.2))

    def build_llm(model_id: str, quant: Optional[str], max_model_len: int, gmu: float, mns: int, mbt: int) -> Any:
        q = None if quant in (None, "auto", "infer", "compressed-tensors", "model") else str(quant)
        kwargs = dict(
            model=model_id,
            max_model_len=max_model_len,
            gpu_memory_utilization=gmu,
            trust_remote_code=trust_remote_code,
            max_num_seqs=mns,
            max_num_batched_tokens=mbt,
            enforce_eager=True,
        )
        if q is not None:
            kwargs["quantization"] = q
        try:
            return LLM(**kwargs)
        except Exception as e:
            if "Quantization method specified in the model config" in str(e):
                kwargs.pop("quantization", None)
                return LLM(**kwargs)
            raise

    asr_json_path = Path(args.asr_json)
    # Optional streaming path for large ASR JSON
    if args.stream_asr and ijson is not None:
        asr_top = load_asr_top_level(asr_json_path)
        seg_iter = iter_asr_segments(asr_json_path)
        asr = asr_top
        segments = list(seg_iter)
    else:
        asr = load_asr_json(asr_json_path)
        segments = asr.get("segments", [])
    transcript_raw: str = asr.get("transcript_raw", "") if isinstance(asr, dict) else ""

    # English prompts that request Vietnamese outputs
    punct_system_default = (
        "You are a Vietnamese punctuation and capitalization assistant. Task: given Vietnamese text, return the exact same words but with correct punctuation (., ?, !, ,) and sentence-case capitalization. Do not translate. Do not add, remove, or reorder any words. Output Vietnamese plain text only, no quotes, no markdown, no explanations. Respond in Vietnamese only."
    )
    sum_system_default = (
        "You are a careful summarization assistant. Summarize the input content in Vietnamese. Respond as strict JSON with keys: bullets (an array of 3-7 concise Vietnamese bullet points, each <= 20 words) and abstract (2-3 Vietnamese sentences). No extra keys, no code fences, no markdown. Do not fabricate details. Use only information from the input. Respond in Vietnamese only."
    )
    punct_system = str(cfg_get(["punctuation", "system_prompt"], punct_system_default))
    sum_system = str(cfg_get(["summarization", "system_prompt"], sum_system_default))
    # Progress and batching controls
    progress_enabled = bool(args.progress)
    if args.no_progress:
        progress_enabled = False
    verbose = bool(args.verbose)
    prompt_batch_prompts = int(args.batch_prompts) if args.batch_prompts is not None else int(p_mns)
    
    if verbose:
        print(f"[post] Config: adopt_case={adopt_case}, enable_paragraphs={enable_paragraphs}")
        print(f"[post] Token budget: ratio={punct_auto_ratio}, margin={punct_auto_margin}")
        print(f"[post] Trust remote code: {trust_remote_code}")

    # Early dry-run path: compute decisions without instantiating LLM
    if bool(args.dry_run):
        # Segmented batching only: estimate batches by approximate per-segment tokens
        approx_tokens = max(1, len((transcript_raw or "").strip()) // 3)
        ratio = max(0.5, min(1.0, float(punct_auto_ratio)))
        token_limit = int(ratio * int(p_mml)) - int(punct_auto_margin)
        token_limit = max(256, token_limit)

        seg_tokens: List[int] = []
        for seg in segments:
            raw = (seg.get("text_raw") or "").strip()
            if not raw:
                continue
            seg_tokens.append(max(1, len(raw) // 3))
        est_batches = 0
        cur = 0
        for t in seg_tokens:
            if cur and cur + t > token_limit:
                est_batches += 1
                cur = 0
            cur += t
        if seg_tokens:
            est_batches += 1

        # Summarization decision
        auto_ratio = float(cfg_get(["summarization", "auto_switch_ratio"], 0.98))
        margin_tokens = int(cfg_get(["summarization", "auto_margin_tokens"], 64))
        ratio_limit = max(0.5, min(1.0, auto_ratio))
        s_token_limit = int(ratio_limit * int(s_mml)) - margin_tokens
        approx_sum_tokens = approx_tokens
        sum_mode = "map_reduce" if (approx_sum_tokens and approx_sum_tokens > s_token_limit or bool(cfg_get(["summarization", "map_reduce"], False))) else "single"

        decisions: Dict[str, Any] = {
            "punctuation_mode": "segments",
            "segments_count": len(segments),
            "estimated_batches": est_batches,
            "summarization_mode": sum_mode,
        }

        final = dict(asr)
        final.update(
            {
                "segments": segments,
                "transcript_punct": (asr.get("transcript_raw") or "") if isinstance(asr, dict) else "",
                "summary": {"bullets": [], "abstract": ""},
                "metadata": {
                    **(asr.get("metadata", {}) if isinstance(asr, dict) else {}),
                    "llm_model_punctuation": p_model_id,
                    "llm_model_summarization": s_model_id,
                    "llm_offline": True,
                    "dry_run": True,
                    "decisions": decisions,
                },
            }
        )
        out_path = Path(args.out)
        save_json(out_path, final)
        return

    # Stage 1: punctuation LLM instance and segmented-batching processing
    llm_punc = build_llm(p_model_id, p_quant, p_mml, p_gmu, p_mns, p_mbt)
    punct_segments: List[Dict[str, Any]]
    transcript_punct: str

    # Segmented punctuation with batching by token budget
    ratio = max(0.5, min(1.0, float(punct_auto_ratio)))
    token_limit = int(ratio * p_mml) - int(punct_auto_margin)
    token_limit = max(256, token_limit)
    if verbose:
        print(f"[post] Punctuation: token_limit={token_limit}, segments={len(segments)}")
    batches = _build_segment_batches_by_token_budget(llm_punc, segments, token_limit, safety_margin=64)
    if verbose:
        print(f"[post] Created {len(batches)} batches for punctuation")
    # Initialize output copy
    punct_segments = [dict(s) for s in segments]
    chunk_puncts: List[str] = []
    # Batched generation across batch texts for throughput
    non_empty_batches = [b for b in batches if b["text"]]
    list_of_messages: List[List[Dict[str, str]]] = [
        [
            {"role": "system", "content": punct_system},
            {"role": "user", "content": b["text"]},
        ]
        for b in non_empty_batches
    ]
    try:
        tokenizer = get_tokenizer(llm_punc)
        est_in = [len(tokenizer.encode(b["text"])) for b in non_empty_batches]
    except Exception:
        est_in = [max(64, len(b["text"]) // 3) for b in non_empty_batches]
    est_out = [max(64, min(int(p_mml) - 64, i + 128)) for i in est_in]
    generated_texts: List[str] = []
    start_idx = 0
    if tqdm is not None and progress_enabled:
        pbar = tqdm(total=len(non_empty_batches), desc="Punct batches")
    else:
        pbar = None
    while start_idx < len(list_of_messages):
        end_idx = min(start_idx + int(max(1, prompt_batch_prompts)), len(list_of_messages))
        group_msgs = list_of_messages[start_idx:end_idx]
        group_max = max(est_out[start_idx:end_idx]) if est_out else 512
        generated = generate_chat_batch(llm_punc, group_msgs, temperature=p_temp, max_tokens=int(group_max))
        generated_texts.extend(generated)
        if pbar is not None:
            pbar.update(len(group_msgs))
        start_idx = end_idx
    if pbar is not None:
        pbar.close()
    gi = 0
    for b in batches:
        if not b["text"]:
            continue
        chunk_punct = generated_texts[gi]
        gi += 1
        sub_segments = segments[b["start_idx"] : b["end_idx"]]
        # Optionally preserve original words or allow LLM to rewrite (for spelling/clean-up)
        if preserve_original_words:
            original_concat = " ".join((s.get("text_raw") or "").strip() for s in sub_segments).strip()
            aligned_chunk = _force_preserve_with_alignment(original_concat, chunk_punct or "", adopt_case=adopt_case)
        else:
            aligned_chunk = (chunk_punct or "").strip()
        if aligned_chunk:
            chunk_puncts.append(aligned_chunk)
        # Distribute with LLM-aware fallback to fill per-segment punctuation if strict slicing fails
        distributed = _safe_distribute_punct_to_segments(
            aligned_chunk,
            sub_segments,
            llm=llm_punc,
            system_prompt=punct_system,
            max_model_len=p_mml,
            temperature=p_temp,
            batch_prompts=prompt_batch_prompts,
            show_progress=False,
        )
        # Write back
        for k, seg_out in enumerate(distributed):
            punct_segments[b["start_idx"] + k]["text_punct"] = seg_out.get("text_punct", "")
    # Build transcript from aggregated chunk outputs with deduplication
    if not chunk_puncts:
        transcript_punct = ""
    else:
        transcript_punct = chunk_puncts[0].strip()
        for cp in chunk_puncts[1:]:
            if cp:
                deduped = _dedup_overlap(transcript_punct, cp.strip(), max_tokens=8)
                transcript_punct = (transcript_punct + " " + deduped).strip()

    # Apply paragraph formatting if enabled
    if enable_paragraphs:
        # Re-process segments through paragraph formatter for better flow
        transcript_punct = join_punctuated_segments(
            punct_segments,
            join_separator=join_sep,
            paragraph_gap_seconds=paragraph_gap,
            use_vi_sentence_segmenter=use_vi_sentence_seg,
        )

    # Keep llm_punc for reuse if settings match summarization

    # 2) Summarize
    long_text = transcript_punct
    # Auto-switch logic: single-pass unless use_map_reduce is forced, or total tokens exceed ratio*context
    use_map_reduce = bool(cfg_get(["summarization", "map_reduce"], False))
    # Stage 2: summarization LLM instance (reuse if identical settings)
    reuse = (
        s_model_id == p_model_id
        and (str(s_quant) == str(p_quant) or (s_quant in (None, "auto") and p_quant in (None, "auto")))
        and s_mml == p_mml
        and abs(float(s_gmu) - float(p_gmu)) < 1e-6
        and s_mns == p_mns
        and s_mbt == p_mbt
    )
    if reuse:
        llm_sum = llm_punc
    else:
        # Free punctuation engine VRAM before building a new model to avoid OOM
        with suppress(Exception):
            del llm_punc
        # Only clear cache if debug flag is set or switching models (higher chance of OOM)
        if DEBUG_EMPTY_CACHE or not reuse:
            with suppress(Exception):
                torch.cuda.empty_cache()  # type: ignore[attr-defined]
        gc.collect()
        llm_sum = build_llm(s_model_id, s_quant, s_mml, s_gmu, s_mns, s_mbt)
    try:
        tokenizer_s = get_tokenizer(llm_sum)
        total_tokens_s = len(tokenizer_s.encode(long_text))
    except Exception:
        tokenizer_s = None
        total_tokens_s = 0

    # Allow configurable ratio threshold for auto-switch
    auto_ratio = float(cfg_get(["summarization", "auto_switch_ratio"], 0.98))
    margin_tokens = int(cfg_get(["summarization", "auto_margin_tokens"], 64))
    ratio_limit = max(0.5, min(1.0, auto_ratio))  # clamp for safety
    token_limit = int(ratio_limit * s_mml) - margin_tokens

    if not use_map_reduce and tokenizer_s is not None and total_tokens_s and total_tokens_s <= token_limit:
        if verbose:
            print(f"[post] Summarization: single-pass, tokens={total_tokens_s}")
        summary = summarize_text(llm_sum, long_text, sum_system, temperature=s_temp)
    else:
        if verbose:
            print(f"[post] Summarization: map-reduce, tokens={total_tokens_s}")
        summary = summarize_long_text_map_reduce(
            llm_sum,
            long_text,
            sum_system,
            temperature=s_temp,
            max_model_len=s_mml,
            batch_prompts=prompt_batch_prompts,
            show_progress=(tqdm is not None and progress_enabled),
        )
    # Optionally free engines (process exits anyway)
    if not reuse:
        with suppress(Exception):
            del llm_sum
    with suppress(Exception):
        del llm_punc
    # Only clear cache at end if debug flag is set
    if DEBUG_EMPTY_CACHE:
        with suppress(Exception):
            torch.cuda.empty_cache()  # type: ignore[attr-defined]
    gc.collect()

    # Consistency check: validate all non-empty text_raw have non-empty text_punct
    non_empty_raw = sum(1 for s in punct_segments if (s.get("text_raw") or "").strip())
    non_empty_punct = sum(1 for s in punct_segments if (s.get("text_punct") or "").strip())
    punct_coverage = non_empty_punct / max(1, non_empty_raw)
    
    # Count punctuation marks for quality metrics
    punct_marks = len([c for c in transcript_punct if c in ".,!?…"])
    punct_density = punct_marks / max(1, len(transcript_punct))
    
    if verbose:
        print(f"[post] Quality metrics: coverage={punct_coverage:.4f}, density={punct_density:.4f}, marks={punct_marks}")
        print(f"[post] Summary: bullets={len(summary.get('bullets', []))}, has_abstract={bool(summary.get('abstract'))}")
    
    # Compose final output with enhanced metadata
    decisions = {
        "punctuation_mode": "segments",
        "segments_count": len(segments),
        "batches_used": len(batches),
        "punct_coverage": round(punct_coverage, 4),
        "punct_marks": punct_marks,
        "punct_density": round(punct_density, 4),
        "adopt_case": adopt_case,
        "enable_paragraphs": enable_paragraphs,
        "summarization_mode": "map_reduce" if use_map_reduce else "single",
    }
    
    final = dict(asr)
    final.update(
        {
            "segments": punct_segments,
            "transcript_punct": transcript_punct,
            "summary": summary,
            "metadata": {
                **asr.get("metadata", {}),
                "llm_model_punctuation": p_model_id,
                "llm_model_summarization": s_model_id,
                "llm_offline": True,
                "decisions": decisions,
            },
        }
    )
    # Auto output dir per input file, if requested
    out_path = Path(args.out)
    if args.auto_outdir:
        from datetime import datetime, timezone
        # If ASR JSON already lives under paths.out_dir/<name>/asr.json, reuse that folder
        base_root = Path(str(cfg_get(["paths", "out_dir"], "data/output")))
        try:
            asr_parent = asr_json_path.resolve().parent
        except Exception:
            asr_parent = asr_json_path.parent
        if base_root in asr_parent.parents or asr_parent == base_root:
            base_dir = asr_parent
        else:
            # Otherwise, create a new folder based on input audio stem + timestamp
            audio_path_in = (asr.get("audio", {}) or {}).get("path")
            stem = Path(audio_path_in).stem if audio_path_in else asr_json_path.stem
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            candidate = base_root / f"{stem}-{timestamp}"
            if candidate.exists():
                suffix = 1
                while (base_root / f"{stem}-{timestamp}-{suffix:02d}").exists():
                    suffix += 1
                candidate = base_root / f"{stem}-{timestamp}-{suffix:02d}"
            candidate.mkdir(parents=True, exist_ok=True)
            base_dir = candidate
        out_path = base_dir / "final.json"

        # Optional separate files per config
        if bool(cfg_get(["output", "write_separate_files"], False)):
            # transcript
            tf_name = str(cfg_get(["output", "transcript_file"], "transcript.txt"))
            sf_name = str(cfg_get(["output", "summary_file"], "summary.txt"))
            # transcript_punct plain text with optional wrapping
            with open(base_dir / tf_name, "w", encoding="utf-8") as tf:
                tf_text = (final.get("transcript_punct") or "").strip()
                wrap_width_cfg = cfg_get(["output", "wrap_width"], 100)
                try:
                    wrap_width = int(wrap_width_cfg) if wrap_width_cfg is not None else 0
                except Exception:
                    wrap_width = 0
                if wrap_width and wrap_width > 0:
                    paragraphs = tf_text.split("\n\n") if tf_text else []
                    wrapped_paragraphs: List[str] = []
                    for para in paragraphs:
                        p = para.strip()
                        if not p:
                            wrapped_paragraphs.append("")
                            continue
                        wrapped_paragraphs.append(
                            textwrap.fill(
                                p,
                                width=wrap_width,
                                replace_whitespace=False,
                                break_long_words=False,
                                break_on_hyphens=False,
                            )
                        )
                    tf_text = "\n\n".join(wrapped_paragraphs).strip()
                tf.write(tf_text + "\n")
            # summary plain text (render bullets + abstract)
            summary = final.get("summary") or {}
            bullets = summary.get("bullets") or []
            abstract = summary.get("abstract") or ""
            with open(base_dir / sf_name, "w", encoding="utf-8") as sf:
                for b in bullets:
                    sf.write(f"- {b}\n")
                if bullets and abstract:
                    sf.write("\n")
                if abstract:
                    sf.write(abstract.strip() + "\n")

    save_json(out_path, final)


if __name__ == "__main__":
    main()
