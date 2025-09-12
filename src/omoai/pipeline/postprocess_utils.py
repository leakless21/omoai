"""
Lightweight loader wrappers for utilities originally in scripts/post.py.

These wrappers avoid inserting the scripts directory into sys.path by loading
the script module directly from its file location. They expose a small set of
functions used by the pipeline:

- apply_chat_template
- generate_chat
- generate_chat_batch
- join_punctuated_segments
- _segmentwise_punctuate_segments
- _safe_distribute_punct_to_segments

If a packaged implementation is added later (preferred), these wrappers can be
migrated to call the packaged functions directly.
"""
from __future__ import annotations

from pathlib import Path
import importlib.util
import types
from typing import Any, Dict, List, Optional

_scripts_post_mod: Optional[types.ModuleType] = None

def _load_scripts_post() -> types.ModuleType:
    global _scripts_post_mod
    if _scripts_post_mod is not None:
        return _scripts_post_mod

    # Resolve scripts/post.py relative to the repository root:
    # src/omoai/pipeline/postprocess_utils.py -> up 3 levels -> repo root
    script_path = Path(__file__).resolve().parents[3] / "scripts" / "post.py"
    if not script_path.exists():
        raise ImportError(f"scripts/post.py not found at expected location: {script_path}")

    spec = importlib.util.spec_from_file_location("scripts.post", str(script_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for scripts.post at {script_path}")

    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)  # type: ignore
    except Exception as e:
        raise ImportError(f"Failed to execute scripts.post module: {e}") from e

    _scripts_post_mod = mod
    return mod

# Exposed wrappers -----------------------------------------------------------

def apply_chat_template(llm: Any, messages: List[Dict[str, str]]) -> str:
    mod = _load_scripts_post()
    fn = getattr(mod, "apply_chat_template", None)
    if fn is None:
        raise ImportError("apply_chat_template not available in scripts/post.py")
    return fn(llm, messages)

def generate_chat(llm: Any, messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> str:
    mod = _load_scripts_post()
    fn = getattr(mod, "generate_chat", None)
    if fn is None:
        raise ImportError("generate_chat not available in scripts/post.py")
    return fn(llm, messages, temperature, max_tokens)

def generate_chat_batch(llm: Any, list_of_messages: List[List[Dict[str, str]]], temperature: float, max_tokens: int) -> List[str]:
    mod = _load_scripts_post()
    fn = getattr(mod, "generate_chat_batch", None)
    if fn is None:
        raise ImportError("generate_chat_batch not available in scripts/post.py")
    return fn(llm, list_of_messages, temperature, max_tokens)

def join_punctuated_segments(segments: List[Dict[str, Any]], join_separator: str = " ", paragraph_gap_seconds: float = 3.0, use_vi_sentence_segmenter: bool = False) -> str:
    mod = _load_scripts_post()
    fn = getattr(mod, "join_punctuated_segments", None)
    if fn is None:
        raise ImportError("join_punctuated_segments not available in scripts/post.py")
    return fn(segments, join_separator=join_separator, paragraph_gap_seconds=paragraph_gap_seconds, use_vi_sentence_segmenter=use_vi_sentence_segmenter)

def _segmentwise_punctuate_segments(*args, **kwargs):
    mod = _load_scripts_post()
    fn = getattr(mod, "_segmentwise_punctuate_segments", None)
    if fn is None:
        raise ImportError("_segmentwise_punctuate_segments not available in scripts/post.py")
    return fn(*args, **kwargs)

def _safe_distribute_punct_to_segments(*args, **kwargs):
    mod = _load_scripts_post()
    fn = getattr(mod, "_safe_distribute_punct_to_segments", None)
    if fn is None:
        raise ImportError("_safe_distribute_punct_to_segments not available in scripts/post.py")
    return fn(*args, **kwargs)