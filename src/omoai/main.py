from __future__ import annotations

import argparse
from pathlib import Path
import os
import sys
from datetime import datetime, timezone
 

from omoai.api.services import run_preprocess_script
from omoai.api.scripts.asr_wrapper import run_asr_script
from omoai.api.scripts.postprocess_wrapper import run_postprocess_script

def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "config.yaml").exists() or (parent / ".git").exists():
            return parent
    return here.parent

def _default_config_path() -> Path:
    # Prefer explicit CLI, then OMOAI_CONFIG env var, then project-relative config
    env_cfg = os.environ.get("OMOAI_CONFIG")
    return Path(env_cfg) if env_cfg else (_repo_root() / "config.yaml")

def run_pipeline(audio_path: Path, out_dir: Path, model_dir: Path | None, config_path: Path | None) -> int:
    cfg_path = config_path or _default_config_path()

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Preprocess to 16kHz mono PCM16 WAV (script-based)
    preprocessed = out_dir / "preprocessed.wav"
    try:
        preprocessed.parent.mkdir(parents=True, exist_ok=True)
        run_preprocess_script(input_path=audio_path, output_path=preprocessed)
    except Exception as e:
        print(f"[orchestrator] ffmpeg failed: {e}")
        return 1

    # 2) ASR (script-based)
    asr_json = out_dir / "asr.json"
    try:
        run_asr_script(audio_path=preprocessed, output_path=asr_json, config_path=cfg_path)
    except Exception as e:
        print(f"[orchestrator] ASR failed: {e}")
        return 2

    # Attempt to free GPU VRAM before launching post-process (vLLM)
    try:
        import gc  # type: ignore
        import torch  # type: ignore
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    # 3) Post-process (punctuation + summary) via script-based pipeline
    final_json = out_dir / "final.json"
    try:
        # Call the script-based postprocess (scripts/post.py) via wrapper
        run_postprocess_script(asr_json_path=asr_json, output_path=final_json, config_path=cfg_path)

        # Read final JSON produced by the script and present as output
        import json
        with open(final_json, "r", encoding="utf-8") as f:
            final_data = json.load(f)

        # Optionally, reformat or log as needed. The script output is expected to contain:
        # { "segments": [...], "summary": {...}, "metadata": {...} }
    except Exception as e:
        print(f"[orchestrator] Post-process failed: {e}")
        return 3

    print(f"[orchestrator] Completed. Output: {final_json}")
    return 0

def main() -> None:
    
    parser = argparse.ArgumentParser(description="OMOAI: preprocess → ASR → punctuation+summary")
    parser.add_argument("audio", type=str, nargs="?", default=None, help="Path to input audio file")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory (auto-create if missing)")
    parser.add_argument("--model-dir", type=str, default=None, help="ChunkFormer checkpoint directory")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml (or set OMOAI_CONFIG)")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    args = parser.parse_args()

    if args.interactive or not args.audio:
        from omoai.interactive_cli import run_interactive_cli
        run_interactive_cli()
        return

    audio_path = Path(args.audio)
    # Resolve config path with env fallback
    cfg_path = Path(args.config) if args.config else _default_config_path()

    # Load out_dir root from config.yaml if available
    base_out_root = None
    try:
        import yaml  # type: ignore

        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg_data = yaml.safe_load(f) or {}
            paths = cfg_data.get("paths", {}) if isinstance(cfg_data, dict) else {}
            out_root_cfg = paths.get("out_dir") if isinstance(paths, dict) else None
            if out_root_cfg:
                base_out_root = Path(str(out_root_cfg))
    except Exception:
        base_out_root = None

    # Determine output directory: CLI overrides, else config paths.out_dir/<stem-YYYYMMDD-HHMMSS>, else repo default
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        default_root = base_out_root if base_out_root else (_repo_root() / "data/output")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        out_dir = Path(default_root) / f"{audio_path.stem}-{timestamp}"

    model_dir = Path(args.model_dir) if args.model_dir else None

    rc = run_pipeline(audio_path, out_dir, model_dir, cfg_path)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
