from __future__ import annotations

import argparse
from pathlib import Path
import os
import sys
from datetime import datetime, timezone
import yaml

from omoai.pipeline.preprocess import preprocess_file_to_wav_bytes
from omoai.pipeline.asr import run_asr_inference
from omoai.pipeline.postprocess import postprocess_transcript

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
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Preprocess to 16kHz mono PCM16 WAV
    preprocessed = out_dir / "preprocessed.wav"
    try:
        wav_bytes = preprocess_file_to_wav_bytes(audio_path)
        preprocessed.parent.mkdir(parents=True, exist_ok=True)
        with open(preprocessed, 'wb') as f:
            f.write(wav_bytes)
    except Exception as e:
        print(f"[orchestrator] ffmpeg failed: {e}")
        return 1

    # 2) ASR
    asr_json = out_dir / "asr.json"
    asr_config = config.get("asr", {})
    paths_config = config.get("paths", {})
    model_checkpoint = model_dir or Path(paths_config.get("chunkformer_checkpoint"))

    try:
        # Run ASR inference using the new API
        asr_result = run_asr_inference(
            audio_input=preprocessed,
            config={
                "asr": {
                    "chunk_size": asr_config.get("chunk_size", 64),
                    "left_context_size": asr_config.get("left_context_size", 128),
                    "right_context_size": asr_config.get("right_context_size", 128),
                    "total_batch_duration_s": asr_config.get("total_batch_duration_s", 1800),
                    "autocast_dtype": asr_config.get("autocast_dtype", "fp16"),
                    "device": asr_config.get("device", "auto"),
                },
                "paths": {
                    "chunkformer_checkpoint": str(model_checkpoint),
                }
            }
        )
        
        # Save ASR result to JSON file
        asr_output = {
            "audio": {
                "sr": asr_result.sample_rate,
                "path": str(preprocessed.resolve()),
                "duration_s": asr_result.audio_duration_seconds,
            },
            "segments": [
                {
                    "start": seg.start,
                    "end": seg.end,
                    "text_raw": seg.text,
                }
                for seg in asr_result.segments
            ],
            "transcript_raw": asr_result.transcript,
            "metadata": asr_result.metadata,
        }
        
        asr_json.parent.mkdir(parents=True, exist_ok=True)
        with open(asr_json, "w", encoding="utf-8") as f:
            import json
            json.dump(asr_output, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[orchestrator] ASR failed: {e}")
        return 2

    # 3) Post-process (punctuation + summary)
    final_json = out_dir / "final.json"
    try:
        # Load ASR result
        import json
        with open(asr_json, "r", encoding="utf-8") as f:
            asr_data = json.load(f)
        
        # Convert to ASRResult format
        from omoai.pipeline.asr import ASRResult, ASRSegment
        segments = [
            ASRSegment(
                start=seg.get("start", 0.0),
                end=seg.get("end", 0.0),
                text=seg.get("text_raw", ""),
            )
            for seg in asr_data.get("segments", [])
        ]
        
        asr_result = ASRResult(
            segments=segments,
            transcript=asr_data.get("transcript_raw", ""),
            audio_duration_seconds=asr_data.get("audio", {}).get("duration_s", 0.0),
            sample_rate=asr_data.get("audio", {}).get("sr", 16000),
            metadata=asr_data.get("metadata", {}),
        )
        
        # Run postprocessing using the new API
        post_result = postprocess_transcript(
            asr_result=asr_result,
            config=config,
        )
        
        # Save final result to JSON file
        final_output = {
            "segments": [
                {
                    "start": seg.start,
                    "end": seg.end,
                    "text_raw": asr_result.segments[i].text if i < len(asr_result.segments) else "",
                    "text_punct": seg.text,
                }
                for i, seg in enumerate(post_result.segments)
            ],
            "transcript_raw": asr_result.transcript,
            "transcript_punct": post_result.transcript_punctuated,
            "summary": {
                "bullets": post_result.summary.bullets,
                "abstract": post_result.summary.abstract,
            },
            "metadata": {
                **asr_result.metadata,
                **post_result.metadata,
            },
        }
        
        final_json.parent.mkdir(parents=True, exist_ok=True)
        with open(final_json, "w", encoding="utf-8") as f:
            json.dump(final_output, f, ensure_ascii=False, indent=2)
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