from __future__ import annotations

import argparse
from pathlib import Path
import os
# Ensure TORCH_CUDA_ARCH_LIST is set before importing CUDA-related modules
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'
import sys
from datetime import datetime, timezone
import yaml

from omoai.pipeline.preprocess import preprocess_file_to_wav_bytes
from omoai.pipeline.asr import run_asr_inference
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
        # Load preprocessed WAV into a torch tensor (float32). Try soundfile first, fall back to wave.
        try:
            import soundfile as sf
            import numpy as np
            import torch
            data, sr = sf.read(str(preprocessed), dtype='float32')
            if data.ndim > 1:
                data = data.mean(axis=1)
            audio_tensor = torch.from_numpy(data).float()
        except Exception:
            try:
                import wave
                import numpy as np
                import torch
                with wave.open(str(preprocessed), 'rb') as wf:
                    frames = wf.readframes(wf.getnframes())
                    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                    if wf.getnchannels() > 1:
                        audio = audio.reshape(-1, wf.getnchannels()).mean(axis=1)
                    audio_tensor = torch.from_numpy(audio).float()
            except Exception as e:
                raise RuntimeError(f"Failed to load preprocessed WAV into tensor: {e}")
        # run_asr_inference expects a tensor; sample rate is 16000 for preprocessed audio
        asr_result = run_asr_inference(
            audio_input=audio_tensor,
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
            },
            sample_rate=16000,
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

    # 3) Post-process (punctuation + summary) via script-based pipeline
    final_json = out_dir / "final.json"
    try:
        # Call the script-based postprocess (scripts/post.py) via wrapper
        run_postprocess_script(asr_json_path=asr_json, output_path=final_json, config_path=None)

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