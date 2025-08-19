import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


def ensure_chunkformer_on_path(chunkformer_dir: Path) -> None:
    if str(chunkformer_dir) not in sys.path:
        sys.path.insert(0, str(chunkformer_dir))


def run_asr(
    audio_path: Path,
    model_checkpoint: Path,
    out_path: Path,
    total_batch_duration: int,
    chunk_size: int,
    left_context_size: int,
    right_context_size: int,
    device_str: str,
    autocast_dtype: str | None,
    chunkformer_dir: Path,
) -> None:
    ensure_chunkformer_on_path(chunkformer_dir)

    # Local imports after sys.path adjustment
    from omoai.chunkformer import decode as cfdecode  # type: ignore
    import torchaudio.compliance.kaldi as kaldi  # type: ignore
    from omoai.chunkformer.model.utils.ctc_utils import (
        get_output_with_timestamps,
    )  # type: ignore
    from contextlib import nullcontext
    from pydub import AudioSegment  # type: ignore

    device = torch.device(device_str)
    dtype_map = {
        None: None,
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }
    amp_dtype = dtype_map.get(autocast_dtype, None)

    # Initialize model and char dict
    model, char_dict = cfdecode.init(str(model_checkpoint), device)

    # Compute internal parameters in the same way as decode.py
    subsampling_factor = model.encoder.embed.subsampling_factor
    conv_lorder = model.encoder.cnn_module_kernel // 2

    # Maximum duration (seconds) the GPU can handle in one batch
    max_length_limited_context = total_batch_duration
    max_length_limited_context = int((max_length_limited_context // 0.01)) // 2  # in 10ms

    multiply_n = max_length_limited_context // chunk_size // subsampling_factor
    truncated_context_size = chunk_size * multiply_n

    # Relative right context in frames
    def get_max_input_context(c: int, r: int, n: int) -> int:
        return r + max(c, r) * (n - 1)

    rel_right_context_size = get_max_input_context(
        chunk_size, max(right_context_size, conv_lorder), model.encoder.num_blocks
    )
    rel_right_context_size = rel_right_context_size * subsampling_factor

    # Load and standardize audio to 16kHz mono PCM16
    audio = AudioSegment.from_file(str(audio_path))
    audio = audio.set_frame_rate(16000).set_sample_width(2).set_channels(1)
    audio_duration_s: float = len(audio) / 1000.0
    waveform = torch.as_tensor(audio.get_array_of_samples(), dtype=torch.float32).unsqueeze(0)

    # Extract log-mel filterbank features (Kaldi fbank) like decode.py
    xs = kaldi.fbank(
        waveform,
        num_mel_bins=80,
        frame_length=25,
        frame_shift=10,
        dither=0.0,
        energy_floor=0.0,
        sample_frequency=16000,
    ).unsqueeze(0)

    # Prepare caches
    offset = torch.zeros(1, dtype=torch.int, device=device)
    att_cache = torch.zeros(
        (
            model.encoder.num_blocks,
            left_context_size,
            model.encoder.attention_heads,
            model.encoder._output_size * 2 // model.encoder.attention_heads,
        )
    ).to(device)
    cnn_cache = torch.zeros(
        (model.encoder.num_blocks, model.encoder._output_size, conv_lorder)
    ).to(device)

    hyps: List[torch.Tensor] = []
    ctx = torch.autocast(device.type, amp_dtype) if amp_dtype is not None else nullcontext()
    with torch.no_grad(), ctx:
        for idx, _ in enumerate(range(0, xs.shape[1], truncated_context_size * subsampling_factor)):
            start = max(truncated_context_size * subsampling_factor * idx, 0)
            end = min(
                truncated_context_size * subsampling_factor * (idx + 1) + 7,
                xs.shape[1],
            )

            x = xs[:, start : end + rel_right_context_size]
            x_len = torch.tensor([x[0].shape[0]], dtype=torch.int).to(device)

            (
                encoder_outs,
                encoder_lens,
                _,
                att_cache,
                cnn_cache,
                offset,
            ) = model.encoder.forward_parallel_chunk(
                xs=x,
                xs_origin_lens=x_len,
                chunk_size=chunk_size,
                left_context_size=left_context_size,
                right_context_size=right_context_size,
                att_cache=att_cache,
                cnn_cache=cnn_cache,
                truncated_context_size=truncated_context_size,
                offset=offset,
            )

            encoder_outs = encoder_outs.reshape(1, -1, encoder_outs.shape[-1])[:, :encoder_lens]
            if (
                chunk_size * multiply_n * subsampling_factor * idx + rel_right_context_size
                < xs.shape[1]
            ):
                # exclude the output of relative right context
                encoder_outs = encoder_outs[:, :truncated_context_size]

            offset = offset - encoder_lens + encoder_outs.shape[1]

            hyp = model.encoder.ctc_forward(encoder_outs).squeeze(0)
            hyps.append(hyp)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if (
                chunk_size * multiply_n * subsampling_factor * idx + rel_right_context_size
                >= xs.shape[1]
            ):
                break

    if len(hyps) == 0:
        segments: List[Dict[str, Any]] = []
        transcript_raw = ""
    else:
        hyps_cat = torch.cat(hyps)
        decode = get_output_with_timestamps([hyps_cat], char_dict)[0]
        segments = [
            {"start": item["start"], "end": item["end"], "text_raw": item["decode"]}
            for item in decode
        ]
        transcript_raw = " ".join(seg["text_raw"].strip() for seg in segments if seg["text_raw"])\
            .replace("  ", " ").strip()

    output: Dict[str, Any] = {
        "audio": {"sr": 16000, "path": str(audio_path.resolve()), "duration_s": audio_duration_s},
        "segments": segments,
        "transcript_raw": transcript_raw,
        "metadata": {
            "asr_model": str(model_checkpoint),
            "params": {
                "total_batch_duration": total_batch_duration,
                "chunk_size": chunk_size,
                "left_context_size": left_context_size,
                "right_context_size": right_context_size,
                "autocast_dtype": autocast_dtype or "none",
            },
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="ASR wrapper for ChunkFormer (JSON output)")
    parser.add_argument("--config", type=str, default="/home/cetech/omoai/config.yaml", help="Path to config.yaml")
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio file")
    parser.add_argument(
        "--model-dir",
        type=str,
        required=False,
        help="Path to local HuggingFace checkpoint repo (ChunkFormer)",
    )
    parser.add_argument("--out", type=str, required=True, help="Path to output JSON")
    parser.add_argument("--auto-outdir", action="store_true", help="Create per-input folder under paths.out_dir/{stem-YYYYMMDD-HHMMSS}")
    parser.add_argument(
        "--total-batch-duration",
        type=int,
        default=1800,
        help="Total audio duration per batch in seconds (default 1800)",
    )
    parser.add_argument("--chunk-size", type=int, default=64, help="Chunk size (default 64)")
    parser.add_argument(
        "--left-context-size", type=int, default=128, help="Left context size (default 128)"
    )
    parser.add_argument(
        "--right-context-size", type=int, default=128, help="Right context size (default 128)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the model on (cuda/cpu)",
    )
    parser.add_argument(
        "--autocast-dtype",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        default="fp16" if torch.cuda.is_available() else None,
        help="Autocast dtype (default fp16 on CUDA)",
    )
    parser.add_argument(
        "--chunkformer-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "chunkformer"),
        help="Path to chunkformer source directory",
    )

    args = parser.parse_args()
    # Load config yaml if available
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

    # Resolve parameters with precedence: CLI flag -> config -> fallback default
    model_dir = args.model_dir or cfg_get(["paths", "chunkformer_checkpoint"], None)
    if not model_dir:
        raise SystemExit("Missing --model-dir and paths.chunkformer_checkpoint in config.yaml")

    # Auto output dir per input file, if requested
    out_path = Path(args.out)
    if args.auto_outdir:
        from datetime import datetime, timezone
        stem = Path(args.audio).stem
        base_root = Path(str(cfg_get(["paths", "out_dir"], "data/output")))
        # Timestamp-based folder name, UTC for stability
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        candidate = base_root / f"{stem}-{timestamp}"
        # Avoid rare collisions within the same second by suffixing a counter
        if candidate.exists():
            suffix = 1
            while (base_root / f"{stem}-{timestamp}-{suffix:02d}").exists():
                suffix += 1
            candidate = base_root / f"{stem}-{timestamp}-{suffix:02d}"
        candidate.mkdir(parents=True, exist_ok=True)
        base_dir = candidate
        out_path = base_dir / "asr.json"

    run_asr(
        audio_path=Path(args.audio),
        model_checkpoint=Path(model_dir),
        out_path=out_path,
        total_batch_duration=int(
            args.total_batch_duration or cfg_get(["asr", "total_batch_duration_s"], 1800)
        ),
        chunk_size=int(args.chunk_size or cfg_get(["asr", "chunk_size"], 64)),
        left_context_size=int(args.left_context_size or cfg_get(["asr", "left_context_size"], 128)),
        right_context_size=int(args.right_context_size or cfg_get(["asr", "right_context_size"], 128)),
        device_str=str(args.device or cfg_get(["asr", "device"], "cuda")),
        autocast_dtype=(args.autocast_dtype or cfg_get(["asr", "autocast_dtype"], None)),
        chunkformer_dir=Path(args.chunkformer_dir or cfg_get(["paths", "chunkformer_dir"], str(Path(__file__).resolve().parents[1] / "chunkformer"))),
    )


if __name__ == "__main__":
    main()


