import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from litestar import Controller, post
from litestar.datastructures import State
from pydub import AudioSegment  # type: ignore

from src.omoai.api.models import ASRRequest, ASRResponse


class ASRModel:
    """Singleton class to hold the ASR model and configuration."""

    _instance = None
    _is_initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ASRModel, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._is_initialized:
            self.model = None
            self.char_dict = None
            self.config = {}
            self._is_initialized = True

    def initialize(self, config_path: Path = Path("/home/cetech/omoai/config.yaml")):
        """Initialize the ASR model with configuration."""
        # Load configuration
        try:
            import yaml  # type: ignore
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {}

        # Store configuration
        self.config = {
            "chunkformer_dir": cfg.get("paths", {}).get("chunkformer_dir", "/home/cetech/omoai/chunkformer"),
            "model_checkpoint": cfg.get("paths", {}).get("chunkformer_checkpoint", ""),
            "total_batch_duration_s": cfg.get("asr", {}).get("total_batch_duration_s", 1800),
            "chunk_size": cfg.get("asr", {}).get("chunk_size", 64),
            "left_context_size": cfg.get("asr", {}).get("left_context_size", 128),
            "right_context_size": cfg.get("asr", {}).get("right_context_size", 128),
            "device": cfg.get("asr", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu"),
            "autocast_dtype": cfg.get("asr", {}).get("autocast_dtype", "fp16" if torch.cuda.is_available() else None),
        }

        # Add chunkformer to path
        chunkformer_dir = Path(self.config["chunkformer_dir"])
        if str(chunkformer_dir) not in sys.path:
            sys.path.insert(0, str(chunkformer_dir))

        # Import and initialize model
        from omoai.chunkformer import decode as cfdecode  # type: ignore

        self.model, self.char_dict = cfdecode.init(str(self.config["model_checkpoint"]), self.config["device"])

        print(f"ASR Model initialized with device: {self.config['device']}")

    def process_audio(self, audio_path: Path) -> Dict[str, Any]:
        """Process audio file and return transcript and segments."""
        if not self.model:
            raise RuntimeError("ASR model not initialized")

        # Load and standardize audio to 16kHz mono PCM16
        audio = AudioSegment.from_file(str(audio_path))
        audio = audio.set_frame_rate(16000).set_sample_width(2).set_channels(1)
        audio_duration_s: float = len(audio) / 1000.0
        waveform = torch.as_tensor(audio.get_array_of_samples(), dtype=torch.float32).unsqueeze(0)

        # Extract log-mel filterbank features (Kaldi fbank) like decode.py
        import torchaudio.compliance.kaldi as kaldi  # type: ignore
        xs = kaldi.fbank(
            waveform,
            num_mel_bins=80,
            frame_length=25,
            frame_shift=10,
            dither=0.0,
            energy_floor=0.0,
            sample_frequency=16000,
        ).unsqueeze(0)

        # Compute internal parameters in the same way as decode.py
        subsampling_factor = self.model.encoder.embed.subsampling_factor
        conv_lorder = self.model.encoder.cnn_module_kernel // 2

        # Maximum duration (seconds) the GPU can handle in one batch
        max_length_limited_context = self.config["total_batch_duration_s"]
        max_length_limited_context = int((max_length_limited_context // 0.01)) // 2  # in 10ms

        multiply_n = max_length_limited_context // self.config["chunk_size"] // subsampling_factor
        truncated_context_size = self.config["chunk_size"] * multiply_n

        # Relative right context in frames
        def get_max_input_context(c: int, r: int, n: int) -> int:
            return r + max(c, r) * (n - 1)

        rel_right_context_size = get_max_input_context(
            self.config["chunk_size"], max(self.config["right_context_size"], conv_lorder), self.model.encoder.num_blocks
        )
        rel_right_context_size = rel_right_context_size * subsampling_factor

        # Prepare caches
        device = torch.device(self.config["device"])
        offset = torch.zeros(1, dtype=torch.int, device=device)
        att_cache = torch.zeros(
            (
                self.model.encoder.num_blocks,
                self.config["left_context_size"],
                self.model.encoder.attention_heads,
                self.model.encoder._output_size * 2 // self.model.encoder.attention_heads,
            )
        ).to(device)
        cnn_cache = torch.zeros(
            (self.model.encoder.num_blocks, self.model.encoder._output_size, conv_lorder)
        ).to(device)

        hyps: List[torch.Tensor] = []

        # Autocast dtype mapping
        dtype_map = {
            None: None,
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }
        amp_dtype = dtype_map.get(self.config["autocast_dtype"], None)

        from contextlib import nullcontext
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
                ) = self.model.encoder.forward_parallel_chunk(
                    xs=x,
                    xs_origin_lens=x_len,
                    chunk_size=self.config["chunk_size"],
                    left_context_size=self.config["left_context_size"],
                    right_context_size=self.config["right_context_size"],
                    att_cache=att_cache,
                    cnn_cache=cnn_cache,
                    truncated_context_size=truncated_context_size,
                    offset=offset,
                )

                encoder_outs = encoder_outs.reshape(1, -1, encoder_outs.shape[-1])[:, :encoder_lens]
                if (
                    self.config["chunk_size"] * multiply_n * subsampling_factor * idx + rel_right_context_size
                    < xs.shape[1]
                ):
                    # exclude the output of relative right context
                    encoder_outs = encoder_outs[:, :truncated_context_size]

                offset = offset - encoder_lens + encoder_outs.shape[1]

                hyp = self.model.encoder.ctc_forward(encoder_outs).squeeze(0)
                hyps.append(hyp)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                if (
                    self.config["chunk_size"] * multiply_n * subsampling_factor * idx + rel_right_context_size
                    >= xs.shape[1]
                ):
                    break

        if len(hyps) == 0:
            segments: List[Dict[str, Any]] = []
            transcript_raw = ""
        else:
            hyps_cat = torch.cat(hyps)
            from omoai.chunkformer.model.utils.ctc_utils import (
                get_output_with_timestamps,
            )  # type: ignore
            decode = get_output_with_timestamps([hyps_cat], self.char_dict)[0]
            segments = [
                {"start": item["start"], "end": item["end"], "text_raw": item["decode"]}
                for item in decode
            ]
            transcript_raw = " ".join(seg["text_raw"].strip() for seg in segments if seg["text_raw"]).replace("  ", " ").strip()

        return {
            "audio": {"sr": 16000, "path": str(audio_path.resolve()), "duration_s": audio_duration_s},
            "segments": segments,
            "transcript_raw": transcript_raw,
        }


from src.omoai.api.services import asr_service


class ASRController(Controller):
    path = "/asr"

    @post("/")
    async def asr(self, data: ASRRequest, state: State) -> ASRResponse:
        """Process ASR request using the refactored service function."""
        return asr_service(data)