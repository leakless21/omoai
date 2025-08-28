# OMOAI — Audio Processing Pipeline (ASR + Punctuation + Summarization)

A production-ready pipeline to transcribe and summarize long-form audio (e.g., podcasts). It uses the `Chunkformer` model for ASR and an LLM (via vLLM) for punctuation, capitalization, and summarization.

## Features

- **High-quality ASR**: Chunkformer-based decoding with timestamped segments.
- **Long-audio support**: Chunked decoding for multi-hour recordings.
- **Punctuation & capitalization**: LLM adds punctuation and sentence casing; preserves original word order by default.
- **Summarization**: Bullet points and abstract (single-pass or map-reduce for long texts).
- **Config-driven**: Single `config.yaml` controls paths, ASR, LLM, API, and outputs.
- **Outputs**: Always writes `final.json`; optionally writes transcript and summary text files. Programmable formatters available for JSON/Text/SRT/VTT/Markdown.
- **CLI & REST API**: Run locally or as a web service.

## Architecture

1. **Preprocess**: Convert input audio to 16kHz mono PCM16 WAV (ffmpeg).
2. **ASR**: Transcribe with Chunkformer, produce segments and raw transcript (`asr.json`).
3. **Post-process**: LLM adds punctuation and generates summary; write `final.json` and optional text files.

See details in `docs/architecture/index.md`.

## Requirements

- Python 3.11+
- Linux/macOS/Windows
- `ffmpeg`
- NVIDIA GPU with CUDA (recommended) for ASR and vLLM

Tip: verify `ffmpeg` with `ffmpeg -version`.

## Installation

1. Clone and enter the repo

```bash
git clone <repository-url>
cd <repository-directory>
```

1. Create a virtual env with uv and install

```bash
uv venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv pip install -e .
```

1. Prepare models

- Place the Chunkformer checkpoint at `models/chunkformer/chunkformer-large-vie` (or update `paths.chunkformer_checkpoint` in `config.yaml`).
- Ensure `paths.chunkformer_dir` points to `./src/chunkformer` (default) or your local Chunkformer source.

## Configuration

All settings are in `config.yaml`. Key sections:

```yaml
paths:
  chunkformer_dir: ./src/chunkformer
  chunkformer_checkpoint: ./models/chunkformer/chunkformer-large-vie
  out_dir: ./data/output

asr:
  total_batch_duration_s: 1800
  chunk_size: 64
  left_context_size: 128
  right_context_size: 128
  device: auto         # auto -> cuda if available else cpu
  autocast_dtype: fp16 # on CUDA

llm:
  model_id: cpatonn/Qwen3-4B-Instruct-2507-AWQ-4bit
  quantization: auto
  max_model_len: 50000
  gpu_memory_utilization: 0.90
  max_num_seqs: 2
  max_num_batched_tokens: 512
  trust_remote_code: true  # Only enable for trusted models

punctuation:
  llm: { ... }             # Overrides; inherits from llm when omitted
  preserve_original_words: true
  auto_switch_ratio: 0.98
  auto_margin_tokens: 128
  adopt_case: true
  enable_paragraphs: true
  join_separator: " "
  paragraph_gap_seconds: 3.0
  system_prompt: |
    <instruction>
    You are a Vietnamese text punctuation engine...
    </instruction>

summarization:
  llm: { ... }             # Overrides; inherits from llm when omitted
  map_reduce: false
  auto_switch_ratio: 0.98
  auto_margin_tokens: 256
  system_prompt: |
    <instruction>
    You are a Vietnamese text analysis engine...
    </instruction>

# Legacy output controls used by scripts/post.py
write_separate_files: true
transcript_file: transcript.txt
summary_file: summary.txt
wrap_width: 0

# Optional structured output config for library usage
formats: ["json", "text", "srt", "vtt", "md"]
transcript:
  include_raw: true
  include_punct: true
  include_segments: true
  timestamps: clock
  wrap_width: 100
summary:
  mode: both
  bullets_max: 7
  abstract_max_chars: 1000
  language: vi
final_json: final.json

api:
  host: 0.0.0.0
  port: 8000
  max_body_size_mb: 100
  request_timeout_seconds: 300
  temp_dir: /tmp
  cleanup_temp_files: true
  enable_progress_output: true
```

- Use `OMOAI_CONFIG=/abs/path/to/config.yaml` to override config location at runtime.

## Quickstart (CLI)

Run the full pipeline on an audio file:

```bash
uv run start data/input/your_audio.mp3
# or
uv run python src/omoai/main.py data/input/your_audio.mp3
```

Interactive mode:

```bash
uv run interactive
# or
uv run python src/omoai/main.py --interactive
```

Outputs are written under `paths.out_dir/<audio-stem>-<UTC-timestamp>/`, e.g.:

```text
data/output/my_episode-20250101-120000/
├── preprocessed.wav
├── asr.json
├── final.json
├── transcript.txt      # if write_separate_files: true
└── summary.txt         # if write_separate_files: true
```

### Stage-by-stage scripts

```bash
# 1) Preprocess
uv run python scripts/preprocess.py --input data/input/a.mp3 --output data/output/a.wav

# 2) ASR (writes asr.json; --auto-outdir creates a per-run folder)
uv run python scripts/asr.py --config config.yaml --audio data/output/a.wav \
  --model-dir models/chunkformer/chunkformer-large-vie --out data/output/asr.json --auto-outdir

# 3) Post-process (writes final.json; reuses ASR folder when --auto-outdir)
uv run python scripts/post.py --config config.yaml --asr-json data/output/.../asr.json \
  --out data/output/final.json --auto-outdir
```

## REST API

Start the server:

```bash
uv run api
# or
uv run litestar --app src.omoai.api.app:app run --host 0.0.0.0 --port 8000
```

Endpoints:

- `POST /pipeline` (multipart form): run full pipeline.
- `POST /preprocess` (multipart form): preprocess only.
- `POST /asr` (JSON): run ASR on a local preprocessed path.
- `POST /postprocess` (JSON): run punctuation+summary on provided ASR output.
- `GET /health`: health check (ffmpeg, config, scripts).

Examples:

```bash
# Full pipeline (default plain text response with transcript + summary)
curl -X POST 'http://localhost:8000/pipeline' \
  -F 'audio_file=@data/input/audio.mp3'

# Structured JSON with options
curl -X POST 'http://localhost:8000/pipeline?include=segments&ts=clock&summary=both&summary_bullets_max=5' \
  -F 'audio_file=@data/input/audio.mp3'

# Health
curl 'http://localhost:8000/health'
```

Notes:

- When no query params are provided to `/pipeline`, the server returns `text/plain` containing the punctuated transcript and summary for convenience.
- With query params, the server returns structured JSON (`PipelineResponse`).

## Outputs and formatters

The pipeline always writes `final.json`. With `write_separate_files: true`, it also writes `transcript.txt` and `summary.txt`.

Advanced, programmatic formatting is available via the output writer and plugins (JSON/Text/SRT/VTT/Markdown). Example:

```python
from pathlib import Path
from omoai.output.writer import write_outputs
from omoai.config.schemas import OutputConfig

config = OutputConfig(formats=["json", "text", "srt", "vtt", "md"])
written = write_outputs(
    output_dir=Path("data/output/my_run"),
    segments=segments,                 # list of {start, end, text_raw, text_punct}
    transcript_raw=transcript_raw,
    transcript_punct=transcript_punct,
    summary=summary,                   # {bullets: [...], abstract: "..."}
    metadata=metadata,                 # optional dict
    config=config,
)
print(written)
```

## Troubleshooting

- **ffmpeg not found**: Install via your OS package manager and ensure it is on PATH.
- **Model checkpoint missing**: Set `paths.chunkformer_checkpoint` to a valid local path.
- **CUDA OOM / memory issues**:
  - Lower `llm.max_num_seqs` and `llm.max_num_batched_tokens`.
  - Reduce `asr.total_batch_duration_s` or `gpu_memory_utilization`.
  - Use quantized models (e.g., AWQ) or set `quantization: auto`.
- **Security**: `trust_remote_code: true` executes remote code from model repos; only enable for trusted sources.
- **Custom config location**: `export OMOAI_CONFIG=/abs/path/config.yaml`.
- **Debug GPU cache**: set `OMOAI_DEBUG_EMPTY_CACHE=true` to hint clearing CUDA cache in some paths.

## Testing

```bash
uv run pytest
```


## Project structure

```text
.
├── config.yaml
├── data/
├── models/
├── scripts/
│   ├── preprocess.py
│   ├── asr.py
│   └── post.py
├── src/
│   ├── chunkformer/
│   └── omoai/
│       ├── api/
│       ├── output/
│       │   ├── formatter.py
│       │   ├── writer.py
│       │   └── plugins/  # json, text, srt, vtt, md
│       ├── config/
│       ├── interactive_cli.py
│       └── main.py
└── tests/
```

## Documentation

- `docs/architecture/index.md`
- `docs/user_guide/configuration.md`
- `docs/development/best_practices.md`
- `docs/project/requirements.md`
- `docs/README.md`

## License

MIT

---

## OMOAI — Hướng dẫn bằng tiếng Việt

Một pipeline xử lý âm thanh để nhận dạng giọng nói (ASR), chấm câu, và tóm tắt nội dung. Sử dụng `Chunkformer` cho ASR và LLM (vLLM) để chấm câu và tóm tắt.

### Tính năng

- **ASR chất lượng cao** với dấu thời gian theo đoạn.
- **Hỗ trợ audio dài** (hàng giờ) bằng giải mã theo khối.
- **Chấm câu & viết hoa** bằng LLM, giữ nguyên thứ tự từ gốc theo mặc định.
- **Tóm tắt**: gạch đầu dòng và đoạn tóm tắt (đơn hoặc map-reduce cho văn bản dài).
- **Cấu hình tập trung** trong `config.yaml`.
- **Đầu ra**: luôn có `final.json`; tùy chọn xuất `transcript.txt`, `summary.txt`. Có thể tạo SRT/VTT/Markdown qua API thư viện.
- **CLI & REST API**.

### Yêu cầu

- Python 3.11+
- `ffmpeg`
- GPU NVIDIA (khuyến nghị) cho tốc độ tốt hơn

### Cài đặt

```bash
git clone <repository-url>
cd <repository-directory>
uv venv && source .venv/bin/activate
uv pip install -e .
```

Chuẩn bị mô hình:

- Đặt checkpoint Chunkformer tại `models/chunkformer/chunkformer-large-vie` hoặc chỉnh `paths.chunkformer_checkpoint`.

### Cấu hình (`config.yaml`)

- Chỉnh `paths.chunkformer_dir`, `paths.chunkformer_checkpoint`, `paths.out_dir`.
- `asr.device: auto` sẽ ưu tiên CUDA nếu có.
- `llm.trust_remote_code: true` chỉ bật khi tin cậy nguồn mô hình.
- Có thể đặt biến môi trường `OMOAI_CONFIG` để trỏ đến file cấu hình khác.

### Chạy nhanh (CLI)

```bash
uv run start data/input/audio.mp3
# hoặc
uv run python src/omoai/main.py data/input/audio.mp3
```

Chế độ tương tác:

```bash
uv run interactive
```

Thư mục kết quả nằm trong `data/output/<tên-file>-<thời-gian-UTC>/`.

### REST API (Tiếng Việt)

Khởi động server:

```bash
uv run api
```

Gửi yêu cầu toàn bộ pipeline:

```bash
curl -X POST 'http://localhost:8000/pipeline' -F 'audio_file=@data/input/audio.mp3'
```

Trả về mặc định là văn bản (`text/plain`). Để nhận JSON có cấu trúc và điều khiển đầu ra:

```bash
curl -X POST 'http://localhost:8000/pipeline?include=segments&ts=clock&summary=both&summary_bullets_max=5' \
  -F 'audio_file=@data/input/audio.mp3'
```

Sức khỏe hệ thống:

```bash
curl 'http://localhost:8000/health'
```

### Đầu ra

- Luôn có `final.json`.
- Nếu `write_separate_files: true`, có thêm `transcript.txt`, `summary.txt`.
- Có thể tạo SRT/VTT/Markdown bằng API thư viện (`omoai.output.writer`).

### Khắc phục sự cố

- **Thiếu ffmpeg**: cài đặt và đảm bảo có trong PATH.
- **Thiếu checkpoint mô hình**: cập nhật `paths.chunkformer_checkpoint`.
- **Lỗi bộ nhớ/CUDA OOM**:
  - Giảm `llm.max_num_seqs`, `llm.max_num_batched_tokens`.
  - Giảm `asr.total_batch_duration_s` hoặc `gpu_memory_utilization`.
  - Dùng mô hình đã lượng tử hóa (AWQ), `quantization: auto`.
- **Bảo mật**: chỉ bật `trust_remote_code` khi chắc chắn về nguồn mô hình.

### Kiểm thử

```bash
uv run pytest
```