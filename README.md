# OMOAI — Audio Processing Pipeline (ASR + Punctuation + Summarization)

A production-ready pipeline to transcribe and summarize long-form audio, with a primary focus on Vietnamese. It uses the `Chunkformer` model for ASR and an LLM (via vLLM) for punctuation, capitalization, and summarization.

## Features

- **High-quality ASR**: Chunkformer-based decoding with word-level timestamps through forced alignment.
- **Long-audio support**: Chunked decoding for multi-hour recordings.
- **Advanced Punctuation & Capitalization**: LLM-based punctuation and capitalization.
- **Summarization**: Abstractive and extractive summarization with timestamps.
- **Config-driven**: A single `config.yaml` controls all aspects of the pipeline.
- **Multiple Output Formats**: Supports JSON, plain text, SRT, and VTT.
- **CLI & REST API**: Can be run locally from the command line or as a web service.
- **Voice Activity Detection (VAD)**: Optional VAD pre-segmentation to skip silence and improve performance on long audio files.

## Architecture

1.  **Preprocessing**: Converts input audio to 16kHz mono PCM16 WAV using `ffmpeg`.
2.  **VAD (Optional)**: `silero-vad` or `webrtcvad` splits the audio into speech segments.
3.  **ASR**: Transcribes the audio with ChunkFormer, producing a raw transcript with segment-level timestamps.
4.  **Alignment**: A wav2vec2 model is used to generate word-level timestamps.
5.  **Post-processing**: An LLM adds punctuation and generates a summary.

For a more detailed explanation of the architecture, see the [Architecture Documentation](docs/ARCHITECTURE.md).

## Requirements

- Python 3.11+
- `ffmpeg`
- NVIDIA GPU with CUDA (recommended for ASR and vLLM)

## Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  Create a virtual environment and install the dependencies:
    ```bash
    uv venv
    source .venv/bin/activate
    uv pip install -e .
    ```

3.  Prepare the models:
    *   Place the Chunkformer checkpoint at `models/chunkformer/chunkformer-large-vie` (or update `paths.chunkformer_checkpoint` in `config.yaml`).

## Configuration

All settings are in `config.yaml`. For a detailed explanation of all the configuration options, please refer to the [Component: Configuration](docs/COMPONENT_CONFIGURATION_DOCS.md) documentation.

## Quickstart (CLI)

Run the full pipeline on an audio file:

```bash
uv run start data/input/your_audio.mp3
```

This will create a new directory in `data/output` containing the results of the pipeline.

## REST API

Start the server:

```bash
uv run api
```

The API will be available at `http://localhost:8000`.

**Endpoints:**

*   `POST /v1/pipeline`: Run the full pipeline on an audio file.
*   `GET /v1/health`: Health check endpoint.
*   `GET /v1/metrics`: Prometheus metrics endpoint.

For more details on the API, please refer to the [Component: API](docs/COMPONENT_API_DOCS.md) documentation.

## Troubleshooting

-   **`ffmpeg` not found**: Install `ffmpeg` using your OS package manager and ensure it is in your `PATH`.
-   **Model checkpoint missing**: Make sure the `paths.chunkformer_checkpoint` in `config.yaml` points to a valid model checkpoint.
-   **CUDA OOM / memory issues**:
    *   Lower `llm.max_num_seqs` and `llm.max_num_batched_tokens`.
    *   Reduce `asr.total_batch_duration_s` or `gpu_memory_utilization`.
    *   Use quantized models (e.g., AWQ) or set `quantization: auto`.

## Documentation

For detailed documentation on all components of the system, please see the [docs/](docs/) directory.

## License

MIT

---

## OMOAI — Hướng dẫn bằng tiếng Việt

Một pipeline xử lý âm thanh để nhận dạng giọng nói (ASR), chấm câu, và tóm tắt nội dung, tập trung vào tiếng Việt.

### Tính năng

-   **ASR chất lượng cao**: Nhận dạng giọng nói với dấu thời gian ở cấp độ từ.
-   **Hỗ trợ audio dài**: Xử lý các file âm thanh dài hàng giờ.
-   **Chấm câu & viết hoa nâng cao**: Tự động chấm câu và viết hoa bằng LLM.
-   **Tóm tắt**: Tóm tắt nội dung văn bản với dấu thời gian.
-   **Cấu hình linh hoạt**: Mọi thứ được điều khiển qua file `config.yaml`.
-   **Nhiều định dạng đầu ra**: Hỗ trợ JSON, text, SRT, và VTT.
-   **CLI & REST API**: Chạy trực tiếp từ dòng lệnh hoặc như một dịch vụ web.
-   **Phát hiện giọng nói (VAD)**: Tùy chọn VAD để bỏ qua khoảng lặng và tăng tốc độ xử lý.

### Yêu cầu

-   Python 3.11+
-   `ffmpeg`
-   GPU NVIDIA với CUDA (khuyến nghị)

### Cài đặt

```bash
git clone <repository-url>
cd <repository-directory>
uv venv && source .venv/bin/activate
uv pip install -e .
```

### Chạy nhanh (CLI)

```bash
uv run start data/input/audio.mp3
```

### REST API

Khởi động server:

```bash
uv run api
```

Xem chi tiết các endpoint trong [Component: API](docs/COMPONENT_API_DOCS.md).