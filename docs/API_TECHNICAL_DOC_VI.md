# Tài liệu kỹ thuật API — omoai

Tài liệu này mô tả cấu trúc, cách sử dụng API, các endpoint, tham số và các lưu ý vận hành cho dự án `omoai` (trong workspace hiện tại). Nội dung được sắp xếp theo mẫu bạn cung cấp, điều chỉnh để khớp với cấu trúc thực tế của mã nguồn.

## 1. Giới thiệu ngắn

Repo này cung cấp một API server (Litestar) cho pipeline xử lý âm thanh: preprocess -> ASR -> postprocess (punctuation, summarization). API triển khai các controller dưới `src/omoai/api` và dùng script wrappers (trong `src/omoai/api/scripts`) để thực thi các bước bằng công cụ có sẵn (ffmpeg, scripts.asr, scripts.post).

## 2. Các thành phần chính

- `src/omoai/api/app.py`: Tạo và cấu hình ứng dụng Litestar (`create_app`, `main`). Đăng ký middleware (request id, metrics, timeout) và exception handlers.

- `src/omoai/api/main_controller.py`: Endpoint chính `/v1/pipeline` (POST) cho việc chạy toàn bộ pipeline; hỗ trợ các query parameters để chọn format/summary/options.

- `src/omoai/api/preprocess_controller.py`: Endpoint `/v1/preprocess` (POST) để chỉ chạy bước preprocess (trả về `output_path`).

- `src/omoai/api/asr_controller.py`: Endpoint `/v1/asr` (POST) để gọi ASR trên file đã được preprocess.

- `src/omoai/api/postprocess_controller.py`: Endpoint `/v1/postprocess` (POST) để thực hiện post-processing (punctuation, summarization).

- `src/omoai/api/jobs.py`: Quản lý job bất đồng bộ (enqueue pipeline jobs). Endpoint `/v1/jobs/{job_id}` (GET) kiểm tra trạng thái job.

- `src/omoai/api/health.py`: Endpoint `/v1/health` (GET) kiểm tra trạng thái dịch vụ và phụ thuộc (ffmpeg, config file...).

- `src/omoai/api/metrics_middleware.py`: Middleware đơn giản thu thập metrics và endpoint `/v1/metrics` trả về metrics ở dạng text (PROM-like).

- `src/omoai/api/models.py`: Định nghĩa Pydantic models cho request/response (`PipelineRequest`, `OutputFormatParams`, `PipelineResponse`,...).

- `src/omoai/api/services.py`: Triển khai logic core (script-based): `run_full_pipeline`, `preprocess_audio_service`, `asr_service`, `postprocess_service`. Làm việc chủ yếu bằng cách gọi các wrappers trong `src/omoai/api/scripts`.

- `src/omoai/api/scripts/`: Chứa wrapper gọi các script gốc (ASR, postprocess). Các script thực thi ffmpeg, mô-đun ASR và bước postprocess (punctuation, LLM summary).

- `config.yaml` / `src/omoai/config/schemas.py` (loader): cấu hình mặc định dùng trong `get_config()` — chứa các tùy chọn như `api.host`, `api.port`, `api.temp_dir`, `output` và `logging`.

- `data/` (dự án): chứa `input/` và `output/` mặc định, nơi các file tạm và kết quả có thể được lưu.

## 3. Phần cứng & khuyến nghị

- Huấn luyện / fine-tune / inference tốc độ cao: GPU NVIDIA tương thích với phiên bản PyTorch được cài (xem `pyproject.toml`/`requirements`).
- Inference trên CPU có thể chạy nhưng chậm. Để tối ưu trên CPU có thể xuất model sang ONNX hoặc dùng quantization.
- RAM: Ít nhất 16GB cho môi trường phát triển; môi trường production tùy theo khối lượng request có thể cần nhiều hơn.

## 4. Phần mềm & dependencies

- Python 3.10+
- Các packages chính (từ `pyproject.toml`): `litestar[standard]`, `pydantic`, `torch` (nếu dùng model PyTorch), `pandas` (nếu cần đọc metadata), `uvicorn`/`uv` wrapper, v.v.
- (Tùy chọn) `k2` nếu sử dụng chức năng training hoặc ASR nâng cao.
- (Tùy chọn) `onnxruntime` để chạy ONNX trên CPU.

Cài đặt ví dụ:

pip install -r requirements.txt

Hoặc theo `pyproject.toml` / môi trường được khuyến nghị trong README.

## 5. Cách cấu hình & cài đặt nhanh

1. Clone repo và vào thư mục:

   git clone <repo-url>
   cd omoai

2. Tạo virtualenv và kích hoạt:

   python3 -m venv .venv
   source .venv/bin/activate

3. Cài dependencies:

   pip install -r requirements.txt

4. (Tùy chọn) Cài `k2` theo hướng dẫn nếu cần training.

5. Cấu hình nếu cần: chỉnh `config.yaml` (tên model, temp dirs, api.host, api.port, logging...) hoặc đặt biến môi trường `OMOAI_CONFIG` trỏ tới file config khác.

6. Chạy API server (development):

   # repo có scrip `uv` wrapper trong README; nếu không có, dùng uvicorn
   uv run litestar --app omoai.api.app:create_app run --host 0.0.0.0 --port 8000

   # hoặc dùng uvicorn trực tiếp
   uvicorn omoai.api.app:create_app --factory --host 0.0.0.0 --port 8000

Lưu ý: nếu dùng lệnh `uv` wrapper trong README, hãy chắc rằng package `uv` đã được cài vào virtualenv. Nếu không, dùng `uvicorn` như phương án thay thế.

## 6. Sử dụng CLI (ví dụ nhanh)

Repo có một số script CLI trong `src/omoai/api/scripts` và các entrypoints cũ. Tuy repository hiện không có một package `zipvoice` như mẫu, ví dụ lệnh chung để gọi script CLI (nếu tồn tại) sẽ giống:

python -m omoai.api.scripts.asr_wrapper --input prompt.wav --output out.json

Tuy nhiên hiện API được thiết kế để gọi qua HTTP (Litestar) hơn là CLI. Nếu bạn muốn tích hợp CLI, tìm `scripts/` và xem docstring của các wrapper để biết các tham số.

## 7. Sử dụng API — endpoint & ví dụ

Mặc định server khởi chạy theo `config.api.host` và `config.api.port` (thường `http://localhost:8000` với `create_app()` đăng ký router tại `/v1`). Vì vậy endpoints thực tế bắt đầu với `/v1`.

Base URL: http://localhost:8000/v1

7.1 Health

Endpoint: GET /v1/health
Response (200 OK):

{
  "status": "healthy",
  "details": { ... }
}

7.2 Metrics

Endpoint: GET /v1/metrics
Response: text/plain (Prometheus-like metrics)
Example body:

# HELP request_total Total HTTP requests
# TYPE request_total counter
request_total 42
# HELP request_latency_seconds_sum Cumulative request latency
# TYPE request_latency_seconds_sum counter
request_latency_seconds_sum 1.234567

7.3 Full pipeline — upload audio and run (synchronous)

Endpoint: POST /v1/pipeline
Content-Type: multipart/form-data

Required form-data fields (per `PipelineRequest` model):
- audio_file: file (.wav or other audio)
- (optional) other model_config fields if provided in request model

Query parameters (via URL) supported (see `OutputFormatParams` in `src/omoai/api/models.py`):
- formats: list of strings (json, text, srt, vtt, md)
- include: list of strings (transcript_raw, transcript_punct, segments, timestamped_summary)
- ts: timestamp format (none, s, ms, clock)
- summary: (bullets, abstract, both, none)
- summary_bullets_max: int
- summary_lang: str
- include_quality_metrics: bool
- include_diffs: bool
- return_summary_raw: bool
- return_timestamped_summary_raw: bool
- include_vad: bool
- async_: bool (if true, job is queued and a 202 with job_id is returned)

Example curl (synchronous):

curl -X POST "http://localhost:8000/v1/pipeline" \
  -F "audio_file=@prompt.wav" \
  -F "model_config={}\"some\":\"value\"};type=application/json" \
  -H "Accept: application/json" \
  --output response.json

Response (200 OK): JSON matching `PipelineResponse` model by default, or audio/text depending on Accept header and configuration.

- If Accept: text/plain (or query formats=text) server can return a plain text composed of punctuated transcript and summary.
- For JSON, response body shape (partial):
  {
    "summary": {"title":"...","abstract":"...","bullets":[...]},
    "segments": [...],
    "transcript_punct": "...",
    "transcript_raw": "..." (only when included),
    "quality_metrics": {...} (optional),
    "diffs": {...} (optional)
  }

Async usage (enqueue job):
- Append query param `async_=true` (or form field `async_`) to POST /v1/pipeline. The controller will read audio content and call `job_manager.submit_pipeline_job`. Response: 202 Accepted with body:
  {
    "job_id": "...",
    "status": "queued",
    "status_url": "/v1/jobs/{job_id}"
  }

7.4 Jobs status

Endpoint: GET /v1/jobs/{job_id}
Response (200 OK):
- If running/completed returns JSON with keys:
  {
    "job_id":"...",
    "status":"succeeded|running|failed|pending",
    "submitted_at": <timestamp>,
    "started_at": <timestamp|null>,
    "ended_at": <timestamp|null>,
    "result": { ... }  # present if SUCCEEDED
    "error": "..."      # present if FAILED
  }
If job not found: 404 with {"error":"job not found","job_id":"..."}

7.5 Preprocess-only

Endpoint: POST /v1/preprocess
Content-Type: multipart/form-data
Body: audio_file upload
Response (200 OK):
{
  "output_path": "/tmp/.../preprocessed_...wav"
}

7.6 ASR-only

Endpoint: POST /v1/asr
Body (application/json):
{
  "preprocessed_path": "/path/to/preprocessed.wav"
}
Response (200 OK):
{
  "segments": [...],
  "transcript_raw": "..."
}

7.7 Postprocess-only

Endpoint: POST /v1/postprocess
Body (application/json):
{
  "asr_output": { ... }  # ASR result dict
}
Optional query params via OutputFormatParams accepted as dependency injection
Response (200 OK):
{
  "summary": {...},
  "segments": [...],
  "quality_metrics": {...} (optional),
  "diffs": {...} (optional)
}

7.8 Response content negotiation and defaults

- The controller checks Accept header and `api` config in `get_config()` for default response format and whether to honor Accept/header or query override.
- Default response format controlled by `config.api.default_response_format` (e.g., "json" or "text").
- If `formats=text` or Accept: text/plain, server will return human-readable text combining transcript and summary.

7.9 OpenAPI / interactive docs (Litestar)

- OpenAPI JSON: http://localhost:8000/openapi
- Swagger UI / Docs: http://localhost:8000/docs

Litestar sẽ tự động sinh schema dựa trên Pydantic models.

## 8. Mẫu trả lời & xử lý lỗi

- Thành công (pipeline sync): 200 OK. Mặc định JSON `PipelineResponse` (see `src/omoai/api/models.py`).
- Tạo resource (enqueue job): 202 Accepted. Body: {"job_id": "...", "status":"queued", "status_url":"/v1/jobs/{job_id}"}
- Preprocess success: 200 OK {"output_path":"..."}
- ASR success: 200 OK {"segments": [...], "transcript_raw":"..."}

Lỗi:
- Bad Request: 400 (ví dụ thiếu file)
- Not Found: 404 (ví dụ job not found)
- Internal Error: 500 (response body: {"code":"internal_error","message":"Internal server error","trace_id":"..."})

Lưu ý: global exception handler (`global_exception_handler` in `app.py`) trả về envelope với fields `code`, `message`, `trace_id` và optional `details`.

## 9. Gợi ý vận hành

- Sử dụng biến môi trường `OMOAI_CONFIG` để chỉ định file cấu hình tuỳ chỉnh.
- Chỉnh `config.api.temp_dir` cho nơi lưu file tạm; đảm bảo quyền ghi và đủ dung lượng.
- Bật `config.output.save_on_api` nếu muốn lưu kết quả đầu ra cho debug/ audit.
- Đặt `UVICORN_WORKERS` env var để scale worker process (uvicorn workers param) khi chạy dưới tải lớn.
- Cấu hình logging via `config.logging` — `setup_logging()` được gọi khi app tạo.

---

Tôi đã tạo file `docs/API_TECHNICAL_DOC_VI.md` trong repo. Bạn muốn tôi bổ sung thêm phần ví dụ curl chi tiết cho mỗi endpoint, hoặc dịch sang tiếng Anh/format PDF không?