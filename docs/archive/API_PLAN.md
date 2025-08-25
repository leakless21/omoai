# Detailed and Aligned Plan for Creating a Minimal API with Litestar

This document provides a detailed, step-by-step plan to create a minimal API for the OMOAI audio transcription and summarization pipeline using the Litestar framework. This plan is aligned with the current state of the program and is designed to be very clear and provide a lot of handholding.

## 1. Project Setup and Virtual Environment

Before we start writing the API code, we need to ensure our project is set up correctly and we are using a virtual environment to manage our dependencies. We will use `uv` for this, which is a fast and modern Python package manager.

1.  **Verify Virtual Environment**: Your project is already configured to use a virtual environment with `uv`. Make sure it is activated by running the command `source .venv/bin/activate` in your terminal. You should see the name of the virtual environment in your terminal prompt.

2.  **Verify Dependencies**: Your project's dependencies are listed in the `pyproject.toml` file. `litestar` is already included in this file. To ensure all dependencies are installed, run the command `uv pip install -r requirements.txt`.

3.  **Create the API Directory**: To keep our project organized, we will create a new directory to house our API-related code. We will create a new directory named `api` inside the `src/omoai` directory. The full path will be `src/omoai/api`.

4.  **Create the Application File**: Inside the `src/omoai/api` directory, we will create a new file named `app.py`. This file will be the main entry point for our Litestar application.

## 2. API Workflow

The API is designed to be used as a series of steps, where the output of one step is used as the input for the next. Here is the intended workflow:

1.  The client sends an audio file to the `/preprocess` endpoint.
2.  The `/preprocess` endpoint returns a path to the preprocessed audio file.
3.  The client sends the path of the preprocessed audio file to the `/asr` endpoint.
4.  The `/asr` endpoint returns the raw transcript and segments.
5.  The client sends the raw transcript and segments to the `/postprocess` endpoint.
6.  The `/postprocess` endpoint returns the final punctuated transcript and summary.

Alternatively, the client can use the `/pipeline` endpoint to run all the steps in a single call.

**Note**: The API now leverages the existing `scripts/` modules directly for processing, which provides better performance and allows progress monitoring through stdout/stderr output in the API server terminal.

## 3. API Structure and Controllers

To keep our API organized and easy to maintain, we will use Litestar's `Controller` feature. A controller is a class that groups related API endpoints together. We will create a controller for each major piece of functionality in our application.

-   **`MainController`**: This controller will be responsible for the main API endpoint that runs the entire audio processing pipeline, from start to finish. It will take an audio file as input and return the final transcript and summary.

-   **`PreprocessController`**: This controller will handle the API endpoint for the audio preprocessing stage. It will take an audio file as input and convert it to the correct format for the ASR model.

-   **`ASRController`**: This controller will be responsible for the Automatic Speech Recognition (ASR) stage. It will take the path to a preprocessed audio file and return the raw transcript.

-   **`PostprocessController`**: This controller will handle the final stage of the pipeline, which involves adding punctuation to the transcript and generating a summary. It will take the raw transcript from the ASR stage as input.

Each of these controllers will be defined in its own separate file within the `src/omoai/api/` directory. For example, the `MainController` will be in a file named `src/omoai/api/main_controller.py`.

## 4. Request and Response Models with Pydantic

To ensure that the data sent to and from our API is in the correct format, we will use Pydantic models. Pydantic is a library that allows us to define data schemas using Python's type hints. Litestar has excellent integration with Pydantic, which means that it will automatically validate incoming requests and outgoing responses against our Pydantic models.

### Request Models

These models define the structure of the data that our API expects to receive in the request body.

-   **`PipelineRequest`**: This model will be used for the main pipeline endpoint. It will have a single field, `audio_file`, which will be of type `UploadFile`. This tells Litestar to expect a file upload for this endpoint.

-   **`PreprocessRequest`**: This model will be used for the preprocessing endpoint. It will also have a single field, `audio_file`, of type `UploadFile`.

-   **`ASRRequest`**: This model will be used for the ASR endpoint. It will have a single field, `preprocessed_path`, which will be a string containing the path to the preprocessed audio file.

-   **`PostprocessRequest`**: This model will be used for the post-processing endpoint. It will have a single field, `asr_output`, which will be a Python dictionary. This dictionary will contain the JSON output from the ASR stage.

### Response Models

These models define the structure of the data that our API will send back in the response.

-   **`PipelineResponse`**: This model contains the final results of the pipeline, including the `summary` (a dictionary containing the summary of the transcript) and the `segments` (a list of transcript segments with timestamps and both raw and punctuated text).

-   **`PreprocessResponse`**: This model contains an `output_path` field with the path to the preprocessed file on the server.

-   **`ASRResponse`**: This model contains the `segments` (a list of transcript segments with timestamps and raw text).

-   **`PostprocessResponse`**: This model contains the `summary` (a dictionary containing the summary) and the `segments` (updated with punctuated text for each segment).

**Note**: The API no longer returns full transcript fields (`transcript_raw`, `transcript_punct`) to reduce redundancy, as all text content is available in the segmented format with timestamps.

## 5. API Endpoints

Here is a detailed description of the API endpoints that we have implemented:

-   **`POST /pipeline`**: This endpoint runs the entire audio processing pipeline in a single request. Send a `POST` request with an audio file (multipart/form-data). The API runs preprocessing, ASR, and post-processing stages sequentially and returns a `PipelineResponse` with the summary and timestamped segments containing both raw and punctuated text. Supports up to 100MB audio files.

-   **`POST /preprocess`**: This endpoint handles audio preprocessing. Send a `POST` request with an audio file (multipart/form-data). The API converts the audio to 16kHz mono PCM16 WAV format and returns a `PreprocessResponse` with the path to the preprocessed file.

-   **`POST /asr`**: This endpoint performs speech recognition using the Chunkformer model. Send a `POST` request with the path to a preprocessed audio file (JSON body). The API returns an `ASRResponse` with timestamped segments containing raw transcribed text.

-   **`POST /postprocess`**: This endpoint adds punctuation and generates summaries using vLLM. Send a `POST` request with the ASR output (JSON body). The API returns a `PostprocessResponse` with the summary and segments updated with punctuated text.

-   **`GET /health`**: This endpoint provides health check information. Returns a simple status response to verify the API is running.

## 6. Progress Monitoring

The API now provides real-time progress monitoring through stdout/stderr output. When running processing requests, you can see:

- **FFmpeg preprocessing progress**: Audio conversion status and file size information
- **Chunkformer ASR progress**: Model loading, feature extraction, and inference progress
- **vLLM processing progress**: LLM initialization, token processing, and generation progress

All progress information is displayed in the terminal where the API server is running, making it easy to monitor long-running requests.

## 7. Error Handling

Our API uses standard HTTP status codes to indicate request success or failure:
- `200 OK`: Request successful
- `400 Bad Request`: Invalid request data or missing parameters
- `413 Request Entity Too Large`: File exceeds 100MB limit
- `500 Internal Server Error`: Server-side processing error

Error responses include a detailed message explaining what went wrong. The API also provides progress output even during error conditions to help with debugging.

## 8. Configuration

The API uses the main `config.yaml` file for all settings. The API-specific configuration section includes:

```yaml
api:
  # Server configuration
  host: "0.0.0.0"
  port: 8000
  # Request limits
  max_body_size_mb: 100
  request_timeout_seconds: 300
  # File handling
  temp_dir: "/tmp"
  cleanup_temp_files: true
  # Progress output
  enable_progress_output: true
  # Health check configuration
  health_check_dependencies:
    - ffmpeg
    - config_file
    - asr_script
    - postprocess_script
```

You can modify these settings to customize the API behavior according to your environment.

## 9. Running the API

To run the API, ensure you're in the virtual environment and use the `litestar` command-line tool:

```bash
# Using uv (recommended) - no need to manually activate venv
uv run litestar --app src.omoai.api.app:app run --host 0.0.0.0 --port 8000

# Or if you prefer to activate the venv first:
# source .venv/bin/activate
# litestar --app src.omoai.api.app:app run --host 0.0.0.0 --port 8000
```

For development, use the `--reload` flag to automatically restart on code changes:

```bash
uv run litestar --app src.omoai.api.app:app run --host 0.0.0.0 --port 8000 --reload
```

The API will automatically load configuration from `config.yaml` and apply the settings for request limits, file handling, and other operational parameters.

## 10. Testing the API

Once the API is running, you can test it by sending requests to it using a tool like `curl`. Here are some example `curl` commands for each endpoint:

-   **Testing the health check:**
    
    `curl http://127.0.0.1:8000/health`

-   **Testing the `/pipeline` endpoint**:

    `curl -X POST -F "audio_file=@/path/to/your/audio.mp3" http://127.0.0.1:8000/pipeline`

-   **Testing the `/preprocess` endpoint**:

    `curl -X POST -F "audio_file=@/path/to/your/audio.mp3" http://127.0.0.1:8000/preprocess`

-   **Testing the `/asr` endpoint**:

    `curl -X POST -H "Content-Type: application/json" -d '{"preprocessed_path": "/path/to/your/preprocessed.wav"}' http://127.0.0.1:8000/asr`

-   **Testing the `/postprocess` endpoint**:

    `curl -X POST -H "Content-Type: application/json" -d '{"asr_output": {"transcript_raw": "hello world", "segments": []}}' http://127.0.0.1:8000/postprocess`

Remember to replace `/path/to/your/audio.mp3` and `/path/to/your/preprocessed.wav` with the actual paths to your audio files.

## Example API Response

The `/pipeline` endpoint returns an optimized response format:

```json
{
  "summary": {
    "bullets": [
      "First key point from the audio content",
      "Second important point discussed",
      "Third main topic covered"
    ],
    "abstract": "A concise 2-3 sentence summary of the entire audio content."
  },
  "segments": [
    {
      "start": "00:00:00:000",
      "end": "00:00:05:120",
      "text_raw": "hello world this is a test",
      "text_punct": "Hello world, this is a test."
    },
    {
      "start": "00:00:05:120", 
      "end": "00:00:10:240",
      "text_raw": "another segment of speech",
      "text_punct": "Another segment of speech."
    }
  ]
}
```

This format eliminates redundant full transcript fields while preserving all information in the timestamped segments.