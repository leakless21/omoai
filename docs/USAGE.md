# OMOAI Usage Guide

This guide provides instructions on how to use the OMOAI application, covering both the RESTful API and the Interactive Command-Line Interface (CLI).

## 1. API Usage

The API provides programmatic access to the audio processing pipeline. It's built with Litestar and offers endpoints for each stage of the pipeline, as well as a full pipeline endpoint.

### Running the API Server

To run the API, ensure you're in the virtual environment and use the `litestar` command-line tool:

```bash
# Run the server
uv run litestar --app src.omoai.api.app:app run --host 0.0.0.0 --port 8000

# Run in development mode with auto-reloading
uv run litestar --app src.omoai.api.app:app run --host 0.0.0.0 --port 8000 --reload
```

### API Endpoints

-   `GET /health`: Provides health check information.
-   `POST /pipeline`: Runs the entire audio processing pipeline in a single request.
-   `POST /preprocess`: Handles audio preprocessing.
-   `POST /asr`: Performs speech recognition.
-   `POST /postprocess`: Adds punctuation and generates summaries.

### Testing the API with `curl`

-   **Health Check:**
    ```bash
    curl http://127.0.0.1:8000/health
    ```

-   **Full Pipeline:**
    ```bash
    curl -X POST -F "audio_file=@/path/to/your/audio.mp3" http://127.0.0.1:8000/pipeline
    ```

-   **Preprocess:**
    ```bash
    curl -X POST -F "audio_file=@/path/to/your/audio.mp3" http://127.0.0.1:8000/preprocess
    ```

-   **ASR:**
    ```bash
    curl -X POST -H "Content-Type: application/json" -d '{"preprocessed_path": "/path/to/your/preprocessed.wav"}' http://127.0.0.1:8000/asr
    ```

-   **Post-process:**
    ```bash
    curl -X POST -H "Content-Type: application/json" -d '{"asr_output": {"transcript_raw": "hello world", "segments": []}}' http://127.0.0.1:8000/postprocess
    ```

### Example API Response (`/pipeline`)

```json
{
  "summary": {
    "bullets": [
      "First key point from the audio content",
      "Second important point discussed"
    ],
    "abstract": "A concise 2-3 sentence summary of the entire audio content."
  },
  "segments": [
    {
      "start": "00:00:00:000",
      "end": "00:00:05:120",
      "text_raw": "hello world this is a test",
      "text_punct": "Hello world, this is a test."
    }
  ]
}
```

## 2. Interactive CLI Usage

The Interactive CLI provides a user-friendly, step-by-step interface for processing audio files.

### Launching the Interactive CLI

To launch the interactive CLI, run the following command from your project's root directory:

```bash
python src/omoai/main.py --interactive
```

### Main Menu Options

Upon launching, you will be presented with a main menu:

-   **Run Full Pipeline**: Execute the entire audio processing pipeline.
-   **Run Individual Stages**: Execute a specific stage (preprocessing, ASR, or post-processing).
-   **Configuration**: View the current configuration settings.
-   **Exit**: Terminate the interactive session.

Navigate the menu using the arrow keys and press `Enter` to select an option.

### Running the Full Pipeline (CLI)

This option guides you through the complete process:

1.  **Select "Run Full Pipeline"** from the main menu.
2.  **Enter the path to your audio file** when prompted.
3.  **Specify an output directory**. A default will be suggested.
4.  **Optionally, provide paths** to a custom model directory or configuration file.
5.  **Confirm your settings** to start the pipeline.
6.  Progress messages for each stage will be displayed.
7.  Upon completion, the path to the output directory will be shown.

### Running Individual Stages (CLI)

This option allows you to run a specific part of the pipeline, which is useful for debugging or resuming a partial workflow.

-   **Preprocess Audio**: Converts your audio file into the required WAV format.
-   **Run ASR**: Transcribes a preprocessed WAV file into text.
-   **Post-process ASR Output**: Adds punctuation and summarizes a raw ASR transcription.

For each option, you will be prompted for the necessary input and output file paths.
