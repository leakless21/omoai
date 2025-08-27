# Audio Processing Pipeline

A sophisticated audio processing pipeline designed to transcribe and summarize audio files, particularly podcasts. It leverages a `Chunkformer` model for Automatic Speech Recognition (ASR) and a Large Language Model (LLM) for punctuation and summarization.

## Features

- **High-Quality Transcription:** Uses the `Chunkformer` model for accurate speech-to-text conversion.
- **Long-Form Audio Support:** Processes audio in chunks to handle long-form content like podcasts efficiently.
- **Punctuation & Capitalization:** Applies natural language rules to the raw transcript using an LLM.
- **Content Summarization:** Generates concise summaries of the transcribed content.
- **Configurable Pipeline:** All parameters are customizable via a central configuration file.
- **Multiple Output Formats:** Enhanced output system supporting JSON, text, SRT subtitles, WebVTT, and Markdown formats.
- **Flexible Output Options:** Configure transcript parts (raw, punctuated, segments), timestamp formats, and summary modes.
- **Professional Outputs:** Generate SRT/VTT files for video production, Markdown for documentation, and structured JSON for programmatic access.
- **Backward Compatibility:** Existing configurations continue to work unchanged while new features are opt-in.

## Architecture

The project is organized into three main stages, orchestrated by a central script:

1. **Preprocessing:** Converts input audio into a 16kHz mono WAV format using `ffmpeg`.
2. **Transcription (ASR):** Uses the `Chunkformer` model to perform speech-to-text conversion.
3. **Post-processing:** Applies punctuation, capitalization, and generates a summary using an LLM.

For a detailed architectural overview, see [`docs/architecture/index.md`](docs/architecture/index.md).

## Installation

### Prerequisites

- Python 3.8 or higher
- A CUDA-compatible GPU (recommended for faster processing)
- `ffmpeg` installed on your system

### Setup

1. **Clone the repository:**

    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Create and activate a virtual environment using UV:**

    ```bash
    uv venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3. **Install dependencies:**

    ```bash
    uv pip install -e .
    ```

    This command will install all Python dependencies listed in [`pyproject.toml`](pyproject.toml), including `torch`, `torchaudio`, `vllm`, and `pydub`.

4. **Ensure `ffmpeg` is in your system's PATH:**
    - **On macOS (using Homebrew):** `brew install ffmpeg`
    - **On Ubuntu/Debian:** `sudo apt-get install ffmpeg`
    - **On Windows:** Download from [the official website](https://ffmpeg.org/download.html) and add it to your system's PATH.

## Usage

This project can be used via the command-line interface (CLI) or the REST API.

### Command-Line Interface (CLI)

The CLI allows you to run the entire pipeline or individual stages.

#### Running the full pipeline:

Execute the main script with the path to your audio file:

```bash
python src/omoai/main.py data/input/your_audio_file.mp3
```

The pipeline will create a uniquely named output directory in `data/output/` containing the results.

#### Interactive CLI:

For a step-by-step interactive experience, run:

```bash
python src/omoai/main.py --interactive
```

### REST API

The project also provides a REST API for programmatic access.

1. **Start the API server:**

   ```bash
   uv run litestar --app src.omoai.api.app:app run --host 0.0.0.0 --port 8000
   ```

2. **Process audio via API:**

   ```bash
   curl -X POST http://localhost:8000/pipeline -F "audio_file=@data/input/audio_file.mp3"
   ```

3. **Check API health:**

   ```bash
   curl http://localhost:8000/health
   ```

For more details on API usage and parameters, see the [User Guide](docs/user_guide/index.md).

## Configuration

The pipeline is configured through the `config.yaml` file. You can customize paths, ASR parameters, LLM settings, and output formats. For detailed information on configuration, see the [Configuration Guide](docs/user_guide/configuration.md).

## Project Structure

```
.
├── src/omoai/main.py       # Main orchestration script
├── config.yaml             # Configuration file
├── pyproject.toml          # Project dependencies
├── README.md               # This file
├── src/chunkformer/        # Source code for the Chunkformer ASR model
├── models/                 # Directory for pre-trained model weights
├── data/                   # Input and output data
├── scripts/                # Pipeline processing scripts
├── src/omoai/              # Main package
└── tests/                  # Unit tests
```

## Testing

To run the test suite:

```bash
uv run pytest
```

## Documentation

For more detailed information, please refer to the documentation in the `docs/` directory:

- **[Architecture](docs/architecture/index.md):** Detailed system architecture and component interactions.
- **[User Guide](docs/user_guide/index.md):** Instructions on how to use the API and CLI.
- **[Configuration](docs/user_guide/configuration.md):** Guide to configuring the pipeline.
- **[Development](docs/development/best_practices.md):** Best practices for development and post-processing scripts.
- **[Project](docs/project/requirements.md):** Project requirements and gap analysis.

The `Chunkformer` model has its own detailed `README.md` located in `src/chunkformer/README.md`.

## License

This project is licensed under the MIT License.