# Audio Processing Pipeline

A sophisticated audio processing pipeline designed to transcribe and summarize audio files, particularly podcasts. It leverages a `Chunkformer` model for Automatic Speech Recognition (ASR) and a Large Language Model (LLM) for punctuation and summarization.

## Features

- **High-Quality Transcription:** Uses the `Chunkformer` model for accurate speech-to-text conversion.
- **Long-Form Audio Support:** Processes audio in chunks to handle long-form content like podcasts efficiently.
- **Punctuation & Capitalization:** Applies natural language rules to the raw transcript using an LLM.
- **Content Summarization:** Generates concise summaries of the transcribed content.
- **Configurable Pipeline:** All parameters are customizable via a central configuration file.
- **Structured Output:** Produces final outputs in JSON and plain text formats for easy integration.

## Architecture

The project is organized into three main stages, orchestrated by a central script:

1. **Preprocessing:** Converts input audio into a 16kHz mono WAV format using `ffmpeg`.
2. **Transcription (ASR):** Uses the `Chunkformer` model to perform speech-to-text conversion.
3. **Post-processing:** Applies punctuation, capitalization, and generates a summary using an LLM.

For a detailed architectural overview, see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md:0).

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

    This command will install all Python dependencies listed in [`pyproject.toml`](pyproject.toml:0), including `torch`, `torchaudio`, `vllm`, and `pydub`.

4. **Ensure `ffmpeg` is in your system's PATH:**
    - **On macOS (using Homebrew):** `brew install ffmpeg`
    - **On Ubuntu/Debian:** `sudo apt-get install ffmpeg`
    - **On Windows:** Download from [the official website](https://ffmpeg.org/download.html) and add it to your system's PATH.

## Usage

### Configuration

Before running the pipeline, you may need to adjust the settings in [`config.yaml`](config.yaml:0). Key sections include:

- **`paths`**: Ensure the `chunkformer_model_path` points to your model directory.
- **`asr`**: Adjust `chunk_size` and `batch_duration` based on your system's capabilities.
- **`llm`**: Configure the LLM settings for post-processing.

### Running the Pipeline

Execute the main script with the path to your audio file:

```bash
python main.py data/input/your_audio_file.mp3
```

The pipeline will create a uniquely named output directory in [`data/output/`](data/output/:0) containing:

- `preprocessed.wav`: The standardized audio file.
- `asr.json`: The raw transcript with timestamps from the ASR model.
- `final.json`: A comprehensive JSON file with the polished transcript and summary.
- `transcript.txt`: The final, human-readable transcript.
- `summary.txt`: The generated summary of the audio content.

### Example

```bash
# Process a podcast episode
python main.py data/input/podcast_episode.mp3

# The output will be saved in a directory like:
# data/output/podcast_episode-<unique-id>/
```

### REST API

The project also provides a REST API for programmatic access:

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

**API Features:**
- Supports up to 100MB audio files
- Real-time progress monitoring via server stdout
- Full pipeline processing in one request
- Individual stage endpoints available (`/preprocess`, `/asr`, `/postprocess`)

For detailed API documentation, see [`docs/API_PLAN.md`](docs/API_PLAN.md).

## Project Structure

```
.
├── main.py                 # Main orchestration script
├── config.yaml             # Configuration file
├── pyproject.toml          # Project dependencies
├── README.md               # This file
├── chunkformer/            # Source code for the Chunkformer ASR model
│   ├── model/              # Neural network components
│   └── decode.py           # Decoding logic
├── models/                 # Directory for pre-trained model weights
├── data/                   # Input and output data
│   ├── input/              # Raw audio files
│   └── output/             # Processing results
├── scripts/                # Pipeline processing scripts
│   ├── preprocess.py       # Audio preprocessing
│   ├── asr.py              # ASR transcription
│   └── post.py             # Post-processing (punctuation, summary)
├── src/omoai/              # Main package
│   ├── api/                # REST API components
│   │   ├── app.py          # API application setup
│   │   ├── controllers/    # API endpoint controllers
│   │   ├── models.py       # Pydantic models for requests/responses
│   │   ├── services.py     # Business logic layer
│   │   └── scripts/        # Script wrappers for API integration
│   └── main.py             # Command-line interface
└── tests/                  # Unit tests
```

## Documentation

For more detailed information, please refer to the documentation in the `docs/` directory:

- [`docs/REQUIREMENTS.md`](docs/REQUIREMENTS.md:0): Functional and non-functional requirements.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md:0): Detailed system architecture and component interactions.
- [`docs/API_PLAN.md`](docs/API_PLAN.md:0): Comprehensive API documentation and usage examples.
- [`docs/COMPONENT_MAIN_DOCS.md`](docs/COMPONENT_MAIN_DOCS.md:0): Documentation for the main application domain.
- [`docs/COMPONENT_CHUNKFORMER_DOCS.md`](docs/COMPONENT_CHUNKFORMER_DOCS.md:0): Documentation for the Chunkformer ASR model domain.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
