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
- `transcript.punct.txt`: The final, human-readable punctuated transcript.
- `summary.md`: The generated summary in Markdown format (bullets + abstract).

### Enhanced Output Options

The enhanced output system supports multiple formats and configurations:

**Available Formats:**
- `json`: Structured JSON data with configurable content
- `text`: Human-readable text with optional timestamps
- `srt`: Standard SRT subtitle format for video players
- `vtt`: WebVTT subtitle format for web applications
- `md`: Markdown documentation format with metadata

**Configuration Examples:**

```yaml
# Basic configuration (JSON + Text)
output:
  formats: ["json", "text"]
  transcript:
    include_punct: true
    include_segments: true
    timestamps: "clock"
  summary:
    mode: "both"
    bullets_max: 7

# Subtitle-focused configuration
output:
  formats: ["srt", "vtt"]
  transcript:
    include_punct: true
    include_segments: true
    timestamps: "clock"
  summary:
    mode: "none"  # No summary for subtitle files

# Documentation configuration
output:
  formats: ["md"]
  transcript:
    include_raw: true
    include_punct: true
    include_segments: true
    timestamps: "s"  # Second-based timestamps
  summary:
    mode: "both"
    bullets_max: 10
    abstract_max_chars: 500
```

**Timestamp Options:**
- `none`: No timestamps
- `s`: Seconds (e.g., "3.50s")
- `ms`: Milliseconds (e.g., "3500ms")
- `clock`: HH:MM:SS.mmm format (e.g., "00:00:03.500")

See [`config_enhanced_output.yaml`](config_enhanced_output.yaml:0) for a complete example configuration.

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

3. **Process with enhanced output options:**
   ```bash
   curl -X POST "http://localhost:8000/pipeline?include=segments&ts=clock&summary=bullets&summary_bullets_max=5" \
        -F "audio_file=@data/input/audio_file.mp3"
   ```

4. **Check API health:**
   ```bash
   curl http://localhost:8000/health
   ```

**API Features:**
- Supports up to 100MB audio files
- Real-time progress monitoring via server stdout
- Full pipeline processing in one request
- Individual stage endpoints available (`/preprocess`, `/asr`, `/postprocess`)
- **Enhanced Output Options:** Configure what to include and output formats via query parameters

**Query Parameters for Output Control:**
- `include`: Comma-separated list of what to include (`transcript_raw`, `transcript_punct`, `segments`)
- `ts`: Timestamp format (`none`, `s`, `ms`, `clock`)
- `summary`: Summary type (`bullets`, `abstract`, `both`, `none`)
- `summary_bullets_max`: Maximum number of bullet points to return
- `summary_lang`: Summary language (e.g., `vi`, `en`)

**Examples:**
```bash
# Get only segments with clock timestamps and bullet summary
curl -X POST "http://localhost:8000/pipeline?include=segments&ts=clock&summary=bullets" \
     -F "audio_file=@audio.mp3"

# Get segments and abstract only (no bullets)
curl -X POST "http://localhost:8000/pipeline?include=segments&summary=abstract" \
     -F "audio_file=@audio.mp3"

# Get everything with detailed timestamps
curl -X POST "http://localhost:8000/pipeline?include=transcript_raw,transcript_punct,segments&ts=ms&summary=both&summary_bullets_max=10" \
     -F "audio_file=@audio.mp3"
```

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
│   ├── output/             # Enhanced output system
│   │   ├── __init__.py     # Public API exports
│   │   ├── formatter.py    # Base formatter interfaces and registry
│   │   ├── writer.py       # Output orchestration
│   │   └── plugins/        # Format-specific implementations
│   │       ├── __init__.py # Plugin registration
│   │       ├── text.py     # Text formatter
│   │       ├── json.py     # JSON formatter
│   │       ├── srt.py      # SRT subtitle formatter
│   │       ├── vtt.py      # WebVTT formatter
│   │       └── markdown.py # Markdown formatter
│   └── main.py             # Command-line interface
└── tests/                  # Unit tests
```

## Testing

Test the enhanced output system:

```bash
# Run the comprehensive output system demo
uv run python test_output_system.py

# Test specific functionality
uv run python -c "from src.omoai.output import list_formatters; print('Available formatters:', list_formatters())"
```

## Documentation

For more detailed information, please refer to the documentation in the `docs/` directory:

- [`docs/REQUIREMENTS.md`](docs/REQUIREMENTS.md:0): Functional and non-functional requirements.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md:0): Detailed system architecture and component interactions.
- [`docs/API_PLAN.md`](docs/API_PLAN.md:0): Comprehensive API documentation and usage examples.
- [`docs/OUTPUT_REDESIGN_PLAN.md`](docs/OUTPUT_REDESIGN_PLAN.md:0): Enhanced output system design and planning.
- [`docs/ARCHITECTURE_ASSESSMENT.md`](docs/ARCHITECTURE_ASSESSMENT.md:0): Architecture assessment for the output redesign.
- [`docs/IMPLEMENTATION_SUMMARY.md`](docs/IMPLEMENTATION_SUMMARY.md:0): Complete implementation summary of the enhanced output system.
- [`docs/COMPONENT_MAIN_DOCS.md`](docs/COMPONENT_MAIN_DOCS.md:0): Documentation for the main application domain.
- [`docs/COMPONENT_CHUNKFORMER_DOCS.md`](docs/COMPONENT_CHUNKFORMER_DOCS.md:0): Documentation for the Chunkformer ASR model domain.
- [`config_enhanced_output.yaml`](config_enhanced_output.yaml:0): Example enhanced configuration with all new options.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
