# System Architecture

This document provides a detailed architectural overview of the audio processing pipeline, outlining the system's components, their responsibilities, and interactions.

## 1. High-Level Overview

The system is designed as a modular, end-to-end pipeline for processing audio files, with a primary focus on podcasts. The architecture is composed of three main stages: Preprocessing, Transcription (ASR), and Post-processing. Each stage is implemented as a distinct Python script, orchestrated by a central controller, `main.py`.

The pipeline accepts an audio file as input, transforms it into a standardized format, transcribes it into text, and then refines the text by adding punctuation and generating a summary. The entire process is configurable through a central YAML configuration file, `config.yaml`.

The system also exposes its functionality through a RESTful API, allowing for programmatic access.

## 2. System Components

### 2.1. Main Orchestrator (`src/omoai/main.py`)

The `main.py` script serves as the primary entry point and central controller for the entire pipeline.

- **Area of Responsibility**: Manages the workflow, loads configurations, and coordinates the interaction between the preprocessing, ASR, and post-processing components.
- **Orchestration Logic**: Executes the pipeline stages (preprocess, ASR, post-process) in the correct order, passing necessary information between them.
- **Configuration Management**: Loads settings from `config.yaml`, validates them, and makes them available to the downstream components.
- **Interfaces**:
  - **Input**: Command-line arguments for the audio file path and optional overrides for the configuration.
  - **Output**: Calls to subsequent scripts via subprocess.
- **Dependencies**: `config.yaml`, `scripts/preprocess.py`, `scripts/asr.py`, `scripts/post.py`.

### 2.2. Preprocessing Component (`scripts/preprocess.py`)

This script is responsible for preparing the raw audio file for the ASR model.

- **Area of Responsibility**: Audio format conversion and normalization.
- **Technical Requirements**: Must use `ffmpeg` to ensure audio is converted to 16kHz mono PCM16 WAV format.
- **Compute**: CPU-bound, as it involves audio transcoding.
- **Storage**: Reads the input audio file and writes the preprocessed WAV file to the output directory.
- **Interface**: Takes the raw audio file path and the designated output path as input. It outputs the path to the preprocessed WAV file.
- **Dependencies**: `pydub` (for audio manipulation), `ffmpeg`.

### 2.3. ASR Component (`scripts/asr.py` & `src/omoai/chunkformer/`)

The ASR component is the heart of the speech recognition process, utilizing the **Chunkformer** model.

- **Area of Responsibility**: Converting preprocessed audio into a text transcript with timestamps.
- **Technical Requirements**: A hybrid CTC-attention model implemented in PyTorch. It must load the pre-trained `Chunkformer` model from the path specified in `config.yaml`. It must process audio in chunks to handle long-form content efficiently.
- **Compute**: GPU-bound for optimal performance, leveraging `torch` for neural network inference.
- **Storage**: Reads the preprocessed WAV file and writes the raw ASR output as a JSON file. The model weights are stored in `models/chunkformer/`.
- **Core Classes & Logic**:
  - `chunkformer.model.asr_model.ASRModel`: The main model class.
  - `chunkformer.decode.py`: Contains the decoding logic, including `endless_decode` for long-form audio.
- **Interface**:
  - **Script Interface**: Takes the path to the preprocessed audio and the output directory as input. It outputs a JSON file containing the transcript and timestamps.
- **Dependencies**: `torch`, `torchaudio`, `pydub`, the `Chunkformer` model files.

### 2.4. Post-processing Component (`scripts/post.py`)

This component refines the raw transcript from the ASR stage.

- **Area of Responsibility**: Applying punctuation and capitalization, and generating a summary.
- **Technical Requirements**: Must use the specified LLM (via `vllm`) for both punctuation and summarization. The system prompts and sampling parameters are configurable in `config.yaml`. It uses a map-reduce strategy for summarizing long texts.
- **Compute**: GPU-bound, as it involves running the LLM.
- **Storage**: Reads the raw ASR JSON file and writes the final polished JSON, transcript text, and summary text files.
- **Interface**: Takes the raw ASR JSON file path and the output directory as input. It produces multiple output files.
- **Dependencies**: `vllm` (for LLM inference), `pyyaml` (for accessing configuration).

### 2.5. API Component (`src/omoai/api/`)

This component is responsible for exposing the audio processing pipeline via a RESTful API.

- **Area of Responsibility**: Provides a web-based interface for submitting audio files and receiving processed transcripts and summaries.
- **Technical Requirements**: Built using the Litestar framework. It can use either script-based wrappers or in-memory processing with cached model singletons for high performance.
- **Compute**: Lightweight for request handling. Processing is delegated to the appropriate backend (scripts or in-memory pipeline).
- **Interface**: Exposes RESTful endpoints for the main pipeline (`/pipeline`), as well as individual stages (`/preprocess`, `/asr`, `/postprocess`), plus health check (`/health`).
- **Key Features**:
  - **Dual-Mode Operation**: Can run in a high-performance in-memory mode or a reliable script-based fallback mode.
  - **Model Caching**: Uses singletons to load models once and cache them for subsequent requests, significantly reducing latency.
  - **Progress Monitoring**: Real-time stdout/stderr output from underlying scripts is visible in the API server terminal.
  - **Large File Support**: Handles up to 100MB audio files.
- **Dependencies**: `litestar`, `pydantic`.

## 3. Data Flow

The data flows through the system in a linear, sequential manner:

1.  **Input**: An audio file (e.g., `podcast.mp3`) is provided to `main.py` or the `/pipeline` API endpoint.
2.  **Preprocessing**: `scripts/preprocess.py` converts `podcast.mp3` to `preprocessed.wav`.
3.  **ASR**: `scripts/asr.py` takes `preprocessed.wav` and generates `asr.json`.
4.  **Post-processing**: `scripts/post.py` reads `asr.json` and produces `final.json`, `transcript.txt`, and `summary.txt`.
5.  **Output**: All processed files are stored in the `data/output/` directory, organized by a unique identifier for each run.

In the API's in-memory mode, this data is passed between stages as in-memory objects (e.g., PyTorch Tensors, Python dicts), avoiding disk I/O.

## 4. Configuration

The entire system's behavior is controlled by `config.yaml`, which is structured into the following sections:

- **`paths`**: Defines the base path for the `Chunkformer` model and the output directory.
- **`asr`**: Configures ASR-specific parameters like `chunk_size` and `batch_duration`.
- **`llm`**: Specifies the LLM to be used for post-processing.
- **`punctuation`**: Contains the system prompt and sampling parameters for the punctuation task.
- **`summarization`**: Contains the system prompt and sampling parameters for the summarization task.
- **`api`**: Configures the API server, including host, port, and request limits.

## 5. Compute and Storage

### 5.1. Compute Requirements

- **Preprocessing**: CPU.
- **ASR**: GPU recommended for faster processing.
- **Post-processing (LLM)**: GPU recommended for efficient LLM inference.

### 5.2. Storage Requirements

- **Input Data**: Raw audio files stored in `data/input/`.
- **Model Weights**: Pre-trained `Chunkformer` model stored in `models/`.
- **Output Data**: All generated files (WAV, JSON, TXT) are stored in `data/output/`.
- **Intermediate Data**: In script-based mode, preprocessed WAV files and ASR JSON files are stored temporarily.

## 6. External Dependencies

The project's dependencies are managed via `pyproject.toml` and include:

- **`torch`, `torchaudio`**: For deep learning and audio processing.
- **`vllm`**: For efficient execution of Large Language Models.
- **`litestar`**: The web framework for the API.
- **`pydub`**: For audio manipulation.
- **`pyyaml`**: For parsing the configuration file.

## 7. Punctuation

The punctuation restoration process is a critical step in the post-processing phase, responsible for transforming the raw, unpunctuated transcript from the ASR model into grammatically correct and readable text.

- **Area of Responsibility**: Restoring proper punctuation and capitalization to the raw transcript generated by the ASR model.
- **Technical Requirements**: Utilizes a Large Language Model (LLM) configured in `config.yaml`. The process is driven by a new, enhanced `system_prompt` designed to improve punctuation quality.
- **`system_prompt`**: The updated system prompt incorporates few-shot learning, providing the model with multiple high-quality examples of input (unpunctuated) and output (punctuated) text. This approach significantly enhances the model's ability to correctly apply nuanced punctuation rules, such as comma splices, complex sentence structures, and dialogue formatting.
- **Performance**: The improved examples in the prompt lead to a higher-quality output, reducing the need for manual editing and improving the overall accuracy of the transcription pipeline.
- **Interface**: Receives a block of unpunctuated text from the post-processing script and returns the text with correct punctuation and capitalization applied.
- **Dependencies**: `vllm`, `config.yaml`.
