# System Architecture

This document provides a detailed architectural overview of the audio processing pipeline, outlining the system's components, their responsibilities, and interactions.

## 1. High-Level Overview

The system is designed as a modular, end-to-end pipeline for processing audio files, with a primary focus on podcasts. The architecture is composed of three main stages: Preprocessing, Transcription (ASR), and Post-processing. These stages can be executed through both a command-line interface and a RESTful API.

The pipeline accepts an audio file as input, transforms it into a standardized format, transcribes it into text, and then refines the text by adding punctuation and generating a summary. The entire process is configurable through a central YAML configuration file, `config.yaml`.

The system features dual-mode operation:

- **Script-based mode**: Traditional execution through external scripts for maximum compatibility
- **In-memory mode**: High-performance processing with cached models for optimal speed
- **Auto mode**: Intelligent selection between the two modes based on system capabilities

The RESTful API provides comprehensive access to all pipeline functionality with automatic fallback between processing modes.

## 2. System Components

[text](../project)

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

This component is responsible for exposing the audio processing pipeline via a RESTful API with dual-mode operation capabilities.

- **Area of Responsibility**: Provides a web-based interface for submitting audio files and receiving processed transcripts and summaries with comprehensive output formatting options.
- **Technical Requirements**: Built using the Litestar framework with Pydantic for request/response validation. Features intelligent service mode selection with automatic fallback.
- **Compute**: Lightweight for request handling. Processing is delegated to the appropriate backend (scripts or in-memory pipeline) based on system capabilities and configuration.
- **Interface**: Exposes RESTful endpoints for the main pipeline (`/pipeline`), as well as individual stages (`/preprocess`, `/asr`, `/postprocess`), plus health check (`/health`).
- **Key Features**:
  - **Dual-Mode Operation**: Can run in a high-performance in-memory mode or a reliable script-based fallback mode, with automatic selection.
  - **Model Caching**: Uses singletons to load models once and cache them for subsequent requests, significantly reducing latency.
  - **Flexible Output Formats**: Supports multiple response formats including JSON, plain text, SRT, VTT, and Markdown.
  - **Query Parameter Configuration**: Extensive customization options through query parameters for output content and formatting.
  - **Quality Metrics**: Optional inclusion of transcription quality metrics and human-readable diffs.
  - **Progress Monitoring**: Real-time stdout/stderr output from underlying scripts is visible in the API server terminal.
  - **Large File Support**: Handles up to 100MB audio files (configurable).
  - **Error Handling**: Comprehensive error handling with appropriate HTTP status codes and detailed error messages.
  - **OpenAPI Documentation**: Interactive API documentation available at `/schema` endpoint.
- **Core Classes & Logic**:
  - `src/omoai/api.app`: Litestar application setup and configuration
  - `src/omoai.api.main_controller.MainController`: Handles the main `/pipeline` endpoint with full pipeline processing
  - `src/omoai.api.models`: Pydantic models for request/response validation including `PipelineRequest`, `PipelineResponse`, `OutputFormatParams`
  - `src.omoai.api.services_enhanced`: Smart service selection with automatic fallback between v1 (script) and v2 (in-memory) implementations
- **Dependencies**: `litestar`, `pydantic`, `uvicorn`, `python-multipart`.

### 2.6. Service Management Layer (`src/omoai/api/services_enhanced.py`)

This component provides intelligent service management with automatic fallback capabilities.

- **Area of Responsibility**: Manages the selection and execution of processing services based on system capabilities and configuration.
- **Technical Requirements**: Implements three service modes (auto, memory, script) with health checking and automatic failover.
- **Compute**: Lightweight service orchestration with minimal overhead.
- **Interface**: Provides unified service interface that abstracts away the underlying implementation details.
- **Key Features**:
  - **Service Mode Selection**: Automatic selection based on configuration, environment variables, and system health.
  - **Health Monitoring**: Continuous monitoring of model availability and system resources.
  - **Graceful Fallback**: Automatic transition to script-based mode when in-memory services are unavailable.
  - **Performance Benchmarking**: Tools for comparing performance between service modes.
  - **Model Warmup**: Preloading of models during application startup for optimal performance.
- **Core Classes & Logic**:
  - `ServiceMode`: Enumeration defining available service modes
  - `get_service_mode()`: Determines the active service mode based on configuration and environment
  - `should_use_in_memory_service()`: Health check for in-memory service availability
  - `run_full_pipeline()`: Smart pipeline execution with automatic fallback
  - `warmup_services()`: Model preloading for performance optimization
- **Dependencies**: Internal service implementations (v1 and v2).

### 2.7. Configuration Management (`src/omoai/config/`)

This component provides comprehensive configuration management with validation and type safety.

- **Area of Responsibility**: Centralized configuration loading, validation, and management for all system components.
- **Technical Requirements**: Uses Pydantic for type-safe configuration with environment variable support and YAML parsing.
- **Compute**: Lightweight configuration loading with caching for performance.
- **Interface**: Provides singleton access pattern with automatic configuration discovery and validation.
- **Key Features**:
  - **Type Safety**: Full type validation using Pydantic dataclasses
  - **Environment Variable Support**: Override any configuration value via environment variables
  - **Automatic Discovery**: Searches for configuration files in standard locations
  - **Validation**: Comprehensive validation with clear error messages
  - **Security**: Secure defaults with warnings for potentially dangerous settings
- **Core Classes & Logic**:
  - `src.omoai.config.schemas.OmoAIConfig`: Main configuration class with nested sub-configurations
  - `src.omoai.config.schemas.PathsConfig`: File and directory path configuration
  - `src.omoai.config.schemas.ASRConfig`: ASR-specific configuration
  - `src.omoai.config.schemas.LLMConfig`: LLM configuration with security settings
  - `src.omoai.config.schemas.APIConfig`: API server configuration
  - `get_config()`: Singleton accessor for configuration instance
- **Dependencies**: `pydantic`, `pydantic-settings`, `pyyaml`.

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
