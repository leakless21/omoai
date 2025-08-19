# System Architecture

This document provides a detailed architectural overview of the audio processing pipeline, outlining the system's components, their responsibilities, and interactions.

## 1. High-Level Overview

The system is designed as a modular, end-to-end pipeline for processing audio files, with a primary focus on podcasts. The architecture is composed of three main stages: Preprocessing, Transcription (ASR), and Post-processing. Each stage is implemented as a distinct Python script, orchestrated by a central controller, [`main.py`](main.py:0).

The pipeline accepts an audio file as input, transforms it into a standardized format, transcribes it into text, and then refines the text by adding punctuation and generating a summary. The entire process is configurable through a central YAML configuration file, [`config.yaml`](config.yaml:0).

## 2. System Components

### 2.1. Main Orchestrator (`main.py`)

The [`main.py`](main.py:0) script serves as the entry point and central controller for the entire pipeline. Its responsibilities include:

- **Workflow Management:** It sequentially calls the preprocessing, ASR, and post-processing scripts.
- **Configuration Loading:** It loads and parses the [`config.yaml`](config.yaml:0) file to obtain all necessary parameters.
- **Path Management:** It constructs file paths for input and output based on the configuration and input arguments.
- **Error Handling:** It provides a top-level error-handling mechanism to ensure failures in one stage are reported clearly.

**Interfaces:**

- **Input:** Command-line arguments for the audio file path and optional overrides for the configuration.
- **Output:** Calls to subsequent scripts via subprocess.
- **Dependencies:** [`config.yaml`](config.yaml:0), [`scripts/preprocess.py`](scripts/preprocess.py:0), [`scripts/asr.py`](scripts/asr.py:0), [`scripts/post.py`](scripts/post.py:0).

### 2.2. Preprocessing Component (`scripts/preprocess.py`)

This script is responsible for preparing the raw audio file for the ASR model.

- **Area of Responsibility:** Audio format conversion and normalization.
- **Technical Requirements:** Must use `ffmpeg` to ensure audio is converted to 16kHz mono PCM16 WAV format.
- **Compute:** CPU-bound, as it involves audio transcoding.
- **Storage:** Reads the input audio file and writes the preprocessed WAV file to the output directory.
- **Interface:** Takes the raw audio file path and the designated output path as input. It outputs the path to the preprocessed WAV file.
- **Dependencies:** `pydub` (for audio manipulation), `ffmpeg`.

### 2.3. ASR Component (`scripts/asr.py`)

The ASR component is the heart of the speech recognition process, utilizing the `Chunkformer` model.

- **Area of Responsibility:** Converting preprocessed audio into a text transcript with timestamps.
- **Technical Requirements:** Must load the pre-trained `Chunkformer` model from the path specified in [`config.yaml`](config.yaml:0). It must process audio in chunks to handle long-form content efficiently.
- **Compute:** GPU-bound for optimal performance, leveraging `torch` for neural network inference.
- **Storage:** Reads the preprocessed WAV file and writes the raw ASR output as a JSON file.
- **Interface:** Takes the path to the preprocessed audio and the output directory as input. It outputs a JSON file containing the transcript and timestamps.
- **Dependencies:** `torch`, `torchaudio`, the `Chunkformer` model files located in [`chunkformer/`](chunkformer/:0) and [`models/`](models/:0).

### 2.4. Post-processing Component (`scripts/post.py`)

This component refines the raw transcript from the ASR stage.

- **Area of Responsibility:** Applying punctuation and capitalization, and generating a summary.
- **Technical Requirements:** Must use the specified LLM (via `vllm`) for both punctuation and summarization. The system prompts and sampling parameters are configurable in [`config.yaml`](config.yaml:0).
- **Compute:** GPU-bound, as it involves running the LLM.
- **Storage:** Reads the raw ASR JSON file and writes the final polished JSON, transcript text, and summary text files.
- **Interface:** Takes the raw ASR JSON file path and the output directory as input. It produces multiple output files.
- **Dependencies:** `vllm` (for LLM inference), `pyyaml` (for accessing configuration).

For a detailed explanation of the post-processing script, including its core functions, logic, and configuration, please refer to [`docs/COMPONENT_POST_DOCS.md`](docs/COMPONENT_POST_DOCS.md:0).

### 2.5. `Chunkformer` Model Domain

This domain encapsulates the custom ASR model.

- **Area of Responsibility:** The core speech recognition logic.
- **Technical Requirements:** A hybrid CTC-attention model implemented in PyTorch.
- **Key Components:**
  - [`ASRModel`](chunkformer/model/asr_model.py:17): The main model class.
  - [`chunkformer/decode.py`](chunkformer/decode.py:0): Contains the decoding logic, including [`endless_decode`](chunkformer/decode.py:51) for long-form audio and [`batch_decode`](chunkformer/decode.py:130) for batch processing.
- **Compute:** GPU-bound for training and inference.
- **Storage:** The model weights are stored in [`models/chunkformer/`](models/chunkformer/:0).
- **Interface:** Exposes methods for encoding and decoding audio.
- **Dependencies:** `torch`, `torchaudio`.

## 3. Data Flow

The data flows through the system in a linear, sequential manner:

1. **Input:** An audio file (e.g., `podcast.mp3`) is provided to [`main.py`](main.py:0).
2. **Preprocessing:** [`scripts/preprocess.py`](scripts/preprocess.py:0) converts `podcast.mp3` to `preprocessed.wav`.
3. **ASR:** [`scripts/asr.py`](scripts/asr.py:0) takes `preprocessed.wav` and generates `asr.json`.
4. **Post-processing:** [`scripts/post.py`](scripts/post.py:0) reads `asr.json` and produces `final.json`, `transcript.txt`, and `summary.txt`.
5. **Output:** All processed files are stored in the `data/output/` directory, organized by a unique identifier for each run.

## 4. Configuration

The entire system's behavior is controlled by [`config.yaml`](config.yaml:0), which is structured into the following sections:

- **`paths`**: Defines the base path for the `Chunkformer` model (`chunkformer_model_path`) and the output directory (`output_dir`).
- **`asr`**: Configures ASR-specific parameters like `chunk_size` and `batch_duration`.
- **`llm`**: Specifies the LLM to be used (e.g., model name, API key if applicable).
- **`punctuation`**: Contains the system prompt (`system_prompt`) and sampling parameters (e.g., `temperature`, `top_p`) for the punctuation task.
- **`summarization`**: Contains the system prompt (`system_prompt`) and sampling parameters for the summarization task.

## 5. Compute and Storage

### 5.1. Compute Requirements

- **Preprocessing:** CPU.
- **ASR:** GPU recommended for faster processing of long audio files.
- **Post-processing (LLM):** GPU recommended for efficient LLM inference.
- **Dependencies:** The project relies on `torch`, `torchaudio`, and `vllm`, which are optimized for GPU acceleration.

### 5.2. Storage Requirements

- **Input Data:** Raw audio files stored in [`data/input/`](data/input/:0).
- **Model Weights:** Pre-trained `Chunkformer` model stored in [`models/`](models/:0).
- **Output Data:** All generated files (WAV, JSON, TXT) are stored in [`data/output/`](data/output/:0), with each processing run creating a uniquely named subdirectory.
- **Intermediate Data:** Preprocessed WAV files are stored temporarily in the output directory before being consumed by the ASR script.

## 6. External Dependencies

The project's dependencies are managed via [`pyproject.toml`](pyproject.toml:0) and include:

- **`torch`, `torchaudio`**: For deep learning and audio processing.
- **`vllm`**: For efficient execution of Large Language Models.
- **`pydub`**: For audio manipulation.
- **`pyyaml`**: For parsing the configuration file.
- **`jiwer`**: For calculating Word Error Rate (WER), primarily used for evaluation.

## 7. For More Information

For more detailed information on each component, please refer to the following documentation:

- **Post-processing Script:** [`docs/COMPONENT_POST_DOCS.md`](docs/COMPONENT_POST_DOCS.md:0)
- **`Chunkformer` Model:** [`docs/COMPONENT_CHUNKFORMER_DOCS.md`](docs/COMPONENT_CHUNKFORMER_DOCS.md:0)
- **Main Orchestrator:** [`docs/COMPONENT_MAIN_DOCS.md`](docs/COMPONENT_MAIN_DOCS.md:0)
