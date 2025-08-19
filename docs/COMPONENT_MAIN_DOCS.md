# Component: Main Application Domain

This document provides a detailed overview of the main application domain, which orchestrates the entire audio processing pipeline. This domain is responsible for managing the workflow, loading configurations, and coordinating the interaction between the preprocessing, ASR, and post-processing components.

## 1. Domain Overview

The main application domain is the central nervous system of the audio processing pipeline. It acts as a controller, ensuring that each component executes in the correct sequence and with the correct parameters. The domain is primarily embodied in the [`main.py`](main.py:0) script but also encompasses the configuration management and the scripts that perform the actual work.

### 1.1. Bounded Context

The "Main Application" bounded context includes:

- **Orchestration Logic:** The rules and sequence of operations for processing an audio file from raw input to final polished output.
- **Configuration Management:** The handling and application of settings defined in [`config.yaml`](config.yaml:0).
- **Workflow Execution:** The invocation and management of the individual processing scripts ([`preprocess.py`](scripts/preprocess.py:0), [`asr.py`](scripts/asr.py:0), [`post.py`](scripts/post.py:0)).
- **Entry Point:** The primary user interface for running the pipeline.

### 1.2. Key Responsibilities

- **Entry Point Management:** Provide a clear and simple command-line interface for initiating the audio processing pipeline.
- **Configuration Loading and Validation:** Load settings from [`config.yaml`](config.yaml:0), validate them, and make them available to the downstream components.
- **Workflow Coordination:** Execute the pipeline stages (preprocess, ASR, post-process) in the correct order, passing necessary information between them.
- **Error Handling:** Implement a top-level error-handling strategy to catch and report failures that occur during any stage of the pipeline.
- **Path and File Management:** Dynamically manage input and output file paths based on the configuration and input arguments.

## 2. Core Components and Classes

### 2.1. `main.py`

This is the primary script and entry point for the application. It contains the logic for parsing command-line arguments, loading the configuration, and orchestrating the entire pipeline.

**Location:** [`main.py`](main.py:0)

**Key Functions:**

- `main()`: The main execution function. It parses arguments, loads the configuration, and calls the respective scripts for each stage of the pipeline.
- `load_config()`: (Implied) A function to load and parse the [`config.yaml`](config.yaml:0) file.

**Interactions:**

- **User:** Receives the audio file path and optional configuration overrides from the command line.
- **[`config.yaml`](config.yaml:0):** Reads the global configuration for the application.
- **[`scripts/preprocess.py`](scripts/preprocess.py:0):** Invokes the preprocessing stage, passing the input audio file path and output directory.
- **[`scripts/asr.py`](scripts/asr.py:0):** Invokes the ASR stage, passing the path to the preprocessed audio file.
- **[`scripts/post.py`](scripts/post.py:0):** Invokes the post-processing stage, passing the path to the raw ASR output.

### 2.2. Configuration Management

The configuration is managed through the [`config.yaml`](config.yaml:0) file, which is central to the main application domain.

**Location:** [`config.yaml`](config.yaml:0)

**Key Sections:**

- **`paths`**: Defines the base paths for models and output directories.
  - `chunkformer_model_path`: The directory containing the pre-trained `Chunkformer` model.
  - `output_dir`: The root directory for all output artifacts.
- **`asr`**: Parameters for the Automatic Speech Recognition stage.
  - `chunk_size`: The size of audio chunks for processing.
  - `batch_duration`: The duration of batches for ASR processing.
- **`llm`**: Configuration for the Large Language Model used in post-processing.
  - `model`: The name or identifier of the LLM.
- **`punctuation`**: Settings for the punctuation restoration task.
  - `system_prompt`: The prompt to provide context to the LLM.
  - `temperature`, `top_p`: Sampling parameters for controlling LLM output randomness.
- **`summarization`**: Settings for the summarization task.
  - `system_prompt`: The prompt to guide the LLM in generating a summary.
  - `temperature`, `top_p`: Sampling parameters.

### 2.3. Processing Scripts (Domain Services)

These scripts are domain services that the main orchestrator activates. They encapsulate the specific logic for each stage of the pipeline.

#### 2.3.1. `scripts/preprocess.py`

**Location:** [`scripts/preprocess.py`](scripts/preprocess.py:0)

**Responsibility:** To prepare the raw audio file for the ASR model by converting it to the required format.

#### 2.3.2. `scripts/asr.py`

**Location:** [`scripts/asr.py`](scripts/asr.py:0)

**Responsibility:** To perform speech-to-text conversion on the preprocessed audio using the `Chunkformer` model.

#### 2.3.3. `scripts/post.py`

**Location:** [`scripts/post.py`](scripts/post.py:0)

**Responsibility:** To refine the raw transcript by applying punctuation and capitalization, and to generate a summary of the content.

## 3. Domain Events and Data Flow

The domain operates on a simple, sequential data flow:

1.  **`AudioFileProvided`**: The user provides an audio file path to [`main.py`](main.py:0).
2.  **`ConfigurationLoaded`**: [`main.py`](main.py:0) loads settings from [`config.yaml`](config.yaml:0).
3.  **`PreprocessingStarted`**: [`main.py`](main.py:0) calls [`scripts/preprocess.py`](scripts/preprocess.py:0).
4.  **`AudioPreprocessed`**: [`scripts/preprocess.py`](scripts/preprocess.py:0) outputs a standardized WAV file.
5.  **`TranscriptionStarted`**: [`main.py`](main.py:0) calls [`scripts/asr.py`](scripts/asr.py:0) with the preprocessed audio.
6.  **`TranscriptionComplete`**: [`scripts/asr.py`](scripts/asr.py:0) outputs a raw transcript in JSON format.
7.  **`PostProcessingStarted`**: [`main.py`](main.py:0) calls [`scripts/post.py`](scripts/post.py:0) with the raw transcript.
8.  **`ProcessingComplete`**: [`scripts/post.py`](scripts/post.py:0) generates the final, polished output files.

## 4. External Dependencies

This domain depends on several external libraries and tools to function:

- **`pyyaml`**: Used for parsing the [`config.yaml`](config.yaml:0) file.
- **`argparse`**: (Built-in Python library) Used for parsing command-line arguments in [`main.py`](main.py:0).
- **`subprocess`**: (Built-in Python library) Used by [`main.py`](main.py:0) to execute the processing scripts.
- **`ffmpeg`**: An external command-line tool used by [`scripts/preprocess.py`](scripts/preprocess.py:0) for audio format conversion.
