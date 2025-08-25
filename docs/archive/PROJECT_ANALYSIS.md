# Project Structure Analysis

This document provides an overview of the project's structure, outlining the main components and their responsibilities.

## High-Level Architecture

The project is designed as a pipeline for processing audio files, performing Automatic Speech Recognition (ASR), and post-processing the output to improve readability and add speaker information. The main components are organized into separate directories, each with a specific role.

## Key Components

### 1. `main.py`

- **Purpose**: The main entry point of the application.
- **Responsibilities**:
  - Parses command-line arguments.
  - Orchestrates the execution of the different pipeline stages (preprocessing, ASR, post-processing).
  - Handles configuration loading and logging.

### 2. `interactive_cli.py`

- **Purpose**: Provides an interactive command-line interface (CLI) for running the pipeline.
- **Responsibilities**:
  - Offers a user-friendly way to interact with the application.
  - Guides the user through the process of selecting input files and configuring the pipeline.

### 3. `scripts/` directory

This directory contains the core scripts that implement the different stages of the audio processing pipeline.

- **`preprocess.py`**:

  - **Purpose**: Prepares the raw audio files for the ASR model.
  - **Responsibilities**:
    - Audio format conversion.
    - Resampling to the required sample rate.
    - Noise reduction and other audio enhancements.

- **`asr.py`**:

  - **Purpose**: Performs Automatic Speech Recognition on the preprocessed audio.
  - **Responsibilities**:
    - Loads the pre-trained ASR model.
    - Transcribes the audio files to text.
    - Outputs the raw ASR results in a structured format (e.g., JSON).

- **`post.py`**:
  - **Purpose**: Post-processes the raw ASR output to improve its quality and add additional information.
  - **Responsibilities**:
    - Punctuation restoration.
    - Truecasing (correcting the capitalization of words).
    - Mapping speaker diarization information to the transcribed text.

### 4. `chunkformer/` directory

- **Purpose**: A git submodule containing the Chunk-based Transformer model for ASR.
- **Responsibilities**:
  - Provides the core ASR engine.
  - Includes the model architecture, training scripts, and decoding logic.

### 5. `models/` directory

- **Purpose**: Stores the pre-trained ASR models.
- **Responsibilities**:
  - Contains the model weights and configuration files.

### 6. `data/` directory

- **Purpose**: Manages the input and output data of the pipeline.
- **Structure**:
  - **`input/`**: Contains the raw audio files to be processed.
  - **`output/`**: Stores the results of each pipeline stage, including preprocessed audio, ASR output, and final post-processed text.

### 7. `docs/` directory

- **Purpose**: Contains all the documentation related to the project.
- **Responsibilities**:
  - Provides detailed information about the project's architecture, components, and usage.

### 8. `tests/` directory

- **Purpose**: Contains the tests for the project.
- **Responsibilities**:
  - Ensures the correctness and reliability of the codebase.
  - Includes unit tests, integration tests, and end-to-end tests.
