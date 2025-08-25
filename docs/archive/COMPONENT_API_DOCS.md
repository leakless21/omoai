# API Component Documentation

This document provides a detailed overview of the API component, which exposes the audio processing pipeline via a RESTful API.

## 1. Overview

The API component is built using the Litestar framework and provides an interface for interact with the audio processing pipeline. It allows users to submit audio files and receive processed transcripts and summaries. The API is organized into controllers, each handling a specific part of the pipeline.

## 2. Controllers

### 2.1. MainController (`src/omoai/api/main_controller.py`)

The `MainController` is the central point of contact for the API. It provides the main endpoint to run the entire audio processing pipeline.

- **Responsibilities:** Orchestrates the entire audio processing pipeline by coordinating with the other controllers.
- **Key Methods:**
  - `pipeline`: Accepts an audio file and processes it through the entire pipeline, returning the final transcript and summary.

### 2.2. PreprocessController (`src/omoai/api/preprocess_controller.py`)

The `PreprocessController` handles the audio preprocessing stage.

- **Responsibilities:** Converts raw audio files into a standardized format suitable for ASR.
- **Key Methods:**
  - `preprocess`: Accepts a raw audio file and returns the preprocessed WAV file.

### 2.3. ASRController (`src/omoai/api/asr_controller.py`)

The `ASRController` manages the Automatic Speech Recognition (ASR) stage.

- **Responsibilities:** Transcribes preprocessed audio into text with timestamps using the Chunkformer model. Implements efficient model loading and management.
- **Key Classes:**
  - [`ASRModel`](src/omoai/api/asr_controller.py:15): Singleton class that manages model loading and audio processing.
- **Key Methods:**
  - `asr`: Accepts a preprocessed audio file path and returns the transcript and segments. Uses the loaded ASR model for efficient processing.</search>

### 2.4. PostprocessController (`src/omoai/api/postprocess_controller.py`)

The `PostprocessController` handles the final stage of the pipeline, refining the transcript.

- **Responsibilities:** Applies punctuation, capitalization, and generates a summary of the transcript.
- **Key Methods:**
  - `postprocess`: Accepts a raw ASR transcript and returns the polished transcript, summary, and other relevant metadata.

## 3. API Endpoints

The API exposes the following endpoints:

- **`/pipeline`**: POST endpoint to run the entire audio processing pipeline. Accepts a multipart/form-data request with an audio file.
- **`/preprocess`**: POST endpoint to preprocess an audio file. Accepts a multipart/form-data request with an audio file.
- **`/asr`**: POST endpoint to transcribe audio. Accepts a multipart/form-data request with an audio file.
- **`/postprocess`**: POST endpoint to post-process a transcript. Accepts a JSON payload containing the raw transcript.

## 4. Data Models

The API uses Pydantic models for data validation and serialization. These models are defined in [`src/omoai/api/models.py`](src/omoai/api/models.py:0).

### 4.1. Request Models

- **`PipelineRequest`**: Defines the structure for requests to the `/pipeline` endpoint.
- **`PreprocessRequest`**: Defines the structure for requests to the `/preprocess` endpoint.
- **`ASRRequest`**: Defines the structure for requests to the `/asr` endpoint.
- **`PostprocessRequest`**: Defines the structure for requests to the `/postprocess` endpoint.

### 4.2. Response Models

- **`PipelineResponse`**: Defines the structure for responses from the `/pipeline` endpoint.
- **`PreprocessResponse`**: Defines the structure for responses from the `/preprocess` endpoint.
- **`ASRResponse`**: Defines the structure for responses from the `/asr` endpoint.
- **`PostprocessResponse`**: Defines the structure for responses from the `/postprocess` endpoint.

## 5. File Structure

The API component is located in the `src/omoai/api/` directory and is structured as follows:

```
src/omoai/api/
├── app.py              # Main Litestar application instance
├── models.py           # Pydantic models for request/response validation
├── main_controller.py  # Main controller for the pipeline endpoint
├── preprocess_controller.py # Controller for preprocessing
├── asr_controller.py   # Controller for ASR transcription
└── postprocess_controller.py # Controller for post-processing
```

### 5.1. Key Files

- **[`src/omoai/api/app.py`](src/omoai/api/app.py:0)**: Contains the main Litestar application instance and registers all the controllers.
- **[`src/omoai/api/models.py`](src/omoai/api/models.py:0)**: Defines all the Pydantic models used for API requests and responses.
- **[`src/omoai/api/main_controller.py`](src/omoai/api/main_controller.py:0)**: Implements the `MainController` which handles the main pipeline endpoint.
- **[`src/omoai/api/preprocess_controller.py`](src/omoai/api/preprocess_controller.py:0)**: Implements the `PreprocessController` for audio preprocessing.
- **[`src/omoai/api/asr_controller.py`](src/omoai/api/asr_controller.py:0)**: Implements the `ASRController` for speech-to-text transcription.
- **[`src/omoai/api/postprocess_controller.py`](src/omoai/api/postprocess_controller.py:0)**: Implements the `PostprocessController` for text refinement and summarization.

## 6. Dependencies

The API component has the following dependencies:

- **`litestar`**: The web framework used to build the API.
- **`pydantic`**: For data validation and serialization.
- **`uvicorn`**: The ASGI server used to run the application.
- **`torch`**: For deep learning and neural network operations (ASR processing).
- **`torchaudio`**: For audio processing and feature extraction.
- **`pydub`**: For audio format conversion and manipulation.
- **`pyyaml`**: For loading configuration files.

### 6.1. Model Loading

The API implements efficient model loading through:

- **Startup Initialization**: ASR models are loaded once when the application starts via the `on_startup` hook in [`app.py`](src/omoai/api/app.py:10).
- **Singleton Pattern**: The [`ASRModel`](src/omoai/api/asr_controller.py:15) class ensures models are loaded only once and reused across requests.
- **State Management**: Models are stored in the Litestar application state for efficient access.</search>
