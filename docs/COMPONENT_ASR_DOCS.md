# ASR Component Documentation

## Overview

The Automatic Speech Recognition (ASR) component is a core part of the OmoAI project, responsible for converting spoken language in audio files into written text. This component is designed to be robust, scalable, and easily maintainable, leveraging state-of-the-art ASR models and a streamlined processing pipeline.

## Domain-Driven Design

The ASR component is treated as a distinct domain within the OmoAI ecosystem, with its own well-defined boundaries, responsibilities, and interfaces. This design ensures that the ASR logic remains encapsulated and can evolve independently of other components.

## Responsibilities

- **Audio Input Handling**: Accept and validate audio files in various formats.
- **Feature Extraction**: Extract relevant acoustic features from the audio signal.
- **Acoustic and Language Modeling**: Utilize pre-trained models to convert audio features into phonemes and subsequently into words.
- **Transcription Generation**: Produce a textual transcript of the audio, including word-level timing information.
- **API Endpoint Management**: Expose a FastAPI endpoint for ASR requests, handling input validation and response formatting.

## Classes and Files

### `ASRController`

- **File**: [`src/omoai/api/asr_controller.py`](src/omoai/api/asr_controller.py)
- **Description**: This class defines the FastAPI router for ASR-related endpoints. It handles incoming requests, interacts with the ASR service layer, and returns the transcribed results.
- **Key Methods**:
  - `transcribe_audio(audio: UploadFile, background_tasks: BackgroundTasks, model: str = "base")`: Handles the `/transcribe` endpoint, accepts an audio file, and returns the transcription.

### `ASRService`

- **File**: [`src/omoai/api/services.py`](src/omoai/api/services.py)
- **Description**: Provides the core business logic for ASR processing. It acts as an intermediary between the API controller and the underlying ASR model wrappers.
- **Key Methods**:
  - `transcribe(audio_data: bytes, model_name: str) -> dict`: Takes raw audio data and a model name, performs the transcription, and returns the result.

### `ASRModelSingleton`

- **File**: [`src/omoai/api/singletons.py`](src/omoai/api/singletons.py)
- **Description**: A singleton class that manages the lifecycle and access to the ASR model instances. It ensures that models are loaded only once and are efficiently reused across multiple requests.
- **Key Methods**:
  - `get_model(model_name: str) -> ASRModel`: Retrieves a pre-loaded ASR model instance by name.

### `ASRModel` (Abstract Base Class)

- **File**: [`src/omoai/api/models.py`](src/omoai/api/models.py)
- **Description**: An abstract base class that defines the interface for all ASR model implementations. This allows for different ASR backends (e.g., Whisper, ChunkFormer) to be plugged in seamlessly.
- **Key Methods**:
  - `transcribe(audio_data: bytes) -> dict`: Abstract method that must be implemented by concrete ASR model classes.

### `WhisperModel` (Concrete Implementation)

- **File**: [`src/omoai/api/models.py`](src/omoai/api/models.py)
- **Description**: A concrete implementation of the `ASRModel` interface for the OpenAI Whisper model. It handles loading the Whisper model and performing transcription.
- **Key Methods**:
  - `transcribe(audio_data: bytes) -> dict`: Implements the transcription logic for Whisper.

### `ChunkFormerModel` (Concrete Implementation)

- **File**: [`src/omoai/api/models.py`](src/omoai/api/models.py)
- **Description**: A concrete implementation of the `ASRModel` interface for the ChunkFormer model. It handles loading the ChunkFormer model and performing transcription.
- **Key Methods**:
  - `transcribe(audio_data: bytes) -> dict`: Implements the transcription logic for ChunkFormer.

## Refactoring History

### Removal of Legacy `ASRModelSingleton`

- **Date**: 2025-09-05
- **Description**: The legacy `ASRModelSingleton` class and its associated ASR processing logic were removed from [`src/omoai/api/asr_controller.py`](src/omoai/api/asr_controller.py). The controller now exclusively uses the modern, centralized service layer for ASR processing. This change simplified the architecture and removed redundant code.

### Resolution of Class Name Collision

- **Date**: 2025-09-05
- **Description**: The `ASRModel` class in [`src/omoai/api/asr_controller.py`](src/omoai/api/asr_controller.py) was renamed to `ASRModelSingleton` to avoid confusion with the core PyTorch model. This improved code clarity and prevented potential naming conflicts.

### Removal of Duplicated `_add_basic_punctuation` Function

- **Date**: 2025-09-05
- **Description**: A duplicated `_add_basic_punctuation` function was removed from [`scripts/post.py`](scripts/post.py), leaving a single, robust implementation. This refactoring improved code maintainability and reduced redundancy.

## Interfaces

### API Endpoint

- **Path**: `/asr/transcribe`
- **Method**: `POST`
- **Request Body**: Multipart form data containing the audio file.
- **Response**: JSON object with the transcribed text and timing information.

  ```json
  {
    "segments": [
      {
        "text": "Hello, world!",
        "start": 0.5,
        "end": 1.2
      }
    ]
  }
  ```

### Internal Service Interface

- **Input**: Raw audio data (bytes) and the name of the model to use.
- **Output**: A dictionary containing the transcription result, structured as shown above.

## Dependencies

- **External Libraries**:
  - `fastapi`: For building the API interface.
  - `python-multipart`: For handling file uploads.
  - `torch`: PyTorch, for deep learning model inference.
  - `transformers`: Hugging Face Transformers, for accessing pre-trained models like Whisper.
  - `chunkformer`: The custom ChunkFormer ASR model.
- **Internal Components**:
  - Relies on the `singletons` module for model management.
  - Interacts with the `logging` module for structured logging.

## Future Enhancements

- **Support for more ASR engines**: Integrate additional ASR backends through the `ASRModel` interface.
- **Real-time streaming**: Extend the API to support real-time audio stream transcription.
- **Model versioning**: Implement a system to manage and serve different versions of ASR models.
- **Diarization**: Add speaker diarization capabilities to distinguish between different speakers in an audio file.
