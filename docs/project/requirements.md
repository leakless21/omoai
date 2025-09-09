# Project Requirements

## 1. Functional Requirements

### 1.1. Audio Preprocessing

- **FR1.1.1:** The system must accept audio files in various formats (e.g., MP3, WAV, FLAC, M4A).
- **FR1.1.2:** The system must convert input audio into a 16kHz mono PCM16 WAV format suitable for the ASR model.
- **FR1.1.3:** The system must handle long-form audio by processing it in manageable chunks.
- **FR1.1.4:** The system must validate audio input and reject invalid or corrupted files.

### 1.2. Transcription (ASR)

- **FR1.2.1:** The system must use the `Chunkformer` model to transcribe preprocessed audio files into text.
- **FR1.2.2:** The transcription output must include timestamps for each transcribed word or segment.
- **FR1.2.3:** The system must generate a raw transcript with confidence scores for each segment.
- **FR1.2.4:** The system must support both GPU and CPU inference for ASR processing.

### 1.3. Post-processing

- **FR1.3.1:** The system must use an enhanced, few-shot prompt to ensure high-quality punctuation and capitalization.
- **FR1.3.2:** The system must generate a concise summary of the transcribed text using an LLM.
- **FR1.3.3:** The system must be configurable to enable or disable summarization and other post-processing steps.
- **FR1.3.4:** The system must provide quality metrics for punctuation alignment and transcription accuracy.
- **FR1.3.5:** The system must support both bullet points and abstract summary formats.

### 1.4. Output Generation

- **FR1.4.1:** The system must produce a final JSON file ([`final.json`](data/output/podacastmt-6acc5ce0/final.json:0)) containing the punctuated transcript, summary, and timestamps.
- **FR1.4.2:** The system must optionally generate separate text files for the final transcript ([`transcript.txt`](data/output/podacastmt-6acc5ce0/transcript.txt:0)) and summary ([`summary.txt`](data/output/podacastmt-6acc5ce0/summary.txt:0)).
- **FR1.4.3:** The system must store all output artifacts in a structured directory within the [`data/output/`](data/output/:0) folder.
- **FR1.4.4:** The system must support multiple output formats including JSON, plain text, SRT, VTT, and Markdown.
- **FR1.4.5:** The system must provide flexible timestamp formatting options (none, seconds, milliseconds, clock time).

### 1.5. API Interface

- **FR1.5.1:** The system must provide a RESTful API endpoint at `/pipeline` for processing audio files.
- **FR1.5.2:** The API must accept multipart/form-data requests with audio file uploads.
- **FR1.5.3:** The API must support query parameters for customizing output format and content.
- **FR1.5.4:** The API must provide individual endpoints for preprocessing (`/preprocess`), ASR (`/asr`), and post-processing (`/postprocess`).
- **FR1.5.5:** The API must include a health check endpoint (`/health`) for monitoring system status.
- **FR1.5.6:** The API must support multiple response formats based on client request.
- **FR1.5.7:** The API must provide comprehensive error handling with appropriate HTTP status codes.

### 1.6. Service Management

- **FR1.6.1:** The system must support dual service modes: high-performance in-memory processing and robust script-based processing.
- **FR1.6.2:** The system must automatically fallback between service modes based on availability and health status.
- **FR1.6.3:** The system must allow manual configuration of service mode through configuration file or environment variables.
- **FR1.6.4:** The system must cache loaded models for improved performance in consecutive requests.

## 2. Non-Functional Requirements

### 2.1. Performance

- **NFR2.1.1:** The transcription process should be efficient for long-form audio, such as podcasts (up to 3 hours).
- **NFR2.1.2:** The system should leverage GPU acceleration for both the ASR and LLM models to ensure timely processing.
- **NFR2.1.3:** The system should have a low Word Error Rate (WER) to ensure high-quality transcriptions.
- **NFR2.1.4:** The API should support concurrent processing of multiple requests.
- **NFR2.1.5:** The system should minimize cold start time through model preloading and caching.
- **NFR2.1.6:** The in-memory processing mode should provide significantly faster processing than script-based mode.

### 2.2. Configurability

- **NFR2.2.1:** All major parameters, including model paths, chunk sizes, and LLM settings, must be configurable via a central YAML file ([`config.yaml`](config.yaml:0)).
- **NFR2.2.2:** The system prompts for the LLM (punctuation and summarization) must be easily customizable.
- **NFR2.2.3:** API-specific settings (host, port, timeouts, file size limits) must be configurable.
- **NFR2.2.4:** The system must support environment variable overrides for configuration values.
- **NFR2.2.5:** The system must validate configuration values on startup and provide clear error messages for invalid settings.

### 2.3. Usability

- **NFR2.3.1:** The project should be easy to install and run, with all dependencies clearly defined in [`pyproject.toml`](pyproject.toml:0).
- **NFR2.3.2:** The command-line interface, managed by [`main.py`](main.py:0), must be simple and intuitive.
- **NFR2.3.3:** The API must provide interactive OpenAPI documentation for easy exploration and testing.
- **NFR2.3.4:** The system must provide clear example requests and responses for all endpoints.
- **NFR2.3.5:** The system must support multiple languages for summarization based on user preference.

### 2.4. Maintainability

- **NFR2.4.1:** The codebase must be modular, with a clear separation of concerns between preprocessing, ASR, and post-processing components.
- **NFR2.4.2:** The project structure must be well-organized, with distinct directories for source code, models, data, and scripts.
- **NFR2.4.3:** The system must use type hints and dataclasses for configuration validation and type safety.
- **NFR2.4.4:** The codebase must follow consistent coding standards and include comprehensive documentation.
- **NFR2.4.5:** The system must include automated tests for all major components with high code coverage.

### 2.5. Reliability

- **NFR2.5.1:** The system must include tests to ensure the core components function as expected.
- **NFR2.5.2:** The pipeline must handle potential errors gracefully, such as invalid input file formats.
- **NFR2.5.3:** The API must implement proper error handling with meaningful error messages.
- **NFR2.5.4:** The system must include health checks for all critical dependencies and components.
- **NFR2.5.5:** The system must log performance metrics and errors for monitoring and debugging.
- **NFR2.5.6:** The system must handle edge cases such as empty audio files, corrupted files, and extremely long audio.

### 2.6. Scalability

- **NFR2.6.1:** The API must handle multiple concurrent requests without significant performance degradation.
- **NFR2.6.2:** The system must support processing of audio files up to 100MB in size.
- **NFR2.6.3:** The system must implement appropriate rate limiting to prevent abuse.
- **NFR2.6.4:** The system must efficiently manage memory usage during processing of large audio files.

### 2.7. Security

- **NFR2.7.1:** The system must validate all input data to prevent injection attacks.
- **NFR2.7.2:** The API must include appropriate security headers for production deployments.
- **NFR2.7.3:** The system must include configurable settings for trusted remote code execution in LLMs.
- **NFR2.7.4:** Temporary files must be securely managed and cleaned up after processing.
- **NFR2.7.5:** The system must not expose sensitive information in error messages or logs.

### 2.8. Compatibility

- **NFR2.8.1:** The system must support multiple audio formats (MP3, WAV, FLAC, M4A).
- **NFR2.8.2:** The API must be compatible with standard HTTP clients and libraries.
- **NFR2.8.3:** The system must work across different operating systems (Linux, macOS, Windows).
- **NFR2.8.4:** The API response formats must follow industry standards for subtitles (SRT, VTT) and data exchange (JSON).

## 3. Quality Metrics Requirements

### 3.1. Transcription Quality

- **QMR3.1.1:** The system must calculate and provide Word Error Rate (WER) metrics.
- **QMR3.1.2:** The system must calculate and provide Character Error Rate (CER) metrics.
- **QMR3.1.3:** The system must provide confidence scores for each transcribed segment.
- **QMR3.1.4:** The system must offer quality comparison between raw and punctuated transcripts.

### 3.2. Processing Quality

- **QMR3.2.1:** The system must track and report processing time for each pipeline stage.
- **QMR3.2.2:** The system must calculate and provide real-time factor (RTF) for performance evaluation.
- **QMR3.2.3:** The system must monitor and report memory usage during processing.
- **QMR3.2.4:** The system must provide alignment confidence metrics for punctuation restoration.
