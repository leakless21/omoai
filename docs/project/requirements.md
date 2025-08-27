# Project Requirements

## 1. Functional Requirements

### 1.1. Audio Preprocessing

- **FR1.1.1:** The system must accept audio files in various formats (e.g., MP3, WAV, FLAC).
- **FR1.1.2:** The system must convert input audio into a 16kHz mono PCM16 WAV format suitable for the ASR model.
- **FR1.1.3:** The system must handle long-form audio by processing it in manageable chunks.

### 1.2. Transcription (ASR)

- **FR1.2.1:** The system must use the `Chunkformer` model to transcribe preprocessed audio files into text.
- **FR1.2.2:** The transcription output must include timestamps for each transcribed word or segment.
- **FR1.2.3:** The system must generate a raw transcript as a JSON file ([`asr.json`](data/output/podacastmt-6acc5ce0/asr.json:0)).

### 1.3. Post-processing

- **FR1.3.1:** The system must apply punctuation and capitalization to the raw transcript using a Large Language Model (LLM).
- **FR1.3.2:** The system must generate a concise summary of the transcribed text using an LLM.
- **FR1.3.3:** The system must be configurable to enable or disable summarization and other post-processing steps.

### 1.4. Output Generation

- **FR1.4.1:** The system must produce a final JSON file ([`final.json`](data/output/podacastmt-6acc5ce0/final.json:0)) containing the punctuated transcript, summary, and timestamps.
- **FR1.4.2:** The system must optionally generate separate text files for the final transcript ([`transcript.txt`](data/output/podacastmt-6acc5ce0/transcript.txt:0)) and summary ([`summary.txt`](data/output/podacastmt-6acc5ce0/summary.txt:0)).
- **FR1.4.3:** The system must store all output artifacts in a structured directory within the [`data/output/`](data/output/:0) folder.

## 2. Non-Functional Requirements

### 2.1. Performance

- **NFR2.1.1:** The transcription process should be efficient for long-form audio, such as podcasts.
- **NFR2.1.2:** The system should leverage GPU acceleration for both the ASR and LLM models to ensure timely processing.
- **NFR2.1.3:** The system should have a low Word Error Rate (WER) to ensure high-quality transcriptions.

### 2.2. Configurability

- **NFR2.2.1:** All major parameters, including model paths, chunk sizes, and LLM settings, must be configurable via a central YAML file ([`config.yaml`](config.yaml:0)).
- **NFR2.2.2:** The system prompts for the LLM (punctuation and summarization) must be easily customizable.

### 2.3. Usability

- **NFR2.3.1:** The project should be easy to install and run, with all dependencies clearly defined in [`pyproject.toml`](pyproject.toml:0).
- **NFR2.3.2:** The command-line interface, managed by [`main.py`](main.py:0), must be simple and intuitive.

### 2.4. Maintainability

- **NFR2.4.1:** The codebase must be modular, with a clear separation of concerns between preprocessing, ASR, and post-processing scripts.
- **NFR2.4.2:** The project structure must be well-organized, with distinct directories for source code, models, data, and scripts.

### 2.5. Reliability

- **NFR2.5.1:** The system must include tests to ensure the core components function as expected.
- **NFR2.5.2:** The pipeline must handle potential errors gracefully, such as invalid input file formats.
