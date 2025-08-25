# Speaker Embedding and Diarization Implementation Plan

## 1. Introduction & Goal

This document outlines a comprehensive plan to integrate speaker diarization capabilities into the OMOAI pipeline. The primary goal is to identify unique speakers in an audio input and attribute the transcribed text to each speaker. This will enrich the final output, making it more readable and useful for analysis.

This plan leverages the `speechbrain/spkrec-ecapa-voxceleb` model for generating high-quality speaker embeddings and the `pyannote.audio` library for a robust, pre-trained diarization pipeline.

## 2. Analysis of Current System

- **Modular Pipeline:** The current system (`src/omoai/pipeline/pipeline.py`) is well-structured with distinct stages: `preprocess`, `asr`, and `postprocess`. This modularity allows for the clean insertion of a new `diarization` stage.
- **Timed ASR Segments:** The `ChunkFormerASR` class (`src/omoai/pipeline/asr.py`) produces timed segments (`ASRSegment`), which is a critical prerequisite for aligning transcription with speaker turns.
- **Existing Dependencies:** `speechbrain` is already listed in `pyproject.toml`, which simplifies the integration of the embedding model.

## 3. Proposed Implementation Plan

The implementation is broken down into six phases, designed for incremental development and testing.

### Phase 1: Core Dependencies & Model Setup

1.  **Add `pyannote.audio`:** This library provides a complete, pre-trained pipeline for speaker diarization that includes Voice Activity Detection (VAD), speaker embedding, and clustering. Add it to the `dependencies` list in `pyproject.toml`.
    ```toml
    # In pyproject.toml
    dependencies = [
        # ... existing dependencies
        "pyannote.audio>=3.1.1",
    ]
    ```
2.  **Hugging Face Token:** The `pyannote.audio` pipelines require authentication with a Hugging Face token. This will need to be handled in the configuration or environment variables.

### Phase 2: Diarization Pipeline Module

1.  **Create New Module:** Create a new file: `src/omoai/pipeline/diarization.py`.
2.  **Implement `SpeakerDiarizer` Class:**
    - This class will initialize the `pyannote.audio` diarization pipeline.
    - It will be configured to use the specified `speechbrain/spkrec-ecapa-voxceleb` model for embeddings to ensure high accuracy.
    - It will expose a primary method, `diarize(audio_tensor, sample_rate)`, which takes the preprocessed audio tensor and returns a list of speaker segments with timestamps and speaker labels (e.g., `[(speaker_0, start_time, end_time), ...]`).

### Phase 3: Integration into the Main Pipeline

1.  **Modify `run_full_pipeline_memory`:** In `src/omoai/pipeline/pipeline.py`, add a new "Diarization" stage between ASR and Postprocessing.
2.  **Conditional Execution:** This stage will be executed only if `diarization.enabled` is `true` in the configuration.
3.  **Merge ASR and Diarization Results:**
    - The core task of this stage is to map the speaker labels from the diarization output to the ASR segments.
    - For each `ASRSegment` from the ASR result, iterate through the speaker turn annotations from the diarizer.
    - Assign the speaker label (`speaker_0`, `speaker_1`, etc.) that has the largest temporal overlap with the ASR segment.
    - The `ASRSegment` dataclass will be updated to store this speaker label.

### Phase 4: Configuration

1.  **Update `config.yaml`:** Add a new `diarization` section to the main configuration file.
    ```yaml
    # In config.yaml
    diarization:
      enabled: true
      # Hugging Face model for pyannote pipeline
      pipeline_model: "pyannote/speaker-diarization-hfin-v2.1" 
      # The embedding model to be used by the pyannote pipeline
      embedding_model: "speechbrain/spkrec-ecapa-voxceleb"
      # Hugging Face access token (can also be an environment variable)
      hf_token: "YOUR_HF_TOKEN_HERE" 
    ```
2.  **Update Config Schemas:** Update the Pydantic models in `src/omoai/config/schemas.py` to reflect these new settings, ensuring type safety and validation.

### Phase 5: Output Modification

1.  **Update `ASRSegment`:** Modify the `ASRSegment` dataclass in `src/omoai/pipeline/asr.py` (or a more central types file) to include the speaker.
    ```python
    # In src/omoai/pipeline/asr.py
    @dataclass
    class ASRSegment:
        start: float
        end: float
        text: str
        confidence: Optional[float] = None
        speaker: Optional[str] = None # New field
    ```
2.  **Update Output Formatters:** Modify the output plugins in `src/omoai/output/plugins/` to utilize the new `speaker` field.
    - **`text.py` / `srt.py` / `vtt.py`:** Prepend the speaker label to each segment (e.g., `[00:00:05.234 -> 00:00:08.123] Speaker A: Hello, world.`).
    - **`json.py`:** Add the `speaker` field to the segment objects in the JSON output.

### Phase 6: Testing

1.  **Create New Test Suite:** Create a new test file: `tests/test_diarization.py`.
2.  **Multi-Speaker Audio Fixture:** Add a test audio file to `tests/fixtures/` that contains at least two clearly distinguishable speakers.
3.  **Write Integration Test:**
    - The test will run the full pipeline on the multi-speaker audio file with diarization enabled.
    - It will assert that the resulting segments contain more than one unique speaker label.
    - It will check that the speaker labels are correctly formatted in the final output files (e.g., SRT, JSON).

## 4. Documentation Updates

- **`ARCHITECTURE.md`:** Update the architecture diagram and description to include the new Diarization component in the pipeline.
- **`COMPONENT_DIARIZATION_DOCS.md`:** Create a new component documentation file that details the `SpeakerDiarizer` class, its configuration, and its role in the system, following the project's documentation standards.
- **`REQUIREMENTS.md`:** Add a new functional requirement for speaker identification.

## 5. Summary of File Changes

- **New Files:**
    - `src/omoai/pipeline/diarization.py`
    - `tests/test_diarization.py`
    - `docs/COMPONENT_DIARIZATION_DOCS.md`
    - `tests/fixtures/multi_speaker_audio.wav` (or similar)
- **Modified Files:**
    - `pyproject.toml`
    - `config.yaml`
    - `src/omoai/pipeline/pipeline.py`
    - `src/omoai/pipeline/asr.py` (or a central types file for `ASRSegment`)
    - `src/omoai/config/schemas.py`
    - `src/omoai/output/plugins/json.py`
    - `src/omoai/output/plugins/srt.py`
    - `src/omoai/output/plugins/text.py`
    - `src/omoai/output/plugins/vtt.py`
    - `docs/ARCHITECTURE.md`
    - `docs/REQUIREMENTS.md`
