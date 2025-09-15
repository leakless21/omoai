# Phoneme Alignment Integration Plan (Version 2)

## 1. Introduction and Goals

This document provides a detailed, step-by-step plan for integrating a phoneme-level timestamp alignment module into the `omoai` ASR pipeline. The primary objective is to replicate the core functionality of the `whisperx` library, as seen in the `ref/` directory, while adapting its architecture to our existing codebase.

The key goals are:

- **High-Quality Word Timestamps:** To produce accurate word-level timestamps for ASR transcriptions.
- **Modular and Extensible Design:** To create a reusable alignment module that can be easily maintained and extended.
- **Seamless Integration:** To ensure the new functionality is well-integrated with our existing CLI, API, and configuration systems.
- **Support for Multiple Languages:** To build a system that can be extended to support various languages, with a focus on both English and Vietnamese.

## 2. Analysis of the Reference Implementation (`whisperx`)

The `whisperx` implementation is structured around a few key components:

- **ASR Model (`asr.py`):** A wrapper around `faster-whisper` that handles the initial transcription.
- **Alignment Model (`alignment.py`):** A wav2vec2-based model that performs the forced alignment.
- **CLI (`__main__.py`):** A command-line interface that orchestrates the transcription and alignment process.
- **Python API:** A set of functions (`load_model`, `load_align_model`, `align`) that allow for programmatic access to the functionality.

Our implementation will follow a similar pattern, with a clear separation of concerns between the ASR and alignment modules.

## 3. Detailed Integration Plan

### 3.1. Dependency Management

While `torch` and `torchaudio` are already available, we will need to add a dependency for Vietnamese tokenization.

- ✅ **Action:** Add `underthesea` to the `[project.dependencies]` section of `pyproject.toml`.

  ```toml
  # pyproject.toml
  [project]
  dependencies = [
      # ... existing dependencies
      "underthesea>=1.3.1",
  ]
  ```

### 3.2. Configuration (`config.yaml` and Pydantic Schemas)

We will introduce a new `alignment` section in our `config.yaml` to manage alignment-specific settings.

- ✅ **Action:** Update `src/omoai/config/schemas.py` with a new `AlignmentConfig` Pydantic model.

  ```python
  # src/omoai/config/schemas.py
  from pydantic import BaseModel, Field
  from typing import Literal

  class AlignmentConfig(BaseModel):
      enabled: bool = Field(
          default=False,
          description="Whether to enable phoneme alignment."
      )
      model_name: str = Field(
          "WAV2VEC2_ASR_BASE_960H",
          description="Name of the alignment model to use."
      )
      language: str = Field(
          "en",
          description="Language code for the alignment model."
      )
      tokenizer: Literal["nltk", "underthesea"] = Field(
          "nltk",
          description="The tokenizer to use for sentence splitting."
      )

  class AppConfig(BaseModel):
      # ... existing AppConfig ...
      alignment: AlignmentConfig = Field(
          default_factory=AlignmentConfig,
          description="Configuration for the alignment model."
      )
  ```

- ✅ **Action:** Update `config.yaml` with the new `alignment` section.

  ```yaml
  # config.yaml
  asr:
    # ... existing ASR config ...

  alignment:
    enabled: true
    model_name: "nguyenvulebinh/wav2vec2-base-vi"
    language: "vi"
    tokenizer: "underthesea"
  ```

### 3.3. Code Implementation (`src/omoai/integrations/alignment.py`)

A new module will be created to house the alignment logic.

- ✅ **Action:** Create a new file at `src/omoai/integrations/alignment.py`.

This module will contain the following key functions:

- **`load_align_model()`:** This function will be responsible for loading the pre-trained wav2vec2 model from either `torchaudio` or Hugging Face, depending on the language. It will closely mirror the logic in `ref/whisperx/alignment.py`.

- **`align()`:** This will be the main function for performing the alignment. It will take the ASR segments, the alignment model, and the audio data as input, and will return the aligned segments with word-level timestamps. The implementation will be adapted from `ref/whisperx/alignment.py`, with the addition of a configurable tokenizer.

- **Tokenizer Logic:** The `align` function will include logic to dynamically select the appropriate sentence tokenizer based on the `tokenizer` configuration.

  ```python
  # src/omoai/integrations/alignment.py (conceptual)
  def align(segments, model, metadata, audio, device, tokenizer):
      if tokenizer == "underthesea":
          from underthesea import sent_tokenize
          # ... use underthesea for sentence splitting ...
      else:
          import nltk
          # ... use nltk for sentence splitting ...

      # ... rest of alignment logic ...
  ```

### 3.4. CLI Integration (`scripts/asr.py`)

The `scripts/asr.py` script will be updated to support the new alignment functionality.

- ✅ **Action:** Add new command-line arguments to `scripts/asr.py` using `argparse`.

  - `--align`: Enable alignment.
  - `--align_model`: Specify the alignment model.
  - `--language`: Specify the language.
  - `--tokenizer`: Specify the tokenizer.

- ✅ **Action:** Update the main logic of `scripts/asr.py` to call the alignment functions when the `--align` flag is present.

  ```python
  # scripts/asr.py (conceptual)
  if args.align:
      from omoai.integrations.alignment import load_align_model, align

      align_model, metadata = load_align_model(
          language_code=args.language,
          device=args.device
      )
      aligned_result = align(
          transcription_result["segments"],
          align_model,
          metadata,
          audio,
          args.device,
          tokenizer=args.tokenizer
      )
      # ... save aligned_result to output file ...
  ```

  The script has been updated to automatically detect the `alignment.enabled` setting from `config.yaml` and call the `align` function from [`src/omoai/integrations/alignment.py`](src/omoai/integrations/alignment.py) if enabled.

### 3.5. Output Structure

The JSON output will be structured to include both the original segments and the new word-level timestamp information, as defined in the previous plan.

### 3.6. Usage in the Pipeline

The phoneme alignment feature is designed to be seamlessly integrated into the existing ASR pipeline.

- **Configuration:** The feature is enabled by setting `alignment.enabled: true` in [`config.yaml`](config.yaml).
- **Detection:** The [`scripts/asr.py`](scripts/asr.py) script automatically detects this setting during execution.
- **Execution:** If enabled, after the initial ASR transcription is complete, the script calls the `align` function from [`src/omoai/integrations/alignment.py`](src/omoai/integrations/alignment.py).
- **Output:** The alignment module enriches the final JSON output with `word_segments` and `segments[*].words` arrays, providing detailed word-level timestamps for the transcribed text.

## 4. Testing Strategy

- **Unit Tests:** Unit tests will be created for the `align` and `load_align_model` functions to verify their correctness in isolation.
- **Integration Tests:** Integration tests will be added to test the end-to-end alignment process, from the CLI to the final JSON output.
- **Language-Specific Tests:** Tests will be included to validate the alignment for both English (with `nltk`) and Vietnamese (with `underthesea`).

## 5. Documentation

The project's documentation will be updated to include:

- A guide on how to use the new alignment feature.
- Details on the new configuration options.
- Examples of both CLI and Python API usage.
