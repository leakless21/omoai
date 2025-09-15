# Phoneme Alignment Integration Plan (Revised)

## 1. Overview

This document outlines a revised plan to integrate phoneme-level timestamp alignment into the omoai ASR pipeline. The goal is to create a robust and maintainable implementation that closely mirrors the structure and features of the `whisperx` reference implementation, while adapting it to our existing repository architecture and supporting multiple tokenizers.

## 2. Core Components

The integration will be divided into several key components:

- **CLI Integration:** Exposing the alignment functionality through our existing command-line scripts.
- **Python API:** Providing a modular, programmatic interface for alignment.
- **Configuration:** Managing alignment settings through `config.yaml` and Pydantic schemas, including a configurable tokenizer.
- **Dependency Management:** Incorporating new dependencies required for tokenization.
- **Code Structure:** Organizing the new alignment code in a logical and maintainable way.
- **Output Structure:** Defining a clear and consistent JSON output format for the aligned transcription.

---

## 3. CLI Integration

The primary entry point for the alignment functionality will be through the `scripts/asr.py` script.

### 3.1. New Command-Line Arguments

The following arguments will be added to `scripts/asr.py`:

- `--align`: A boolean flag to enable the alignment process.
- `--align_model TEXT`: The name of the alignment model to use (e.g., `WAV2VEC2_ASR_BASE_960H`).
- `--language TEXT`: The language code for the alignment model.
- `--tokenizer TEXT`: The tokenizer to use (e.g., `nltk`, `underthesea`).

### 3.2. Example Usage

```bash
python scripts/asr.py --input audio.wav --output output.json --align --language vi --tokenizer underthesea
```

---

## 4. Python API Design

The alignment functionality will be exposed through a new module, `src/omoai/integrations/alignment.py`.

### 4.1. Core Functions

- **`load_align_model(language_code: str, device: str) -> Tuple[torch.nn.Module, dict]`:**

  - Loads a wav2vec2 model for the specified language.
  - Returns the model and its metadata (dictionary, language).

- **`align(segments: List[dict], model: torch.nn.Module, metadata: dict, audio: np.ndarray, device: str, tokenizer: str) -> dict:`:**
  - Takes the output of the ASR transcription and aligns it with the audio.
  - Uses the specified tokenizer for sentence splitting.
  - Returns a dictionary with the aligned segments, including word-level timestamps.

### 4.2. Example Usage

```python
import omoai
from omoai.integrations.alignment import load_align_model, align

# 1. Load configuration and initialize ASR model
config = omoai.load_config("config.yaml")
asr_model = omoai.load_asr_model(config)

# 2. Transcribe audio
audio = omoai.load_audio("audio.mp3")
transcription_result = asr_model.transcribe(audio)

# 3. Load alignment model
align_model, metadata = load_align_model(language_code="vi", device="cuda")

# 4. Align transcription
aligned_result = align(
    transcription_result["segments"],
    align_model,
    metadata,
    audio,
    "cuda",
    tokenizer="underthesea"
)

print(aligned_result)
```

---

## 5. Configuration

Alignment settings will be managed through `config.yaml` and validated using Pydantic schemas.

### 5.1. `config.yaml` Structure

A new `alignment` section will be added to `config.yaml`:

```yaml
asr:
  # ... existing ASR config ...

alignment:
  model_name: "nguyenvulebinh/wav2vec2-base-vi"
  language: "vi"
  tokenizer: "underthesea" # or "nltk"
```

### 5.2. Pydantic Schema (`src/omoai/config/schemas.py`)

A new `AlignmentConfig` schema will be added:

```python
from pydantic import BaseModel, Field
from typing import Literal

class AlignmentConfig(BaseModel):
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

---

## 6. Dependency Management

The following dependency will be added to `pyproject.toml`:

- `underthesea`: For Vietnamese sentence tokenization.

---

## 7. Code Structure

The new alignment code will be placed in `src/omoai/integrations/alignment.py`.

- **`src/omoai/integrations/alignment.py`**:
  - `load_align_model()`: Function to load the alignment model.
  - `align()`: The main alignment function, which will dynamically select the tokenizer based on the configuration.
  - Helper functions for the alignment process (e.g., `get_trellis`, `backtrack`).

---

## 8. Output Structure

The final JSON output will be a dictionary containing the aligned segments and word-level timestamps.

### 8.1. Example JSON Output

```json
{
  "segments": [
    {
      "start": 0.5,
      "end": 3.2,
      "text": "This is a test sentence.",
      "words": [
        {
          "word": "This",
          "start": 0.5,
          "end": 0.8,
          "score": 0.95
        },
        {
          "word": "is",
          "start": 0.8,
          "end": 1.0,
          "score": 0.98
        },
        {
          "word": "a",
          "start": 1.0,
          "end": 1.1,
          "score": 0.99
        },
        {
          "word": "test",
          "start": 1.1,
          "end": 1.5,
          "score": 0.97
        },
        {
          "word": "sentence.",
          "start": 1.5,
          "end": 2.5,
          "score": 0.96
        }
      ]
    }
  ],
  "word_segments": [
    {
      "word": "This",
      "start": 0.5,
      "end": 0.8,
      "score": 0.95
    },
    {
      "word": "is",
      "start": 0.8,
      "end": 1.0,
      "score": 0.98
    },
    {
      "word": "a",
      "start": 1.0,
      "end": 1.1,
      "score": 0.99
    },
    {
      "word": "test",
      "start": 1.1,
      "end": 1.5,
      "score": 0.97
    },
    {
      "word": "sentence.",
      "start": 1.5,
      "end": 2.5,
      "score": 0.96
    }
  ]
}
```
