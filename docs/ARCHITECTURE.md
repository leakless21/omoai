# Architecture

This document outlines the overall architecture of the OmoAI project, detailing its components, their responsibilities, and interactions.

## Overview

The OmoAI project is a sophisticated audio processing pipeline designed to transcribe, punctuate, and summarize audio content. It leverages a combination of Automatic Speech Recognition (ASR) and Large Language Models (LLMs) to deliver high-quality text processing.

The system is composed of several key components:

- **ASR Engine**: Converts audio into raw text transcripts.
- **Punctuation Module**: Enhances raw transcripts by adding correct punctuation and capitalization.
- **Summarization Module**: Generates concise summaries of the processed text.
- **Output Formatters**: Produce the final output in various formats, such as SRT, VTT, and plain text.

## Component Details

### ASR Engine

The ASR engine is responsible for converting audio data into a textual representation. It processes audio segments and generates a JSON output containing the transcribed text along with timing information.

**Responsibilities:**

- Audio input handling and validation.
- Feature extraction from audio signals.
- Acoustic modeling to phonetically transcribe speech.
- Language modeling to convert phonemes into words.

**Interface:**

- Input: Audio files (various formats).
- Output: JSON object with `segments` array, each containing `text`, `start`, and `end` fields.

**Dependencies:**

- External ASR libraries (e.g., Whisper, ChunkFormer).

### Punctuation Module

The punctuation module takes the raw text from the ASR engine and refines it by adding appropriate punctuation and capitalization. This is crucial for improving the readability and usability of the transcribed text.

**Responsibilities:**

- Analyzing raw text for sentence boundaries.
- Applying grammatical rules for punctuation.
- Utilizing LLMs for context-aware punctuation restoration.

**Interface:**

- Input: Raw text string from ASR output.
- Output: Punctuated text string.

**Dependencies:**

- LLMs (e.g., vLLM, Hugging Face Transformers).
- Tokenizers for text processing.

#### Punctuation Restoration Logic

The core of the punctuation module is the `_force_preserve_with_alignment` function in [`scripts/post.py`](scripts/post.py:532). This function intelligently merges the original ASR output with the LLM's punctuated version, ensuring that the original word order and content are preserved while incorporating the LLM's punctuation and capitalization.

**Bug Fix and Correction**

A recent bug fix in the [`_force_preserve_with_alignment`](scripts/post.py:532) function has significantly improved the punctuation restoration process.

**Previous Issue:**
The previous implementation had a critical flaw where it would inadvertently drop punctuation during `delete` and `insert` operations performed by the `SequenceMatcher`. When the `SequenceMatcher` identified words to be deleted from the original text or inserted from the LLM's output, the associated punctuation marks were also discarded. This led to a degradation in the quality and accuracy of the final punctuated text, as essential punctuation was lost in the alignment process.

**Corrected Logic:**
The corrected logic now ensures that all punctuation from the LLM's output is preserved throughout the alignment process. The key improvements are:

1.  **Punctuation Mapping**: Punctuation is now explicitly mapped to the word it follows. This is achieved using a `punct_after` dictionary, where each key corresponds to a word index, and the value is a list of punctuation marks that should appear after that word.
2.  **Preservation during Operations**:
    - During `delete` operations, only the original words are skipped, and their associated punctuation is preserved.
    - During `insert` operations, the LLM's words are skipped, but their punctuation is retained and correctly placed in the final output.
3.  **Robust Alignment**: The `SequenceMatcher`'s operations are now augmented with explicit punctuation handling, ensuring that the alignment process is both word-aware and punctuation-aware.

This fix guarantees that the punctuation restoration process is more accurate and reliable, leading to higher quality transcripts that are ready for consumption.
This fix has been validated with a new test suite in [`tests/test_punctuation_alignment.py`](tests/test_punctuation_alignment.py), ensuring the correctness of the improved logic.

### Summarization Module

The summarization module processes the punctuated text to generate concise summaries. It employs LLMs to understand the context and extract key points.

**Responsibilities:**

- Analyzing the input text for key themes and topics.
- Generating a concise summary while retaining the original meaning.
- Structuring the output for readability.

**Interface:**

- Input: Punctuated text string.
- Output: JSON object with `bullets` (key points) and `abstract` (a short summary).

**Dependencies:**

- LLMs for summarization.
- Tokenizers for input processing.

### Output Formatters

The output formatters are responsible for converting the processed text into various user-friendly formats. They handle the final presentation of the transcribed and summarized content.

**Responsibilities:**

- Formatting text into specified output types (SRT, VTT, JSON, etc.).
- Handling metadata and timing information for subtitles.
- Ensuring output files are correctly structured and encoded.

**Interface:**

- Input: Processed text and metadata.
- Output: Formatted text files (e.g., `.srt`, `.vtt`).

**Dependencies:**

- Format-specific libraries for output generation.

## Data Flow

1.  **Audio Input**: The process begins with an audio file being fed into the ASR engine.
2.  **Transcription**: The ASR engine converts the audio into a raw text transcript, which is outputted as a JSON object.
3.  **Punctuation**: The raw text is then processed by the punctuation module, which uses an LLM to add punctuation and correct capitalization.
4.  **Summarization**: The punctuated text is passed to the summarization module to generate a concise summary.
5.  **Output**: The final processed text and summary are formatted into the desired output files by the output formatters.

## Deployment Considerations

The system is designed to be scalable and can be deployed on various infrastructure setups, from single machines to distributed clusters.

**Compute:**

- GPU acceleration is recommended for both ASR and LLM inference to ensure low latency and high throughput.
- CPU-only deployment is possible but will result in significantly slower processing times.

**Storage:**

- Sufficient storage is required for audio input files, intermediate JSON outputs, and final formatted files.
- Model weights and caches should be stored on fast storage (SSD) for optimal performance.

**Dependencies:**

- Python 3.8+
- PyTorch or TensorFlow
- vLLM or other LLM serving framework
- ASR-specific libraries

## Future Enhancements

- **Real-time Processing**: The architecture can be extended to support real-time audio stream processing.
- **Multi-language Support**: Expansion to support a wider range of languages for transcription and summarization.
- **Custom Model Integration**: A flexible plugin system for integrating custom ASR and LLM models.

## Testing

### Punctuation Alignment Tests

A dedicated test suite has been created to ensure the correctness and robustness of the punctuation alignment logic. The file [`tests/test_punctuation_alignment.py`](tests/test_punctuation_alignment.py) contains a comprehensive suite of tests for the `_force_preserve_with_alignment` function.

**Purpose:**

This test suite is designed to validate the behavior of the punctuation alignment logic under various conditions, ensuring that the bug fix is effective and that the function handles different scenarios correctly.

**Test Coverage:**

The tests cover a wide range of scenarios, including:

- **Simple Punctuation**: Verifies that basic punctuation (e.g., periods, commas) is correctly preserved and aligned.
- **Insertions**: Ensures that when words are inserted from the LLM's output, their associated punctuation is also correctly inserted.
- **Deletions**: Confirms that when words are deleted from the original text, the punctuation from the LLM's output is still applied.
- **Replacements**: Tests that when words are replaced, the punctuation is correctly mapped to the new words.

This comprehensive testing approach helps to prevent regressions and ensures the long-term stability of the punctuation restoration process.
