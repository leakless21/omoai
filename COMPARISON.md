# OmoAI Phoneme Alignment vs. WhisperX: A Comparative Analysis

This document provides a detailed comparison between the OmoAI self-contained phoneme alignment implementation and the original WhisperX library.

## 1. Core Alignment Logic

The core alignment algorithms (`get_trellis`, `backtrack_beam`, `merge_words`) in our implementation are functionally equivalent to the original WhisperX versions.

- **`get_trellis`**: Both implementations use the same dynamic programming approach to build the trellis for Viterbi alignment.
- **`backtrack_beam`**: Our implementation of beam search backtracking is functionally identical to the original.
- **`merge_words`**: The logic for merging character segments into word segments is consistent across both implementations.

The primary difference is that our implementation consolidates these functions into a single file, [`src/omoai/integrations/alignment.py`](src/omoai/integrations/alignment.py), whereas WhisperX separates them into `ref/whisperx/alignment.py`.

## 2. Feature Parity

### Implemented Features:

- **Core Phoneme Alignment**: We have successfully replicated the core functionality of aligning transcriptions to audio at the phoneme level.
- **Multi-language Support**: Our implementation supports all the same languages as WhisperX, using the same default alignment models.

### Missing Features:

- **Speaker Diarization**: We have not implemented speaker diarization (`diarize.py`), which is a key feature of the WhisperX ecosystem.
- **Advanced Subtitle Generation**: The `SubtitlesProcessor.py` in WhisperX provides advanced subtitle generation capabilities, which are not present in our implementation.
- **VAD Integration**: While we have a `vad.py` file, our integration of Voice Activity Detection (VAD) is not as sophisticated as in WhisperX.
- **Command-Line Interface**: We do not have a dedicated CLI for our alignment tool, unlike WhisperX's `__main__.py`.

## 3. Dependencies and Models

- **Python Dependencies**: Our implementation has fewer dependencies than the full WhisperX library, as we have not implemented features like diarization. Our implementation primarily relies on `torch`, `torchaudio`, and `transformers`, while WhisperX also uses `pyannote.audio`, `nltk`, and others.
- **Alignment Models**: Both implementations use the same default wav2vec2 models for phoneme alignment, sourced from both `torchaudio` and Hugging Face.

## 4. API and Entry Points

- **OmoAI**: Our solution is integrated into the `omoai` module and is executed via `scripts/asr.py`. The entry point for alignment is the `align_segments` function in `src/omoai/integrations/alignment.py`.
- **WhisperX**: The reference implementation has a more traditional library structure, with a public API defined in `__init__.py` and a command-line interface in `__main__.py`. This makes it more versatile for standalone use.

## 5. Architectural Differences

The primary architectural difference is that our implementation is designed as a self-contained, integrated module within the `omoai` ecosystem, whereas WhisperX is a standalone library.

- **Modularity**: WhisperX is more modular, with different functionalities (ASR, alignment, diarization) separated into different files. Our implementation is more monolithic, with all alignment-related code in a single file.
- **Integration**: Our implementation is tightly integrated with our existing `asr.py` script and `omoai` pipeline, while WhisperX is designed to be more generic and adaptable to different workflows.
- **Configuration**: WhisperX uses a more complex configuration system with command-line arguments, while our implementation is configured more directly within our scripts.

## 6. Conclusion

Our self-contained implementation of phoneme alignment is a successful replication of the core functionality of the WhisperX library. However, it is important to note that we have not implemented several key features from the WhisperX ecosystem, such as speaker diarization and advanced subtitle generation. Our implementation is well-suited for our current needs within the `omoai` module, but it is not a full replacement for the WhisperX library.
