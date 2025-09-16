# Codebase Comparison Report: `omoai` vs. `ref/whisperx`

This report summarizes the key differences between the `omoai` codebase and the `ref/whisperx` reference implementation.

## High-Level Summary

The `omoai` codebase is a re-implementation of the `whisperx` functionality with a different architectural philosophy. It prioritizes robustness, flexibility, and loose coupling over the raw performance of the original `whisperx` library. The `omoai` codebase is a more "production-ready" implementation, with a proper API layer and configuration management. However, it has sacrificed some of the performance and accuracy of the original library to achieve this.

## Key Architectural Differences

| Feature           | `ref/whisperx`                | `src/omoai`                               |
| :---------------- | :---------------------------- | :---------------------------------------- |
| **Architecture**  | In-memory, library-based      | Out-of-process, script-based              |
| **Performance**   | High                          | Lower (due to subprocess overhead)        |
| **Robustness**    | Less robust to memory leaks/crashes | More robust                               |
| **Flexibility**   | Less flexible                 | More flexible                             |
| **Dependencies**  | Tightly coupled               | Loosely coupled                           |
| **API**           | CLI-focused                   | Litestar-based REST API                   |
| **Configuration** | Hardcoded values              | `config.yaml` file                        |

## Component-Level Comparison

### 1. Alignment (`alignment.py`)

*   **`ref/whisperx`:** Uses `nltk` for robust sentence splitting. Relies on other `whisperx` modules.
*   **`src/omoai`:**  Removes the `nltk` dependency, resulting in less accurate sentence splitting. The core logic is duplicated, but with added compatibility layers to fit into the `omoai` application.
*   **Suggestion:** Refactor `src/omoai/integrations/alignment.py` to import the core alignment logic from a `whisperx_legacy` module to reduce code duplication.

### 2. Voice Activity Detection (VAD)

*   **`ref/whisperx`:** Provides a choice between `pyannote.audio` and `silero-vad`. The `pyannote` implementation is sophisticated and accurate.
*   **`src/omoai`:**  The VAD has been completely rewritten to be lightweight and dependency-free, using `webrtcvad` if available. This removes the powerful `pyannote.audio` VAD and its advanced features.
*   **Suggestion:** Add `pyannote.audio` back as an optional dependency to allow users to choose between performance and accuracy.

### 3. Automatic Speech Recognition (ASR)

*   **`ref/whisperx`:** Uses an in-memory, library-based approach with `faster-whisper` for high performance.
*   **`src/omoai`:** Uses an out-of-process, script-based approach, calling external scripts via `subprocess`. This is more robust and flexible but less performant.
*   **Suggestion:** Centralize the configuration for the script paths and improve error handling.

## Overall Suggestions for `omoai`

1.  **Adopt a Hybrid Approach:** For performance-critical components like ASR and VAD, consider offering both the current script-based approach and an optional in-memory, library-based approach. This would allow users to choose the right trade-off between performance and robustness for their needs.
2.  **Reduce Code Duplication:** Refactor the codebase to reduce duplication, especially in the `alignment` module. This will improve maintainability and make it easier to incorporate upstream changes from `whisperx`.
3.  **Improve Configuration Management:** Centralize all configuration options in the `config.yaml` file to make the application easier to configure and manage.
4.  **Enhance Error Handling:** Implement more specific and informative error handling to improve the user experience.
