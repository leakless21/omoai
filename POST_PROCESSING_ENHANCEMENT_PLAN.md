# Post-Processing Enhancement Plan for Word-Level Timestamps

## 1. Introduction

The ASR pipeline now provides word-level timestamps, opening up opportunities for more precise and feature-rich post-processing. This document outlines a plan to enhance the `scripts/post.py` script to leverage this new data.

## 2. Proposed Enhancements

### 2.1. Enhanced Subtitle Generation (Karaoke-Style)

**Concept:** Generate SRT/VTT subtitles where each word appears in sync with the audio, creating a "karaoke-style" reading experience. This is a significant improvement over the current segment-based subtitles.

**Code Changes in `scripts/post.py`:**

1.  **Create new functions `generate_srt_word_level` and `generate_vtt_word_level`:** These will iterate through the `word_segments` array (or `segments[i].words`) in the ASR output.
2.  **Logic:**
    - For each word in `word_segments`, create a subtitle entry.
    - The start and end times will be taken directly from the `start` and `end` fields of each word object.
    - The text for each entry will be the `word` itself.
3.  **Integration:** In the `main` function, add new command-line arguments (e.g., `--word_level_subtitles`) to trigger these new functions. The existing subtitle generation logic will be preserved as the default.

### 2.2. New Output Format: Detailed Word-by-Word Transcript

**Concept:** Produce a new output file format (e.g., a JSON or CSV file) that provides a detailed, word-by-word transcript with timestamps and confidence scores. This format would be highly valuable for data analysis, phonetic research, or integration with other systems.

**Code Changes in `scripts/post.py`:**

1.  **Create a new function `generate_word_transcript`:** This function will be responsible for creating the new output file.
2.  **Logic:**
    - The function will iterate through the `word_segments` array.
    - It will write each word's data (word, start, end, score) to the file in the chosen format (e.g., as a JSON object per line or a CSV row).
3.  **Integration:** Add a new command-line argument (e.g., `--output_word_transcript`) to enable this feature.

### 2.3. Improved Punctuation and Summarization Accuracy

**Concept:** While the direct impact of word timings on punctuation and summarization models is less obvious, we can explore if this data can be used to improve their accuracy. For example, longer pauses between words might indicate sentence boundaries, which could be a useful hint for the punctuation model.

**Code Changes in `scripts/post.py`:**

1.  **Punctuation:**
    - **Modify `punctuate_text_with_splitting`:** Before sending text to the LLM, analyze the word timings to detect long pauses (e.g., > 0.5 seconds) between words.
    - **Insert "markers":** Insert special tokens or simple textual cues (e.g., "[PAUSE]") into the text at these points. This could provide a subtle hint to the LLM about sentence or clause breaks.
    - **Prompt Engineering:** The system prompt for the punctuation model could be updated to explain the meaning of these markers.
2.  **Summarization:**
    - **Modify `summarize_long_text_map_reduce`:** Similar to the punctuation enhancement, use pause detection to guide the chunking of text for summarization. Breaking chunks at natural pauses might result in more coherent summaries.

## 3. Implementation Plan

The enhancements will be implemented in the following order:

1.  **Enhanced Subtitle Generation:** This is the most direct and high-impact feature.
2.  **New Output Format:** This is a relatively straightforward feature to implement and provides immediate value.
3.  **Improved Punctuation/Summarization:** This is more experimental and will require some testing and tuning to determine its effectiveness.

By following this plan, we can significantly improve the capabilities of the post-processing script and provide users with more valuable and accurate outputs.
