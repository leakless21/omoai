# Component: Post-processing Script (post.py)

## Overview

The post-processing script, [`scripts/post.py`](scripts/post.py), is a critical component in the audio processing pipeline. It takes the raw output from an Automatic Speech Recognition (ASR) model, which typically consists of a transcript without punctuation and capitalization, and enriches it using a Large Language Model (LLM). The primary functions of this script are to:

1. **Add Punctuation and Capitalization**: Transform the raw, unpunctuated transcript into a grammatically correct and human-readable format.
2. **Generate a Summary**: Create a concise summary of the transcribed content, including bullet points and an abstract.

This script is designed to work offline and leverages the `vllm` library for efficient LLM inference.

## Core Functions and Logic

The script is structured into several logical sections, each handling a specific part of the post-processing workflow.

### 1. Main Execution Flow (`main`)

The [`main()`](scripts/post.py:410) function orchestrates the entire post-processing pipeline:

- **Argument Parsing**: It uses `argparse` to accept command-line arguments for configuration, such as the path to the ASR JSON output, LLM model settings, and output paths.
- **Configuration Loading**: It loads a `config.yaml` file to get default values and overrides for various parameters, including model IDs, quantization settings, and prompts.
- **Data Loading**: It loads the ASR results from a JSON file, which contains the raw transcript and segmented audio data.
- **Punctuation Stage**:
  - It initializes an LLM instance specifically for punctuation, respecting any overrides from the command line or configuration file.
  - It decides on a punctuation strategy: `single` (punctuate the entire transcript at once), `segments` (punctuate segments individually), or `auto` (automatically choose the best approach based on token count and context window size).
  - It calls the appropriate punctuation function(s) to add punctuation to the transcript.
  - It handles the distribution of the punctuated text back to individual segments if necessary.
- **Summarization Stage**:
  - It initializes an LLM for summarization, which may be the same instance as the punctuation LLM if settings are identical (to save resources).
  - It decides between a single-pass summarization or a map-reduce approach for long transcripts that exceed the model's context window.
  - It calls the summarization function to generate bullet points and an abstract.
- **Output Generation**:
  - It assembles the final output, combining the original ASR data with the newly created punctuated transcript and summary.
  - It saves the final result to a JSON file.
  - Optionally, it can generate separate text files for the transcript and summary.

### 2. Punctuation Logic

The script employs sophisticated logic for adding punctuation:

- **[`punctuate_text()`](scripts/post.py:48)**: This core function sends a raw text segment to the LLM along with a system prompt that instructs it to add punctuation and capitalization without modifying the words themselves.
- **[`process_segments_with_punctuation()`](scripts/post.py:85)**: This function iterates through each segment from the ASR output, applying `punctuate_text()` to get a punctuated version for each segment.
- **[`join_punctuated_segments()`](scripts/post.py:163)**: After punctuation, this function intelligently joins the segments into a coherent transcript. It handles:
  - **Sentence boundaries**: It emits complete sentences as they are formed.
  - **Paragraph breaks**: It inserts paragraph breaks if there is a significant time gap between segments.
  - **Overlap deduplication**: It removes duplicated words at the joins between segments using [`_dedup_overlap()`](scripts/post.py:102).
- **[`_distribute_punct_to_segments()`](scripts/post.py:259)**: If the "single" punctuation mode is used, this function distributes the fully punctuated text back to the original segments based on word count, ensuring that each segment in the final output has its corresponding punctuated piece.
- **[`_build_segment_batches_by_token_budget()`](scripts/post.py:286)**: To manage long transcripts and stay within LLM context limits, this function groups adjacent segments into batches that fit within a specified token budget. This is used in the "segments" mode with batching enabled.

### 3. Summarization Logic

The summarization process is also designed to handle long-form content:

- **[`summarize_text()`](scripts/post.py:66)**: This function sends the text to the LLM with a system prompt requesting a summary in Vietnamese, structured as JSON with "bullets" and "abstract" keys.
- **[`summarize_long_text_map_reduce()`](scripts/post.py:389)**: For texts that are too long for a single LLM call, this function implements a map-reduce strategy:
  - **Map**: It splits the long text into smaller chunks using [`split_text_into_chunks()`](scripts/post.py:337), summarizes each chunk individually, and collects the partial summaries.
  - **Reduce**: It then feeds these partial summaries to the LLM again to generate a final, consolidated summary.
- **Smart Chunking**: The [`split_text_into_chunks()`](scripts/post.py:337) function splits text at sentence boundaries to preserve context, with an option for sentence overlap between chunks.

### 4. Utility and Helper Functions

The script includes several utility functions to support its main operations:

- **File I/O**: [`load_asr_json()`](scripts/post.py:19) and [`save_json()`](scripts/post.py:24) handle loading and saving data in JSON format.
- **LLM Interaction**: [`get_tokenizer()`](scripts/post.py:30), [`apply_chat_template()`](scripts/post.py:34), and [`generate_chat()`](scripts/post.py:41) manage communication with the `vllm` instance.
- **Tokenization**: [`_count_tokens()`](scripts/post.py:247) is used to estimate the number of tokens in a text for context window management.
- **Time Parsing**: [`_parse_time_to_seconds()`](scripts/post.py:113) converts various timestamp formats into seconds, which is used for determining paragraph gaps.

## Configuration (`config.yaml`)

The script heavily relies on a `config.yaml` file for its configuration. The command-line arguments can override these settings. Here's a breakdown of the configuration structure:

### Global LLM Settings

These settings provide defaults for both punctuation and summarization LLMs.

```yaml
llm:
  model_id: "Qwen/Qwen2.5-7B-Instruct-AWQ" # Default Hugging Face model ID
  quantization: null # e.g., "awq", "gptq", or null
  max_model_len: 2048 # Maximum context length for the LLM
  gpu_memory_utilization: 0.90 # Target GPU memory utilization (0.0 to 1.0)
  max_num_seqs: 1 # Maximum number of concurrent sequences
  max_num_batched_tokens: 512 # Maximum number of tokens to batch together
```

### Punctuation-Specific Settings

These settings allow for fine-tuning the punctuation process.

```yaml
punctuation:
  mode: "auto" # Strategy: "auto", "single", or "segments"
  auto_switch_ratio: 0.98 # Ratio of context to use for auto-switch decision
  auto_margin_tokens: 128 # Tokens to reserve as margin in auto-switch
  batching: true # Whether to batch segments in "segments" mode
  join_separator: " " # Separator to use when joining segments
  paragraph_gap_seconds: 3.0 # Time gap in seconds to trigger a paragraph break
  llm: # Overrides for the punctuation LLM
    model_id: "Qwen/Qwen2.5-7B-Instruct-AWQ"
    quantization: null
    max_model_len: 2048
    gpu_memory_utilization: 0.90
    max_num_seqs: 1
    max_num_batched_tokens: 512
  sampling:
    temperature: 0.0 # Sampling temperature for punctuation (0.0 is deterministic)
  system_prompt: "You are a Vietnamese punctuation..." # Custom system prompt for the LLM
```

### Summarization-Specific Settings

These settings control the behavior of the summarization LLM.

```yaml
summarization:
  map_reduce: false # Whether to force map-reduce for summarization
  auto_switch_ratio: 0.98 # Ratio of context to use for auto-switch to map-reduce
  auto_margin_tokens: 64 # Margin for summarization auto-switch
  llm: # Overrides for the summarization LLM
    model_id: "Qwen/Qwen2.5-7B-Instruct-AWQ"
    quantization: null
    max_model_len: 2048
    gpu_memory_utilization: 0.90
    max_num_seqs: 1
    max_num_batched_tokens: 512
  sampling:
    temperature: 0.2 # Sampling temperature for summarization
  system_prompt: "You are a careful summarization..." # Custom system prompt for the LLM
```

### Output and Path Settings

These settings control where the script saves its output.

```yaml
paths:
  out_dir: "data/output" # Default base directory for output files

output:
  write_separate_files: false # Whether to write transcript.txt and summary.txt
  transcript_file: "transcript.txt" # Filename for the plain text transcript
  summary_file: "summary.txt" # Filename for the plain text summary
  wrap_width: 100 # Column width for text wrapping in transcript file (0 for no wrap)
```

## Inputs and Outputs

### Inputs

The script expects the following inputs:

1. **ASR JSON File (`--asr-json`)**:

    - **Type**: JSON file path.
    - **Description**: This is the primary input, containing the raw output from the ASR model. It's a dictionary with at least the following keys:
      - `segments`: A list of segment objects, where each object contains `text_raw` (the unpunctuated text for that segment), `start` (start time), and `end` (end time).
      - `transcript_raw`: The complete unpunctuated transcript as a single string.
      - `audio`: (Optional) A dictionary containing metadata about the original audio file, such as its path.
      - `metadata`: (Optional) Any other metadata from the ASR process.

2. **Configuration File (`--config`)**:

    - **Type**: YAML file path.
    - **Description**: The `config.yaml` file that provides default settings for the script.

3. **Command-Line Arguments**:
    - Various arguments can be provided to override the configuration, specify the LLM, control punctuation modes, and set output paths.

### Outputs

The script produces the following outputs:

1. **Final JSON File (`--out`)**:

    - **Type**: JSON file path.
    - **Description**: The main output file. It's a comprehensive dictionary containing:
      - All fields from the original ASR JSON.
      - `segments`: The list of segments, now augmented with a `text_punct` key containing the punctuated text for each segment.
      - `transcript_punct`: The complete, fully punctuated, and paragraphed transcript as a single string.
      - `summary`: A dictionary containing the generated summary, with keys `bullets` (a list of strings) and `abstract` (a string).
      - `metadata`: Updated metadata, including information about the LLM models used for punctuation and summarization.

2. **Optional Separate Text Files**:
    - If `output.write_separate_files` is set to `true` in the configuration, the script will also create:
      - A plain text transcript file (e.g., `transcript.txt`).
      - A plain text summary file (e.g., `summary.txt`).

## Related Classes and Functions

| Function/Class                           | Location                                     | Purpose                                                          |
| :--------------------------------------- | :------------------------------------------- | :--------------------------------------------------------------- |
| `main`                                   | [`scripts/post.py:410`](scripts/post.py:410) | Main entry point and orchestration function.                     |
| `punctuate_text`                         | [`scripts/post.py:48`](scripts/post.py:48)   | Core function for adding punctuation to a text segment.          |
| `summarize_text`                         | [`scripts/post.py:66`](scripts/post.py:66)   | Core function for generating a summary of a text segment.        |
| `join_punctuated_segments`               | [`scripts/post.py:163`](scripts/post.py:163) | Joins punctuated segments into a coherent transcript.            |
| `summarize_long_text_map_reduce`         | [`scripts/post.py:389`](scripts/post.py:389) | Handles summarization of long texts using a map-reduce strategy. |
| `_build_segment_batches_by_token_budget` | [`scripts/post.py:286`](scripts/post.py:286) | Groups segments into batches based on token count.               |
| `load_asr_json`                          | [`scripts/post.py:19`](scripts/post.py:19)   | Loads the input ASR JSON file.                                   |
| `save_json`                              | [`scripts/post.py:24`](scripts/post.py:24)   | Saves the final output to a JSON file.                           |
| `build_llm`                              | [`scripts/post.py:486`](scripts/post.py:486) | Initializes and returns a `vllm.LLM` instance.                   |
