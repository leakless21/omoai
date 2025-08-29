# Punctuation Architecture

This document outlines the architecture and workflow of the punctuation subsystem, which is responsible for applying punctuation to the raw output of the Automatic Speech Recognition (ASR) engine.

## 1. Punctuation Workflow

The punctuation process is a post-processing step that enhances the readability of the ASR transcript. It involves several key components that work together to receive the raw text, send it to a punctuation model, and integrate the results back into the final transcript while preserving timestamps.

The end-to-end workflow is as follows:

1.  **API Request**: The process begins when a client sends a request to the `/postprocess/punctuate` endpoint. The request body contains the ASR output, which is a list of segments, each with a text transcript and timestamps.

    - **Example Request Body**:
      ```json
      {
        "segments": [
          { "start": 0.0, "end": 2.5, "text": "hello world" },
          { "start": 2.5, "end": 5.0, "text": "how are you today" }
        ]
      }
      ```

2.  **Controller Layer (`postprocess_controller.py`)**: The [`PunctuateController.punctuate_v2`](src/omoai/api/postprocess_controller.py:17) receives the request. It extracts the ASR segments and passes them to the `punctuate_text` service function in `services_v2.py`.

3.  **Service Layer (`services_v2.py`)**: The [`punctuate_text`](src/omoai/api/services_v2.py:12) function orchestrates the punctuation process. It joins the text from all segments into a single string to be sent to the punctuation model.

    - **Example**: From the segments above, the joined text would be `"hello world how are you today"`.
    - It then retrieves the `VLLMProcessor` instance (often a singleton) which acts as a client to the vLLM inference server. This processor is a wrapper around the `vllm.LLM` class, providing a simplified interface for generating text.
    - The `VLLMProcessor` is configured with model parameters (like model name, temperature, max tokens, quantization) which are typically loaded from the application's configuration file (`config.yaml`). These parameters are used to initialize the `vllm.LLM` object and `vllm.SamplingParams`.
    - The service layer calls a method on the `VLLMProcessor` (e.g., `process` or `generate`) passing the consolidated text. This method internally handles the interaction with the vLLM library.

4.  **VLLM Invocation via `VLLMProcessor` and `scripts/post.py`**:

    - The `VLLMProcessor`'s method (e.g., `generate`) is responsible for the actual communication with the vLLM library. It abstracts away the direct API calls to vLLM.
    - **Internal VLLM Call**: When the `VLLMProcessor`'s method is called, it internally performs the following, often leveraging helper functions from `scripts/post.py`:

      1.  **Prompt Formatting (`apply_chat_template`)**: It constructs the final prompt to be sent to the LLM. This often involves a system prompt (instructing the model to act as a punctuation restorer) and the user prompt (the raw text to be punctuated). The function [`apply_chat_template`](scripts/post.py:91) from `scripts/post.py` is used here if the LLM requires a specific chat format (e.g., for instruction-tuned models).

          - **Logic of `apply_chat_template`**:
            - It attempts to use the tokenizer's `apply_chat_template` method if available. This is the standard way to format prompts for models that expect a specific chat structure (e.g., `[INST] ... [/INST]` for Llama, or `<|im_start|>user ... <|im_end|>` for ChatML).
            - If `apply_chat_template` is not available or fails, it falls back to a simpler format:
              - It iterates through the list of message dictionaries.
              - For each message, it checks the `role` (`system`, `user`, `assistant`).
              - It prepends a tag like `[SYSTEM]`, `[USER]`, or `[ASSISTANT]` to the message content.
              - It joins these formatted messages with newlines.
              - Finally, it appends `[ASSISTANT]\n` to signal where the model should start generating.
          - **Example System Prompt**: `"You are a helpful assistant that restores punctuation to text. Output only the punctuated text."`
          - **Example User Prompt**: `"hello world how are you today"`
          - **Example Formatted Prompt (fallback)**:

            ```
            [SYSTEM]
            You are a helpful assistant that restores punctuation to text. Output only the punctuated text.

            [USER]
            hello world how are you today

            [ASSISTANT]
            ```

      2.  **Parameter Setting (`SamplingParams`)**: It sets up `vllm.SamplingParams` for the vLLM generation. This object controls the generation behavior.
          - **Key Parameters**:
            - `temperature`: Controls randomness. A value of `0.0` makes the output deterministic.
            - `max_tokens`: The maximum number of tokens the model is allowed to generate for the output.
            - `stop`: Optional list of stop token IDs or strings that, when generated, will halt the generation process.
          - These parameters are typically derived from the application configuration or passed directly.
      3.  **Text Generation (`llm.generate`)**: The core of the invocation is calling the `generate` method of the `vllm.LLM` instance.
          - The `generate` method takes a list of prompts (even if it's just one) and the `SamplingParams`.
          - **Under the hood**: vLLM handles batching (even for a single prompt), tokenization, and efficient inference on the specified hardware (GPU).
          - It uses a PagedAttention mechanism for optimized memory usage and throughput.
          - The `vllm.LLM` object is initialized with parameters like the model name/path, tensor parallel size (if using multiple GPUs), GPU memory utilization, quantization settings, etc.
      4.  **Response Handling**: The `llm.generate` call returns a list of `RequestOutput` objects (one for each prompt).
          - Each `RequestOutput` contains a list of `SequenceOutput` objects (usually one per request unless `n > 1` in sampling params).
          - The generated text is extracted from `outputs[0].outputs[0].text` (assuming a single sequence output).
          - This raw text from the LLM is the punctuated version of the input.

    - **Example**: The punctuation model might return `"Hello world, how are you today?"`.

5.  **Alignment and Merging (`post.py`)**: The core logic for integrating the punctuated text back into the original segments resides in the [`join_punctuated_segments`](scripts/post.py:372) function in `scripts/post.py`.

    - This function takes the original ASR segments (with timestamps) and the punctuated text string received from the `VLLMProcessor`.
    - It uses the `_force_preserve_with_alignment` function to intelligently merge the punctuation from the model's output back into the original text segments. This alignment process ensures that the original word timings are preserved. It handles mismatches between the original and punctuated text by finding the best alignment.
    - **Example**: The function will map `"Hello world, how are you today?"` back to the original segments, resulting in:
      ```json
      [
        { "start": 0.0, "end": 2.5, "text": "Hello world," },
        { "start": 2.5, "end": 5.0, "text": "how are you today?" }
      ]
      ```
    - The result is a new set of segments where the text is punctuated, but the original start and end times are maintained.

6.  **API Response**: The controller receives the punctuated and aligned segments and returns them to the client in the API response.

## 2. Key Functions in `scripts/post.py`

The `scripts/post.py` module contains the essential functions for handling the punctuation and alignment logic.

### `punctuate_text(segments: list[dict], punctuator) -> list[dict]`

- **Purpose**: This function was the original entry point for the punctuation process. It takes a list of ASR segments and a `punctuator` object, joins the text, punctuates it, and then uses `join_punctuated_segments` to merge the results. _Note: In the current implementation, the primary orchestration is handled by `services_v2.punctuate_text`, which then calls `join_punctuated_segments` directly._
- **Parameters**:
  - `segments` (list[dict]): A list of segment dictionaries from the ASR output. Each dictionary contains `start`, `end`, and `text` keys.
    - **Example**: `[{"start": 0.0, "end": 1.0, "text": "this is a test"}]`
  - `punctuator`: An object with a `punctuate` method that takes a string and returns a punctuated string. In the current VLLM-based setup, this `punctuator` would be the `VLLMProcessor` instance.
    - **Example**: `punctuator.punctuate("this is a test")` would internally trigger the VLLM workflow described above and might return `"This is a test."`.
- **Return Value**: A list of punctuated segment dictionaries, with original timestamps preserved.
  - **Example**: `[{"start": 0.0, "end": 1.0, "text": "This is a test."}]`

### `_force_preserve_with_alignment(original_text: str, llm_text: str, adopt_case: bool = True) -> str`

- **Purpose**: This is a core alignment function responsible for merging the punctuated text (`llm_text`) with the original, unpunctuated text (`original_text`) while preserving the word-level structure. It meticulously handles discrepancies (like hallucinations, dropped words, or substitutions) between the ASR output and the punctuation model's output.
- **Detailed Logic**:
  1.  **Tokenization**: Both the original and LLM-generated texts are tokenized into words and punctuation marks using [`_tokenize_words_and_punct`](scripts/post.py:488). This function separates words (e.g., "Hello") from punctuation characters (e.g., ",", "!").
      - **Example**: `original_text = "hello world"` becomes `["hello", "world"]`.
      - **Example**: `llm_text = "Hello, world!"` becomes `["Hello", ",", "world", "!"]`.
  2.  **Punctuation Mapping**: The punctuation from the LLM text is mapped to the word it immediately follows. This is stored in a dictionary `punct_after`, where keys are word indices and values are lists of punctuation marks.
      - **Example**: For `llm_tokens = ["Hello", ",", "world", "!"]`, `punct_after` would be `{0: [","], 1: ["!"]}`. Punctuation at the very beginning is stored under index `-1`.
  3.  **Sequence Matching**: The words from the original text and the LLM text are compared using Python's `difflib.SequenceMatcher`. This identifies sequences of words that are "equal" (matched), "replace" (substituted), "delete" (words in original but not in LLM), and "insert" (words in LLM but not in original).
  4.  **Reconstruction**:
      - **`equal`**: For matched words, the original word is kept. If `adopt_case` is true, the casing from the LLM's word is adopted. The punctuation associated with the LLM word is appended.
        - **Example**: `original_word = "world"`, `llm_word = "world"`. If `adopt_case` is true, it becomes "world". Punctuation `!` is appended, resulting in `"world!"`.
      - **`replace`**: If the LLM replaced an original word(s), the LLM's word(s) are used (respecting `adopt_case`). Its associated punctuation is also appended.
        - **Example**: Original has "foo", LLM has "bar". "bar" (and its punctuation) is used.
      - **`delete`**: If the LLM omitted some original words, those words are simply skipped and not added to the final output.
      - **`insert`**: If the LLM added new words not present in the original, these new words (respecting `adopt_case`) and their punctuation are inserted into the final text.
  5.  **Joining**: The final list of tokens (words and punctuation) is joined into a single string using [`_join_tokens_with_spacing`](scripts/post.py:517), which ensures correct spacing (e.g., no space before a comma).
- **Parameters**:
  - `original_text` (str): The raw, unpunctuated text from the ASR.
    - **Example**: `"hello world how are you"`
  - `llm_text` (str): The punctuated text generated by the language model.
    - **Example**: `"Hello, world! How are you?"`
  - `adopt_case` (bool, optional): If true, the casing of words from the LLM text is adopted. Defaults to `True`.
- **Return Value**: A single string that is the aligned and punctuated version of the original text.
  - **Example**: `"Hello, world! How are you?"`

### `join_punctuated_segments(punctuated_text: str, segments: list[dict]) -> list[dict]`

- **Purpose**: This function serves as the main driver for the punctuation integration process. It takes the fully punctuated text (from the LLM) and the original ASR segments (with timestamps) and produces a final list of punctuated segments where the original timestamps are preserved.
- **Detailed Logic**:
  1.  **Initial Check**: If there's no `punctuated_text` or no `segments`, it returns a copy of the original segments.
  2.  **Concatenation**: It creates a single string (`original_concat`) by joining the `text_raw` from all original segments.
  3.  **Word Splitting**: Both `original_concat` and `punctuated_text` are split into lists of words using [`_split_words`](scripts/post.py:480).
  4.  **Distribution Strategy**:
      - **Exact Match**: If the number of words in `original_concat` and `punctuated_text` is identical, it calls [`_distribute_exact_match`](scripts/post.py:635). This function simply distributes the words from `punctuated_text` back into the original segments based on the original word count per segment.
        - **Example**: If segment 1 had 2 words and segment 2 had 3 words, it takes the first 2 words from the punctuated text for segment 1 and the next 3 for segment 2.
      - **Fuzzy Match**: If word counts differ, it calls [`_distribute_fuzzy_match`](scripts/post.py:652). This is a more complex alignment:
        - It calculates the total character length of `original_concat` and `punctuated_text`.
        - For each original segment, it calculates what proportion of the total `original_concat` characters that segment's text represents.
        - It then applies this same proportion to the `punctuated_text` to estimate the start and end character indices for the corresponding punctuated section.
        - It extracts this `punct_section` from the `punctuated_text`.
        - **Fallback**: If the extracted `punct_section` is too short or seems invalid (e.g., less than half the length of the original segment's text), it falls back to [`_add_basic_punctuation`](scripts/post.py:691), which simply capitalizes the first letter and adds a period at the end of the segment's original text.
  5.  **Result Construction**: It constructs a new list of segments. Each segment is a copy of the original, but with its `text_punct` field populated by the distributed text (either from exact or fuzzy matching).
- **Parameters**:
  - `punctuated_text` (str): The single string of punctuated text from the punctuation model.
    - **Example**: `"This is the first sentence. This is the second."`
  - `segments` (list[dict]): The original list of ASR segments, each with `start`, `end`, and `text`.
    - **Example**: `[{"start": 0.0, "end": 2.0, "text": "this is the first sentence"}, {"start": 2.0, "end": 4.0, "text": "this is the second"}]`
- **Return Value**: A new list of segment dictionaries, where the `text` field is punctuated and the `start` and `end` timestamps are preserved from the original segments. The function intelligently distributes the punctuated text across the original segment boundaries.
  - **Example**:
    ```json
    [
      { "start": 0.0, "end": 2.0, "text": "This is the first sentence." },
      { "start": 2.0, "end": 4.0, "text": "This is the second." }
    ]
    ```
  - **Detailed Example**:
    - **Input**:
      - `punctuated_text`: `"Hello, world. How are you today?"`
      - `segments`: `[{start: 0, end: 1, text: "hello world"}, {start: 1, end: 2, text: "how are you today"}]`
    - **Process**:
      1. `_force_preserve_with_alignment` is called, which returns: `[("Hello", ","), ("world", "."), ("How", ""), ("are", ""), ("you", ""), ("today", "?")]`.
      2. The function then iterates through the original segments. For the first segment (`"hello world"`), it takes the first two aligned words and punctuation: `"Hello,"` and `"world."`. It joins them to form `"Hello, world."`.
      3. For the second segment (`"how are you today"`), it takes the remaining words and punctuation: `"How"`, `"are"`, `"you"`, and `"today?"`. It joins them to form `"How are you today?"`.
      4. The original timestamps are preserved for each new segment.
    - **Output**:
      ```json
      [
        { "start": 0, "end": 1, "text": "Hello, world." },
        { "start": 1, "end": 2, "text": "How are you today?" }
      ]
      ```
