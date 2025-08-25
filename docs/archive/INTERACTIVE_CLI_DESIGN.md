# Interactive CLI Design for OMOAI

## 1. Overview

This document outlines the design for an interactive command-line interface (CLI) for the OMOAI project. The CLI is designed to be intuitive and user-friendly, guiding users through the process of transcribing and summarizing audio files. It leverages the `questionary` library to create interactive prompts, making the tool accessible to users of all experience levels.

The primary goal of the interactive CLI is to expose the program's core functionality in a structured and easy-to-navigate manner, reducing the learning curve and improving the overall user experience.

## 2. CLI Structure

The interactive CLI will be launched by running the main script with the `--interactive` flag:

```bash
python main.py --interactive
```

Upon launching, the user will be presented with a main menu that serves as the central navigation point. From here, users can choose to run the full pipeline, execute individual stages, or configure settings.

### Main Menu

The main menu will be implemented using a `questionary.select` prompt and will offer the following options:

- **Run Full Pipeline:** Execute the entire transcription and summarization process, from preprocessing to post-processing.
- **Run Individual Stages:**
  - **Preprocess Audio:** Convert an audio file to the required format.
  - **Run ASR:** Transcribe a preprocessed audio file.
  - **Post-process ASR Output:** Add punctuation and summarize a transcription.
- **Configuration:** View or modify the application's settings.
- **Exit:** Terminate the interactive session.

## 3. Commands and User Flows

This section details the user interactions for each of the main menu options.

### 3.1. Run Full Pipeline

This is the primary user flow, designed to be a seamless, step-by-step process.

1.  **Prompt for Audio File:** The user will be asked to provide the path to the input audio file.

    - **Prompt Type:** `questionary.text`
    - **Validation:** The input will be validated to ensure the file exists.

2.  **Prompt for Output Directory:** The user will be asked to specify an output directory. A default will be suggested based on the input file's name and a timestamp.

    - **Prompt Type:** `questionary.text`
    - **Default Value:** `data/output/{audio_file_stem}-{YYYYMMDD-HHMMSS}`

3.  **Prompt for Model Directory (Optional):** The user can optionally specify a path to the ChunkFormer model directory. If not provided, the default from the configuration will be used.

    - **Prompt Type:** `questionary.text`
    - **Default:** Value from `config.yaml` or a standard path.

4.  **Prompt for Config File (Optional):** The user can optionally specify a path to a custom configuration file.

    - **Prompt Type:** `questionary.text`
    - **Default:** `config.yaml` or value from `OMOAI_CONFIG` environment variable.

5.  **Confirmation:** Before starting the pipeline, a summary of the selected options will be displayed, and the user will be asked for confirmation.

    - **Prompt Type:** `questionary.confirm`

6.  **Execution:** The pipeline will be executed, with progress indicators for each stage (Preprocessing, ASR, Post-processing).

7.  **Completion:** Upon completion, a message will be displayed with the path to the final output files.

**Example User Flow:**

```
? Select an action: Run Full Pipeline
? Please enter the path to the audio file: data/input/danba.mp3
? Please enter the output directory [data/output/danba-20250815-103406]:
? Please enter the path to the model directory (optional) [models/chunkformer/chunkformer-large-vie]:
? Please enter the path to the config file (optional) [config.yaml]:
? Do you want to start the pipeline with these settings? Yes
[INFO] Starting pipeline...
[INFO] Preprocessing audio...
[SUCCESS] Preprocessing complete. Output: data/output/danba-20250815-103406/preprocessed.wav
[INFO] Running ASR...
[SUCCESS] ASR complete. Output: data/output/danba-20250815-103406/asr.json
[INFO] Post-processing ASR output...
[SUCCESS] Post-processing complete. Output: data/output/danba-20250815-103406/final.json
[INFO] Pipeline finished. All outputs saved to data/output/danba-20250815-103406.
```

### 3.2. Run Individual Stages

This option provides more granular control over the pipeline, allowing users to execute each stage separately.

#### 3.2.1. Preprocess Audio

1.  **Prompt for Input File:** Ask for the path to the audio file to be preprocessed.

    - **Prompt Type:** `questionary.text`
    - **Validation:** Ensure the file exists.

2.  **Prompt for Output File:** Ask for the path where the preprocessed WAV file will be saved.

    - **Prompt Type:** `questionary.text`
    - **Default:** `{output_directory}/preprocessed.wav`

3.  **Execution:** Run the preprocessing step.
4.  **Completion:** Inform the user of success and the output file path.

**Example User Flow:**

```
? Select an action: Run Individual Stages
? Select a stage: Preprocess Audio
? Please enter the path to the input audio file: data/input/giangoi.mp3
? Please enter the path for the output WAV file [data/output/giangoi-20250815-091231/preprocessed.wav]:
[INFO] Preprocessing audio...
[SUCCESS] Preprocessing complete. Output: data/output/giangoi-20250815-091231/preprocessed.wav
```

#### 3.2.2. Run ASR

1.  **Prompt for Preprocessed Audio File:** Ask for the path to the preprocessed WAV file.

    - **Prompt Type:** `questionary.text`
    - **Validation:** Ensure the file exists and is a WAV file.

2.  **Prompt for Output ASR JSON File:** Ask for the path to save the ASR output.

    - **Prompt Type:** `questionary.text`
    - **Default:** `{output_directory}/asr.json`

3.  **Prompt for Model Directory (Optional):** Same as in the full pipeline.

4.  **Prompt for Config File (Optional):** Same as in the full pipeline.

5.  **Execution:** Run the ASR step.
6.  **Completion:** Inform the user of success and the output file path.

**Example User Flow:**

```
? Select an action: Run Individual Stages
? Select a stage: Run ASR
? Please enter the path to the preprocessed audio file: data/output/giangoi-20250815-091231/preprocessed.wav
? Please enter the path for the output ASR JSON file [data/output/giangoi-20250815-091231/asr.json]:
? Please enter the path to the model directory (optional) [models/chunkformer/chunkformer-large-vie]:
? Please enter the path to the config file (optional) [config.yaml]:
[INFO] Running ASR...
[SUCCESS] ASR complete. Output: data/output/giangoi-20250815-091231/asr.json
```

#### 3.2.3. Post-process ASR Output

1.  **Prompt for ASR JSON File:** Ask for the path to the ASR output JSON file.

    - **Prompt Type:** `questionary.text`
    - **Validation:** Ensure the file exists.

2.  **Prompt for Output Final JSON File:** Ask for the path to save the final post-processed output.

    - **Prompt Type:** `questionary.text`
    - **Default:** `{output_directory}/final.json`

3.  **Prompt for Config File (Optional):** Same as in the full pipeline.

4.  **Execution:** Run the post-processing step.
5.  **Completion:** Inform the user of success and the output file path.

**Example User Flow:**

```
? Select an action: Run Individual Stages
? Select a stage: Post-process ASR Output
? Please enter the path to the ASR JSON file: data/output/giangoi-20250815-091231/asr.json
? Please enter the path for the output final JSON file [data/output/giangoi-20250815-091231/final.json]:
? Please enter the path to the config file (optional) [config.yaml]:
[INFO] Post-processing ASR output...
[SUCCESS] Post-processing complete. Output: data/output/giangoi-20250815-091231/final.json
```

### 3.3. Configuration

This section allows users to view and modify application settings. For simplicity, this initial version will focus on displaying the current configuration and its source.

1.  **Display Current Configuration:** Show the key settings from the loaded `config.yaml` (or the custom config file if used).

    - Paths (output directory, model directory, chunkformer directory)
    - ASR parameters (device, chunk size, context sizes, etc.)
    - LLM parameters (model ID, quantization, max model length, etc.)
    - Output settings (transcript file, summary file)

2.  **Show Configuration Source:** Indicate which configuration file is being used (e.g., `config.yaml` or a custom path).

3.  **Future Enhancements:**
    - Interactive modification of configuration values.
    - Saving changes to a new or existing configuration file.

**Example User Flow:**

```
? Select an action: Configuration
--- Current Configuration ---
Source: config.yaml
paths:
  out_dir: data/output
  chunkformer_checkpoint: models/chunkformer/chunkformer-large-vie
  chunkformer_dir: chunkformer
asr:
  device: cuda
  chunk_size: 64
  ...
llm:
  model_id: vilm/vinallm-7b-chat
  ...
output:
  transcript_file: transcript.txt
  summary_file: summary.txt
...
---
Press Enter to return to the main menu.
```

### 3.4. Exit

This option will terminate the interactive CLI session.

```
? Select an action: Exit
Goodbye!
```

## 4. Implementation Considerations

- **Error Handling:** The CLI should gracefully handle errors, such as invalid file paths, missing dependencies, or failures during pipeline execution. Clear, user-friendly error messages should be displayed.
- **Progress Indicators:** For long-running tasks, such as ASR and post-processing, progress indicators (e.g., a spinner or a progress bar if feasible) should be shown to keep the user informed.
- **Defaults:** Sensible defaults should be provided for all prompts to minimize user input. These defaults should be derived from the `config.yaml` file or common usage patterns.
- **Back Navigation:** Where appropriate, consider adding an option to go back to the previous menu or the main menu.
- **Help Text:** For each prompt, clear and concise help text should be available to guide the user on what input is expected.
