# OMOAI Interactive CLI User Guide

This guide provides instructions on how to use the OMOAI Interactive CLI, a user-friendly command-line interface for transcribing and summarizing audio files.

## Table of Contents

1.  [Prerequisites](#prerequisites)
2.  [Launching the Interactive CLI](#launching-the-interactive-cli)
3.  [Main Menu Options](#main-menu-options)
    - [Run Full Pipeline](#run-full-pipeline)
    - [Run Individual Stages](#run-individual-stages)
    - [Configuration](#configuration)
    - [Exit](#exit)
4.  [Feature Walkthrough](#feature-walkthrough)
    - [Running the Full Pipeline](#running-the-full-pipeline)
    - [Running Individual Stages](#running-individual-stages-1)
      - [Preprocess Audio](#preprocess-audio)
      - [Run ASR](#run-asr)
      - [Post-process ASR Output](#post-process-asr-output)
5.  [Troubleshooting](#troubleshooting)

## Prerequisites

Before using the interactive CLI, ensure you have the following:

- Python 3.8 or higher installed.
- The OMOAI project and its dependencies installed. See the main [`README.md`](README.md) for setup instructions.
- An audio file in a supported format (e.g., MP3, WAV).
- A pre-trained ChunkFormer model. The default path is `models/chunkformer/chunkformer-large-vie`, but you can specify a different path.

## Launching the Interactive CLI

To launch the interactive CLI, run the following command from your project's root directory:

```bash
python main.py --interactive
```

This will start the interactive session and display the welcome message and main menu.

## Main Menu Options

Upon launching, you will be presented with a main menu with the following options:

- **Run Full Pipeline**: Execute the entire audio processing pipeline, from preprocessing to post-processing.
- **Run Individual Stages**: Execute a specific stage of the pipeline (preprocessing, ASR, or post-processing).
- **Configuration**: View the current configuration settings.
- **Exit**: Terminate the interactive session.

You can navigate the menu using the arrow keys and select an option by pressing `Enter`.

## Feature Walkthrough

### Running the Full Pipeline

The **Run Full Pipeline** option guides you through the complete process of transcribing and summarizing an audio file.

1.  **Select "Run Full Pipeline"** from the main menu.
2.  **Enter the path to your audio file** when prompted. The CLI will validate that the file exists.
    - **Example**: `data/input/my_audio.mp3`
3.  **Specify an output directory**. The CLI will suggest a default path based on your audio file's name and the current timestamp. You can accept the default or provide a custom path.
    - **Example**: `data/output/my_audio-20230815-123456`
4.  **Optionally, provide a path to the model directory**. If you have a specific ChunkFormer model you want to use, you can provide its path here. Otherwise, the default from the configuration will be used.
    - **Example**: `models/chunkformer/my-custom-model`
5.  **Optionally, provide a path to a custom configuration file**. If you want to use a configuration file other than the default `config.yaml`, you can specify its path here.
    - **Example**: `config.custom.yaml`
6.  **Confirm your settings**. The CLI will display a summary of your choices and ask for confirmation before proceeding.
7.  **Wait for the pipeline to complete**. The CLI will show progress messages for each stage:
    - `[INFO] Preprocessing audio...`
    - `[INFO] Running ASR...`
    - `[INFO] Post-processing ASR output...`
8.  **View the results**. Once the pipeline is complete, the CLI will display a success message and the path to the output directory, which will contain:
    - `transcript.txt`: The final transcript with punctuation.
    - `summary.txt`: The summary of the audio.
    - `final.json`: A JSON file containing both the transcript and summary.

**Example Session:**

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

### Running Individual Stages

The **Run Individual Stages** option allows you to run a specific part of the pipeline. This is useful if you have already processed some stages and want to resume or debug a particular step.

#### Preprocess Audio

This stage converts your audio file into the required format for the ASR model.

1.  From the main menu, select **Run Individual Stages** and then **Preprocess Audio**.
2.  **Enter the path to your input audio file**.
    - **Example**: `data/input/giangoi.mp3`
3.  **Enter the path for the output WAV file**. The CLI will provide a default path.
    - **Example**: `data/output/giangoi-20250815-091231/preprocessed.wav`
4.  The CLI will run the preprocessing command and display the result.

**Example Session:**

```
? Select an action: Run Individual Stages
? Select a stage: Preprocess Audio
? Please enter the path to the input audio file: data/input/giangoi.mp3
? Please enter the path for the output WAV file [data/output/giangoi-20250815-091231/preprocessed.wav]:
[INFO] Preprocessing audio...
[SUCCESS] Preprocessing complete. Output: data/output/giangoi-20250815-091231/preprocessed.wav
```

#### Run ASR

This stage transcribes the preprocessed audio file into text.

1.  From the main menu, select **Run Individual Stages** and then **Run ASR**.
2.  **Enter the path to your preprocessed WAV file**.
    - **Example**: `data/output/giangoi-20250815-091231/preprocessed.wav`
3.  **Enter the path for the output ASR JSON file**. This file will contain the raw transcription.
    - **Example**: `data/output/giangoi-20250815-091231/asr.json`
4.  **Optionally, provide a model directory and config file** as described in the full pipeline.
5.  The CLI will run the ASR process and display the result.

**Example Session:**

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

#### Post-process ASR Output

This stage adds punctuation and summarizes the raw ASR transcription.

1.  From the main menu, select **Run Individual Stages** and then **Post-process ASR Output**.
2.  **Enter the path to your ASR JSON output file**.
    - **Example**: `data/output/giangoi-20250815-091231/asr.json`
3.  **Enter the path for the final JSON output file**. This file will contain the polished transcript and summary.
    - **Example**: `data/output/giangoi-20250815-091231/final.json`
4.  **Optionally, provide a config file**.
5.  The CLI will run the post-processing script and display the result.

**Example Session:**

```
? Select an action: Run Individual Stages
? Select a stage: Post-process ASR Output
? Please enter the path to the ASR JSON file: data/output/giangoi-20250815-091231/asr.json
? Please enter the path for the output final JSON file [data/output/giangoi-20250815-091231/final.json]:
? Please enter the path to the config file (optional) [config.yaml]:
[INFO] Post-processing ASR output...
[SUCCESS] Post-processing complete. Output: data/output/giangoi-20250815-091231/final.json
```

### Configuration

This option allows you to view the current configuration settings loaded from `config.yaml`. You can see paths for models and output, ASR parameters, LLM settings, and more. This is useful for verifying that the CLI is using the settings you expect.

1.  **Select "Configuration"** from the main menu.
2.  The CLI will display the current configuration settings.
3.  Press any key to return to the main menu.

**Example Output:**

```
=== Configuration ===
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
Press any key to return to the main menu...
```

## Exit

To close the interactive CLI, select **Exit** from the main menu or press `Ctrl+C` at any time.

## Troubleshooting

- **File not found errors**: Ensure that the file paths you enter are correct and that the files exist.
- **Permission errors**: Make sure you have read/write permissions for the input audio files and the output directories.
- **CUDA out of memory**: If you are using a GPU and encounter memory issues, try reducing the `chunk_size` in your configuration file or use a smaller model.
- **Model loading errors**: Verify that the model directory path is correct and that all model files are present.
- **Configuration errors**: Check your `config.yaml` file for syntax errors. You can use a YAML linter to validate it.
