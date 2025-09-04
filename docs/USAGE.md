# OMOAI Usage Guide

This document provides detailed usage instructions for the OMOAI audio processing pipeline, including the enhanced punctuation alignment functionality.

## Table of Contents

1. [Quick Start](#quick-start)
2. [CLI Usage](#cli-usage)
3. [API Usage](#api-usage)
4. [Enhanced Punctuation Alignment](#enhanced-punctuation-alignment)
5. [Configuration](#configuration)
6. [Output Formats](#output-formats)
7. [Quality Metrics](#quality-metrics)

## Quick Start

### Basic CLI Usage

```bash
# Process an audio file with full pipeline
uv run start data/input/your_audio.mp3

# Interactive mode
uv run interactive
```

### Basic API Usage

```bash
# Start the API server
uv run api

# Process audio via API
curl -X POST 'http://localhost:8000/pipeline' \
  -F 'audio_file=@data/input/audio.mp3'
```

## CLI Usage

### Full Pipeline

```bash
# Basic usage
uv run python src/omoai/main.py data/input/audio.mp3

# With custom config
uv run python src/omoai/main.py data/input/audio.mp3 \
  --config /path/to/custom_config.yaml

# With output directory
uv run python src/omoai/main.py data/input/audio.mp3 \
  --output-dir /path/to/output
```

### Stage-by-Stage Processing

```bash
# 1. Preprocessing
uv run python scripts/preprocess.py \
  --input data/input/audio.mp3 \
  --output data/output/preprocessed.wav

# 2. ASR
uv run python scripts/asr.py \
  --config config.yaml \
  --audio data/output/preprocessed.wav \
  --out data/output/asr.json \
  --auto-outdir

# 3. Post-processing (with enhanced punctuation)
uv run python scripts/post.py \
  --config config.yaml \
  --asr-json data/output/asr.json \
  --out data/output/final.json \
  --auto-outdir
```

## API Usage

### Endpoints

#### POST /pipeline

Run the complete audio processing pipeline.

```bash
# Basic request
curl -X POST 'http://localhost:8000/pipeline' \
  -F 'audio_file=@data/input/audio.mp3'

# With output formatting options
curl -X POST 'http://localhost:8000/pipeline?include=segments&ts=clock&summary=both' \
  -F 'audio_file=@data/input/audio.mp3'

# With quality metrics
curl -X POST 'http://localhost:8000/pipeline?include_quality_metrics=true' \
  -F 'audio_file=@data/input/audio.mp3'

# With diffs
curl -X POST 'http://localhost:8000/pipeline?include_diffs=true' \
  -F 'audio_file=@data/input/audio.mp3'

# With both quality metrics and diffs
curl -X POST 'http://localhost:8000/pipeline?include_quality_metrics=true&include_diffs=true' \
  -F 'audio_file=@data/input/audio.mp3'
```

#### POST /postprocess

Apply punctuation and summarization to existing ASR results.

```bash
curl -X POST 'http://localhost:8000/postprocess' \
  -H 'Content-Type: application/json' \
  -d '{
    "asr_output": {
      "segments": [
        {"start": 0.0, "end": 2.5, "text": "hello world"},
        {"start": 2.5, "end": 5.0, "text": "how are you today"}
      ],
      "transcript_raw": "hello world how are you today",
      "audio_duration": 5.0
    }
  }'
```

### Query Parameters

| Parameter                 | Description                  | Values                                           | Default            |
| ------------------------- | ---------------------------- | ------------------------------------------------ | ------------------ |
| `include`                 | What to include in response  | `transcript_raw`, `transcript_punct`, `segments` | `transcript_punct` |
| `include_quality_metrics` | Include quality metrics      | `true`, `false`                                  | `false`            |
| `include_diffs`           | Include human-readable diffs | `true`, `false`                                  | `false`            |
| `ts`                      | Timestamp format             | `none`, `s`, `ms`, `clock`                       | `none`             |
| `summary`                 | Summary type                 | `none`, `bullets`, `abstract`, `both`            | `both`             |
| `summary_bullets_max`     | Maximum bullet points        | 1-20                                             | 7                  |
| `summary_lang`            | Summary language             | `vi`, `en`                                       | `vi`               |

## Enhanced Punctuation Alignment

The OMOAI system now includes an enhanced punctuation alignment algorithm that provides superior accuracy when distributing punctuation from LLM output back to original ASR segments.

### Key Features

1. **Word-Level Alignment**: Uses Levenshtein distance to find optimal alignment between original and punctuated words
2. **Character-Level Refinement**: Performs granular character-level analysis for aligned word pairs
3. **Quality Metrics**: Computes WER, CER, PER, U-WER, and F-WER for comprehensive evaluation
4. **Human-Readable Diffs**: Generates side-by-side diffs for qualitative analysis

### Quality Metrics

The system provides multiple quality metrics:

- **WER (Word Error Rate)**: `(S + D + I) / (N + I)` where S=substitutions, D=deletions, I=insertions, N=original words
- **CER (Character Error Rate)**: Character-level error rate
- **PER (Punctuation Error Rate)**: Error rate specific to punctuation marks
- **U-WER (Unpunctuated WER)**: WER after removing punctuation
- **F-WER (Formatted WER)**: WER including punctuation

### Accessing Quality Metrics and Diffs

#### Via API

```bash
# With quality metrics only
curl -X POST 'http://localhost:8000/pipeline?include_quality_metrics=true' \
  -F 'audio_file=@data/input/audio.mp3'

# With diffs only
curl -X POST 'http://localhost:8000/pipeline?include_diffs=true' \
  -F 'audio_file=@data/input/audio.mp3'

# With both quality metrics and diffs
curl -X POST 'http://localhost:8000/pipeline?include_quality_metrics=true&include_diffs=true' \
  -F 'audio_file=@data/input/audio.mp3'
```

#### API Response Examples

**Quality Metrics Response:**

```json
{
  "quality_metrics": {
    "wer": 0.05,
    "cer": 0.03,
    "per": 0.02,
    "uwer": 0.04,
    "fwer": 0.05,
    "alignment_confidence": 0.98
  }
}
```

**Diffs Response:**

```json
{
  "diffs": {
    "original_text": "hello world how are you today",
    "punctuated_text": "Hello, world! How are you today?",
    "diff_output": "  Original: hello world how are you today\n  Punctuated: Hello, world! How are you today?\n  Changes:   ^     ^    ^              ^",
    "alignment_summary": "Successfully aligned 6 words with 2 punctuation additions"
  }
}
```

#### Via Interactive CLI

```bash
# Start interactive mode
uv run interactive

# Select "View Quality Metrics & Diffs"
# Enter path to your final.json file
# View detailed analysis and optionally save to file
```

The interactive CLI provides:

- 📊 Quality metrics with interpretation (excellent/good/acceptable/poor)
- 📝 Human-readable diffs showing before/after text
- 💾 Option to save analysis report to text file
- 📁 File information and statistics

### Configuration for Enhanced Punctuation

```yaml
punctuation:
  llm:
    model_id: cpatonn/Qwen3-4B-Instruct-2507-AWQ-4bit
    quantization: auto
    max_model_len: 50000
    gpu_memory_utilization: 0.90
  preserve_original_words: true
  adopt_case: true
  enable_paragraphs: true
  join_separator: " "
  paragraph_gap_seconds: 3.0
  # Fallback behavior for empty segments
  keep_nonempty_segments: false
  # Prompt-level mitigation for deletions
  prevent_deletions_prompt: true
  system_prompt: |
    You are a helpful assistant that restores punctuation to text.
    Output only the punctuated text.
    Do not delete any words from the original text.
  # Enhanced alignment settings
  alignment:
    use_levenshtein: true
    enable_character_level: true
    compute_quality_metrics: true
    generate_diffs: true

#### New Configuration Options

1. **keep_nonempty_segments** (boolean, default: false)

   When enabled, prevents empty `text_punct` fields by applying basic punctuation fallback:
   - Capitalizes the first letter of the segment
   - Adds a period at the end
   - Only applies to segments that would otherwise be empty after alignment

   This is useful when you want to ensure every segment has some punctuation, even if the LLM alignment process fails to map words properly.

2. The default punctuation system prompt in config.yaml already contains an anti-deletion policy; the separate `prevent_deletions_prompt` option has been removed.

#### Debugging Empty Segments

When you encounter empty `text_punct` fields, the system now provides debug logging to help diagnose the issue:

```

[punct-align] empty_map seg=3 cnt=0 keep_nonempty=false

````

This log indicates:
- Segment 3 mapped to 0 words (empty)
- `keep_nonempty_segments` was disabled, so no fallback was applied

To enable debug logging:
```bash
export OMOAI_LOG_LEVEL=DEBUG
uv run api
````

````

## Configuration

### Environment Variables

| Variable                  | Description                | Default         |
| ------------------------- | -------------------------- | --------------- |
| `OMOAI_CONFIG`            | Path to configuration file | `./config.yaml` |
| `OMOAI_DEBUG_EMPTY_CACHE` | Enable GPU cache clearing  | `false`         |

### Configuration File Structure

```yaml
# Main paths
paths:
  chunkformer_dir: ./src/chunkformer
  chunkformer_checkpoint: ./models/chunkformer/chunkformer-large-vie
  out_dir: ./data/output

# ASR configuration
asr:
  total_batch_duration_s: 1800
  chunk_size: 64
  device: auto
  autocast_dtype: fp16

# LLM configuration (shared)
llm:
  model_id: cpatonn/Qwen3-4B-Instruct-2507-AWQ-4bit
  quantization: auto
  max_model_len: 50000
  gpu_memory_utilization: 0.90

# Enhanced punctuation configuration
punctuation:
  llm: { ... } # Inherits from llm section if omitted
  preserve_original_words: true
  adopt_case: true
  enable_paragraphs: true
  # Fallback behavior for empty segments
  keep_nonempty_segments: false
  # Prompt-level mitigation for deletions
  prevent_deletions_prompt: true
  alignment:
    use_levenshtein: true
    enable_character_level: true
    compute_quality_metrics: true
  system_prompt: |
    You are a helpful assistant that restores punctuation to text.
    Do not delete any words from the original text.

# Summarization configuration
summarization:
  llm: { ... } # Inherits from llm section if omitted
  map_reduce: false
  system_prompt: |
    You are a helpful assistant that summarizes text.

# API configuration
api:
  host: 0.0.0.0
  port: 8000
  max_body_size_mb: 100
  request_timeout_seconds: 300
````

## Output Formats

### JSON Output

The system always generates `final.json` with the following structure:

```json
{
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "text_raw": "hello world",
      "text_punct": "Hello, world!",
      "confidence": 0.95
    }
  ],
  "transcript_raw": "hello world how are you today",
  "transcript_punct": "Hello, world! How are you today?",
  "summary": {
    "bullets": ["First point", "Second point"],
    "abstract": "Summary text..."
  },
  "quality_metrics": {
    "wer": 0.05,
    "cer": 0.03,
    "per": 0.02,
    "uwer": 0.04,
    "fwer": 0.05
  },
  "metadata": {
    "processing_time": 15.2,
    "audio_duration": 30.5,
    "model_info": {...}
  }
}
```

### Text Files

When `write_separate_files: true`, the system generates:

- `transcript.txt`: Punctuated transcript
- `summary.txt`: Summary with bullets and abstract

### Formatted Outputs

The system supports multiple output formats through the API:

```bash
# Request SRT format
curl -X POST 'http://localhost:8000/pipeline?format=srt' \
  -F 'audio_file=@data/input/audio.mp3'

# Request VTT format
curl -X POST 'http://localhost:8000/pipeline?format=vtt' \
  -F 'audio_file=@data/input/audio.mp3'

# Request Markdown format
curl -X POST 'http://localhost:8000/pipeline?format=md' \
  -F 'audio_file=@data/input/audio.mp3'
```

## Quality Metrics

### Understanding the Metrics

1. **WER (Word Error Rate)**:

   - Measures word-level accuracy
   - Lower values indicate better performance
   - Formula: `(S + D + I) / (N + I)`

2. **CER (Character Error Rate)**:

   - Measures character-level accuracy
   - More sensitive to small changes
   - Formula: `(S + D + I) / N_chars`

3. **PER (Punctuation Error Rate)**:

   - Measures punctuation accuracy specifically
   - Computed only on punctuation characters
   - Formula: Same as WER but only for punctuation

4. **U-WER (Unpunctuated WER)**:

   - WER after removing all punctuation
   - Measures underlying word recognition quality

5. **F-WER (Formatted WER)**:
   - WER including punctuation
   - Measures overall quality including formatting

### Interpreting Results

- **WER < 0.05**: Excellent performance
- **WER 0.05-0.10**: Good performance
- **WER 0.10-0.15**: Acceptable performance
- **WER > 0.15**: Poor performance, may need investigation

### Quality-Based Decisions

The system can use quality metrics for automated decisions:

```yaml
punctuation:
  quality_thresholds:
    wer_max: 0.15
    cer_max: 0.10
    per_max: 0.08
  fallback_on_poor_quality: true
```

## Troubleshooting

### Common Issues

1. **Enhanced punctuation not working**:

   - Ensure `scripts/post.py` is accessible
   - Check that `alignment.use_levenshtein: true` in config
   - Verify LLM model is properly loaded

2. **Quality metrics not available**:

   - Set `alignment.compute_quality_metrics: true`
   - Check API response includes `quality_metrics` object

3. **Poor alignment quality**:

   - Increase `max_model_len` for better context

4. **Empty text_punct segments**:

   - Check debug logs for `[punct-align] empty_map` messages
   - Enable `keep_nonempty_segments: true` for fallback behavior
   - Enable `prevent_deletions_prompt: true` to reduce word deletions
   - Review system prompt to ensure it's not encouraging deletions
   - Adjust system prompt for clearer instructions
   - Consider using a more capable LLM model

5. **Empty text_punct segments**:
   - Check debug logs for `[punct-align] empty_map` messages
   - Enable `keep_nonempty_segments: true` for fallback behavior
   - Enable `prevent_deletions_prompt: true` to reduce word deletions
   - Review system prompt to ensure it's not encouraging deletions

### Debug Mode

Enable debug mode for detailed logging:

```bash
export OMOAI_DEBUG_EMPTY_CACHE=true
export OMOAI_LOG_LEVEL=DEBUG
uv run api
```

### Performance Optimization

1. **Memory Management**:

   - Adjust `gpu_memory_utilization` based on available GPU memory
   - Use quantized models for lower memory usage

2. **Speed Optimization**:

   - Reduce `max_model_len` for faster processing
   - Use smaller batch sizes for constrained hardware

3. **Quality Optimization**:
   - Use larger, more capable models for better punctuation
   - Enable all quality metrics for comprehensive evaluation
