# OmoAI API Reference

This document provides a comprehensive reference for the OmoAI RESTful API, including all endpoints, request/response formats, authentication, and usage examples.

## Table of Contents

- [Overview](#overview)
- [Authentication](#authentication)
- [Base URL](#base-url)
- [Endpoints](#endpoints)
  - [Pipeline Endpoint](#pipeline-endpoint)
  - [Preprocess Endpoint](#preprocess-endpoint)
  - [ASR Endpoint](#asr-endpoint)
  - [Postprocess Endpoint](#postprocess-endpoint)
  - [Health Check Endpoint](#health-check-endpoint)
- [Response Formats](#response-formats)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)
- [Service Modes](#service-modes)

## Overview

The OmoAI API provides a RESTful interface for processing audio files through a complete speech recognition and analysis pipeline. The API supports:

- **Audio Preprocessing**: Convert and normalize audio files for optimal ASR performance
- **Automatic Speech Recognition (ASR)**: Transcribe audio using the Chunkformer model
- **Post-processing**: Apply punctuation, capitalization, and generate summaries
- **Full Pipeline**: Execute the entire workflow in a single request

The API features dual-mode operation:

- **High-performance mode**: Uses in-memory processing with cached models for maximum speed
- **Robust mode**: Falls back to script-based processing for maximum compatibility

## Authentication

Currently, the API does not require authentication. This is suitable for local development and controlled environments. For production deployments, consider implementing authentication through:

- API keys
- OAuth 2.0
- JWT tokens

## Base URL

The base URL for the API depends on your deployment:

```
http://localhost:8000  # Local development
https://api.yourdomain.com  # Production deployment
```

## Endpoints

### Pipeline Endpoint

**POST** `/pipeline`

Execute the complete audio processing pipeline in a single request.

#### Request

- **Method**: POST
- **Content-Type**: multipart/form-data
- **Body**:
  - `audio_file`: Audio file to process (required)
  - Query parameters for output formatting (optional)

#### Query Parameters

| Parameter                 | Type    | Default                | Description                                                       |
| ------------------------- | ------- | ---------------------- | ----------------------------------------------------------------- |
| `formats`                 | array   | `["json"]`             | Output formats: `json`, `text`, `srt`, `vtt`, `md`                |
| `include`                 | array   | `["transcript_punct"]` | What to include: `transcript_raw`, `transcript_punct`, `segments` |
| `ts`                      | string  | `none`                 | Timestamp format: `none`, `s`, `ms`, `clock`                      |
| `summary`                 | string  | `both`                 | Summary type: `bullets`, `abstract`, `both`, `none`               |
| `summary_bullets_max`     | integer | `7`                    | Maximum number of bullet points                                   |
| `summary_lang`            | string  | `vi`                   | Summary language code                                             |
| `include_quality_metrics` | boolean | `false`                | Include quality metrics in response                               |
| `include_diffs`           | boolean | `false`                | Include human-readable diffs                                      |

#### Example Request

```bash
curl -X POST "http://localhost:8000/pipeline?summary=both&include=segments" \
  -F "audio_file=@/path/to/audio.mp3" \
  -H "Accept: application/json"
```

#### Response

Returns a `PipelineResponse` object with the following structure:

```json
{
  "summary": {
    "bullets": [
      "Key point from the audio content",
      "Another important point",
      "Final summary point"
    ],
    "abstract": "A concise abstract summarizing the main points of the audio content."
  },
  "segments": [
    {
      "start": 0.5,
      "end": 3.2,
      "text": "Transcribed text segment",
      "confidence": 0.95
    }
  ],
  "transcript_punct": "The full punctuated transcript text.",
  "quality_metrics": {
    "wer": 0.05,
    "cer": 0.03,
    "per": 0.02,
    "uwer": 0.06,
    "fwer": 0.04,
    "alignment_confidence": 0.92
  },
  "diffs": {
    "original_text": "unpunctuated text",
    "punctuated_text": "Punctuated text.",
    "diff_output": "diff visualization",
    "alignment_summary": "alignment details"
  }
}
```

#### Example Responses

See the [Pipeline Endpoint Guide](./pipeline_endpoint.md) for detailed examples of different response configurations.

### Preprocess Endpoint

**POST** `/preprocess`

Preprocess an audio file for ASR processing.

#### Request

- **Method**: POST
- **Content-Type**: multipart/form-data
- **Body**:
  - `audio_file`: Audio file to preprocess (required)

#### Example Request

```bash
curl -X POST "http://localhost:8000/preprocess" \
  -F "audio_file=@/path/to/audio.mp3" \
  -H "Accept: application/json"
```

#### Response

```json
{
  "output_path": "/tmp/preprocessed_audio.wav"
}
```

### ASR Endpoint

**POST** `/asr`

Perform automatic speech recognition on a preprocessed audio file.

#### Request

- **Method**: POST
- **Content-Type**: application/json
- **Body**:
  - `preprocessed_path`: Path to preprocessed audio file (required)

#### Example Request

```bash
curl -X POST "http://localhost:8000/asr" \
  -H "Content-Type: application/json" \
  -d '{"preprocessed_path": "/tmp/preprocessed_audio.wav"}'
```

#### Response

```json
{
  "segments": [
    {
      "start": 0.5,
      "end": 3.2,
      "text": "Transcribed text segment",
      "confidence": 0.95
    }
  ]
}
```

### Postprocess Endpoint

**POST** `/postprocess`

Apply post-processing to ASR output including punctuation and summarization.

#### Request

- **Method**: POST
- **Content-Type**: application/json
- **Body**:
  - `asr_output`: ASR output object (required)
  - Query parameters for post-processing options (optional)

#### Example Request

```bash
curl -X POST "http://localhost:8000/postprocess?summary=both" \
  -H "Content-Type: application/json" \
  -d '{"asr_output": {"segments": [...]}}'
```

#### Response

```json
{
  "summary": {
    "bullets": ["Key point 1", "Key point 2"],
    "abstract": "Abstract summary text"
  },
  "segments": [
    {
      "start": 0.5,
      "end": 3.2,
      "text": "Punctuated text segment",
      "confidence": 0.95
    }
  ],
  "quality_metrics": {
    "wer": 0.05,
    "cer": 0.03,
    "per": 0.02
  },
  "diffs": {
    "original_text": "unpunctuated text",
    "punctuated_text": "Punctuated text.",
    "diff_output": "diff visualization"
  }
}
```

### Health Check Endpoint

**GET** `/health`

Check the health status of the API and its dependencies.

#### Example Request

```bash
curl -X GET "http://localhost:8000/health"
```

#### Response

```json
{
  "status": "healthy",
  "details": {
    "ffmpeg": "available",
    "config_file": "found at /home/cetech/omoai/config.yaml",
    "asr_script": "found",
    "postprocess_script": "found",
    "temp_dir": "accessible at /tmp",
    "config_loaded": "yes",
    "max_body_size": "100MB",
    "service_mode": "auto",
    "in_memory_available": true,
    "models": {
      "asr_model": "loaded",
      "llm_model": "loaded"
    }
  }
}
```

## Response Formats

The API supports multiple response formats:

### JSON Format

Default format with structured data:

```json
{
  "summary": {...},
  "segments": [...],
  "transcript_punct": "Text"
}
```

### Text Format

Plain text response suitable for simple integration:

```
The punctuated transcript text.

# Summary Points
- First key point
- Second key point

# Abstract
The abstract summary text.
```

### SRT Format

SubRip subtitle format for video integration:

```srt
1
00:00:00,500 --> 00:00:03,200
Transcribed text segment

2
00:00:03,200 --> 00:00:06,800
Next text segment
```

### VTT Format

WebVTT format for web applications:

```vtt
WEBVTT

00:00:00.500 --> 00:00:03.200
Transcribed text segment

00:00:03.200 --> 00:00:06.800
Next text segment
```

### Markdown Format

Formatted markdown with timestamps:

```md
# Audio Transcript

## 00:00:00 - 00:00:03

Transcribed text segment

## 00:00:03 - 00:00:07

Next text segment

# Summary

- Key point 1
- Key point 2
```

## Error Handling

The API uses standard HTTP status codes and provides detailed error information:

### Status Codes

| Status Code | Description                                 |
| ----------- | ------------------------------------------- |
| 200         | Success                                     |
| 400         | Bad Request - Invalid input parameters      |
| 413         | Payload Too Large - File exceeds size limit |
| 422         | Unprocessable Entity - Validation error     |
| 500         | Internal Server Error - Processing failure  |
| 503         | Service Unavailable - Models not loaded     |

### Error Response Format

```json
{
  "status_code": 422,
  "detail": "Validation error: audio_file is required",
  "extra": {
    "field": "audio_file",
    "error": "required"
  }
}
```

## Rate Limiting

The API implements the following rate limits:

- **File size**: Maximum 100MB per request (configurable)
- **Request timeout**: 300 seconds (configurable)
- **Concurrent requests**: Limited by available system resources

Rate limits can be configured in the `config.yaml` file:

```yaml
api:
  max_body_size_mb: 100
  request_timeout_seconds: 300
```

## Service Modes

The API operates in three service modes:

### Auto Mode (Default)

Automatically selects the best available service:

- Uses high-performance in-memory processing when models are loaded
- Falls back to script-based processing if models are unavailable
- Provides the best balance of performance and reliability

### Memory Mode

Forces the use of high-performance in-memory processing:

- Maximum performance with cached models
- Lower latency for consecutive requests
- May fail if models cannot be loaded

### Script Mode

Forces the use of script-based processing:

- Maximum compatibility with different environments
- More robust to model loading issues
- Slower performance due to process overhead

### Configuring Service Mode

Service mode can be configured in three ways:

1. **Configuration file** (`config.yaml`):

```yaml
api:
  service_mode: "auto" # "auto", "memory", or "script"
```

2. **Environment variable**:

```bash
export OMOAI_SERVICE_MODE=memory
```

3. **Runtime override** (for testing):

```python
from src.omoai.api.services_enhanced import force_service_mode
await force_service_mode("script")
```

## OpenAPI Documentation

Interactive API documentation is available at `/schema` when the server is running. This provides:

- Interactive endpoint testing
- Request/response schema validation
- Example requests and responses
- Real-time API exploration

Access the documentation at: http://localhost:8000/schema
