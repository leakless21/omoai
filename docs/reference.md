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

The API uses a robust, script-based pipeline for all stages:

- Preprocess via ffmpeg
- ASR via the `scripts.asr` module
- Postprocess via the `scripts.post` module

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

| Parameter                 | Type    | Default                                  | Description                                                       |
| ------------------------- | ------- | ---------------------------------------- | ----------------------------------------------------------------- |
| `formats`                 | array   | `["json"]`                               | Output formats: `json`, `text`, `srt`, `vtt`, `md`                |
| `include`                 | array   | `["transcript_raw", "transcript_punct"]` | What to include: `transcript_raw`, `transcript_punct`, `segments` |
| `ts`                      | string  | `none`                                   | Timestamp format: `none`, `s`, `ms`, `clock`                      |
| `summary`                 | string  | `both`                                   | Summary type: `bullets`, `abstract`, `both`, `none`               |
| `summary_bullets_max`     | integer | `7`                                      | Maximum number of bullet points                                   |
| `summary_lang`            | string  | `vi`                                     | Summary language code                                             |
| `include_quality_metrics` | boolean | `false`                                  | Include quality metrics in response                               |
| `include_diffs`           | boolean | `false`                                  | Include human-readable diffs                                      |

#### Example Request

```bash
curl -X POST "http://localhost:8000/pipeline?summary=both&include=segments" \
  -F "audio_file=@/path/to/audio.mp3" \
  -H "Accept: application/json"
```

#### Response

Returns a `PipelineResponse` object with the following structure:

The `summary` field can be either a dictionary (containing "abstract" and/or "bullets" keys) or a string.

**Dictionary format example:**

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
  "transcript_raw": "the full raw transcript text without punctuation",
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

**String format example:**

```json
{
  "summary": "A concise summary of the audio content, presented as a single string.",
  "segments": [
    {
      "start": 0.5,
      "end": 3.2,
      "text": "Transcribed text segment",
      "confidence": 0.95
    }
  ],
  "transcript_punct": "The full punctuated transcript text.",
  "transcript_raw": "the full raw transcript text without punctuation",
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

The response now includes the raw transcription by default.

```json
{
  "segments": [
    {
      "start": 0.5,
      "end": 3.2,
      "text": "Transcribed text segment",
      "confidence": 0.95
    }
  ],
  "transcript_raw": "transcribed text segment"
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

**Note:** The `summary` field can also be returned as a string:

```json
{
  "summary": "Abstract summary text as a single string.",
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
    "models": { "status": "script_based" }
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

## Service Mode

There is no in-memory service mode. The API always uses the script-based pipeline for reliability and predictable resource usage.

## OpenAPI Documentation

Interactive API documentation is available at `/schema` when the server is running. This provides:

- Interactive endpoint testing
- Request/response schema validation
- Example requests and responses
- Real-time API exploration

Access the documentation at: http://localhost:8000/schema

# Pipeline Endpoint Guide

This document provides a comprehensive guide to the `/pipeline/` endpoint, including detailed examples of different configurations and their responses.

## Overview

The `/pipeline/` endpoint is the primary interface to the OmoAI audio processing system. It executes the complete workflow:

1. **Preprocessing**: Converts and normalizes audio for optimal ASR performance
2. **ASR**: Transcribes audio using the Chunkformer model
3. **Post-processing**: Applies punctuation, capitalization, and generates summaries

The endpoint is highly configurable through query parameters, allowing you to control exactly what data is returned and in what format.

## Endpoint Details

- **URL**: `/pipeline`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Authentication**: None (currently)

## Request Format

### Basic Request

```bash
curl -X POST "http://localhost:8000/pipeline" \
  -F "audio_file=@/path/to/audio.mp3" \
  -H "Accept: application/json"
```

### Request with Query Parameters

```bash
curl -X POST "http://localhost:8000/pipeline?summary=both&include=segments&ts=ms" \
  -F "audio_file=@/path/to/audio.mp3" \
  -H "Accept: application/json"
```

## Query Parameters

All query parameters are optional. If none are provided, the endpoint returns a default response with punctuated transcript and summary.

| Parameter                 | Type    | Default                                  | Description                                                       |
| ------------------------- | ------- | ---------------------------------------- | ----------------------------------------------------------------- |
| `formats`                 | array   | `["json"]`                               | Output formats: `json`, `text`, `srt`, `vtt`, `md`                |
| `include`                 | array   | `["transcript_raw", "transcript_punct"]` | What to include: `transcript_raw`, `transcript_punct`, `segments` |
| `ts`                      | string  | `none`                                   | Timestamp format: `none`, `s`, `ms`, `clock`                      |
| `summary`                 | string  | `both`                                   | Summary type: `bullets`, `abstract`, `both`, `none`               |
| `summary_bullets_max`     | integer | `7`                                      | Maximum number of bullet points                                   |
| `summary_lang`            | string  | `vi`                                     | Summary language code                                             |
| `include_quality_metrics` | boolean | `false`                                  | Include quality metrics in response                               |
| `include_diffs`           | boolean | `false`                                  | Include human-readable diffs                                      |

## Summary Field Format

The `summary` field in the response can be returned in one of two formats:

### 1. Dictionary Format (Default)

When the summary is generated with both bullets and abstract, it's returned as a dictionary:

```json
{
  "summary": {
    "bullets": [
      "Key point from the audio content",
      "Another important point",
      "Final summary point"
    ],
    "abstract": "A concise abstract summarizing the main points of the audio content."
  }
}
```

### 2. String Format

When the summary is generated as a simple text summary, it's returned as a string:

```json
{
  "summary": "A concise summary of the audio content, presented as a single string."
}
```

The format returned depends on the processing configuration and the content of the audio. Both formats provide the same essential information, just structured differently for optimal use cases.

## Response Examples

### 1. Default Response (No Query Parameters)

When no query parameters are provided, the endpoint returns a minimal response with the raw transcript, punctuated transcript, and summary.

**Request:**

```bash
curl -X POST "http://localhost:8000/pipeline" \
  -F "audio_file=@podcast.mp3" \
  -H "Accept: application/json"
```

**Response:**

The `summary` field in the response can be either a dictionary (containing "abstract" and/or "bullets" keys) or a string. Here's an example with the dictionary format:

```json
{
  "summary": {
    "bullets": [
      "The podcast discusses the future of artificial intelligence",
      "Experts debate the timeline for AGI development",
      "Ethical considerations are paramount in AI development"
    ],
    "abstract": "A thoughtful discussion about artificial intelligence's future, featuring expert opinions on AGI timelines and the importance of ethical considerations in AI development."
  },
  "segments": [],
  "transcript_punct": "Welcome to today's podcast. We're discussing the future of artificial intelligence. Our experts believe that AGI might be developed within the next decade. However, ethical considerations must be at the forefront of this development."
}
```

And here's an example with the string format:

```json
{
  "summary": "A thoughtful discussion about artificial intelligence's future, featuring expert opinions on AGI timelines and the importance of ethical considerations in AI development.",
  "segments": [],
  "transcript_punct": "Welcome to today's podcast. We're discussing the future of artificial intelligence. Our experts believe that AGI might be developed within the next decade. However, ethical considerations must be at the forefront of this development."
}
```

### 2. Full Response with Segments and Raw Transcript

**Request:**

```bash
curl -X POST "http://localhost:8000/pipeline?include=segments" \
  -F "audio_file=@interview.mp3" \
  -H "Accept: application/json"
```

**Response:**

```json
{
  "summary": {
    "bullets": [
      "The interview covers machine learning applications",
      "Guest shares insights from their research",
      "Future trends in AI are discussed"
    ],
    "abstract": "An insightful interview about machine learning applications, featuring expert research insights and discussion of future AI trends."
  },
  "segments": [
    {
      "start": 0.5,
      "end": 3.2,
      "text": "Welcome to the show.",
      "confidence": 0.98
    },
    {
      "start": 3.5,
      "end": 8.1,
      "text": "Today we're discussing machine learning applications in healthcare.",
      "confidence": 0.96
    },
    {
      "start": 8.5,
      "end": 12.3,
      "text": "Our guest has extensive research experience in this field.",
      "confidence": 0.94
    }
  ],
  "transcript_punct": "Welcome to the show. Today we're discussing machine learning applications in healthcare. Our guest has extensive research experience in this field.",
  "transcript_raw": "welcome to the show today were discussing machine learning applications in healthcare our guest has extensive research experience in this field"
}
```

### 3. Bullet Points Only Summary

**Request:**

```bash
curl -X POST "http://localhost:8000/pipeline?summary=bullets&summary_bullets_max=5" \
  -F "audio_file=@lecture.mp3" \
  -H "Accept: application/json"
```

**Response:**

````json
{
  "summary": {
    "bullets": [
      "Quantum computing represents a paradigm shift",
      "Qubits can exist in multiple states simultaneously",
      "Entanglement enables quantum teleportation",
      "Quantum algorithms can solve certain problems exponentially faster",
      "Current quantum computers are in the NISQ era"
    ]
  },
### 6. String Format Summary

**Request:**

```bash
curl -X POST "http://localhost:8000/pipeline" \
  -F "audio_file=@presentation.mp3" \
  -H "Accept: application/json"
````

**Response:**

```json
{
  "summary": "This presentation explores the intersection of blockchain technology and sustainable development. It examines how blockchain's transparency, immutability, and decentralization features can address challenges in supply chain management, renewable energy trading, and carbon credit verification.",
  "segments": [],
  "transcript_punct": "Good morning, everyone. Today's presentation explores the intersection of blockchain technology and sustainable development. We'll examine how blockchain's transparency, immutability, and decentralization features can address challenges in supply chain management, renewable energy trading, and carbon credit verification. Throughout this presentation, we'll highlight both opportunities and challenges in implementing blockchain solutions for environmental sustainability."
}
```

"segments": [],
"transcript_punct": "Quantum computing represents a fundamental paradigm shift in computation. Unlike classical bits, qubits can exist in multiple states simultaneously through superposition. The phenomenon of entanglement enables quantum teleportation and quantum communication. Quantum algorithms can solve certain problems exponentially faster than classical algorithms. However, current quantum computers are in the NISQ era, facing significant challenges in error correction and scalability."
}

````

### 4. Abstract Summary Only

**Request:**

```bash
curl -X POST "http://localhost:8000/pipeline?summary=abstract" \
  -F "audio_file=@presentation.mp3" \
  -H "Accept: application/json"
````

**Response:**

```json
{
  "summary": {
    "abstract": "This presentation explores the intersection of blockchain technology and sustainable development. It examines how blockchain's transparency, immutability, and decentralization features can address challenges in supply chain management, renewable energy trading, and carbon credit verification. The presentation highlights both opportunities and challenges in implementing blockchain solutions for environmental sustainability."
  },
  "segments": [],
  "transcript_punct": "Good morning, everyone. Today's presentation explores the intersection of blockchain technology and sustainable development. We'll examine how blockchain's transparency, immutability, and decentralization features can address challenges in supply chain management, renewable energy trading, and carbon credit verification. Throughout this presentation, we'll highlight both opportunities and challenges in implementing blockchain solutions for environmental sustainability."
}
```

### 5. No Summary

**Request:**

```bash
curl -X POST "http://localhost:8000/pipeline?summary=none" \
  -F "audio_file=@recording.mp3" \
  -H "Accept: application/json"
```

**Response:**

```json
{
  "summary": {},
  "segments": [],
  "transcript_punct": "The meeting minutes from last session have been distributed. Please review them before our next discussion. We need to finalize the budget proposal by Friday. All department heads should submit their reports by Wednesday."
}
```

### 6. Text Format Response

**Request:**

```bash
curl -X POST "http://localhost:8000/pipeline?formats=text" \
  -F "audio_file=@meeting.mp3" \
  -H "Accept: text/plain"
```

**Response:**

```
The quarterly review meeting covered several key topics. Financial performance exceeded expectations with a 15% growth in revenue. The marketing team presented their new campaign strategy. HR announced upcoming changes to remote work policies.

# Summary Points
- Financial performance showed 15% revenue growth
- New marketing campaign strategy was presented
- HR announced remote work policy changes
- Budget approval process will be streamlined

# Abstract
The quarterly review meeting highlighted strong financial performance with 15% revenue growth, introduced a new marketing campaign strategy, and announced upcoming remote work policy changes. The meeting also discussed streamlining the budget approval process.
```

### 7. SRT Format Response with Timestamps

**Request:**

```bash
curl -X POST "http://localhost:8000/pipeline?formats=srt&include=segments&ts=ms" \
  -F "audio_file=@video.mp3" \
  -H "Accept: text/plain"
```

**Response:**

```srt
1
00:00:00,500 --> 00:00:03,200
Welcome to today's tutorial.

2
00:00:03,500 --> 00:00:07,800
We'll be learning about neural networks and deep learning.

3
00:00:08,100 --> 00:00:12,400
Let's start with the basic concepts of artificial neurons.
```

### 8. VTT Format Response

**Request:**

```bash
curl -X POST "http://localhost:8000/pipeline?formats=vtt&include=segments&ts=ms" \
  -F "audio_file=@webinar.mp3" \
  -H "Accept: text/plain"
```

**Response:**

```vtt
WEBVTT

00:00:00.500 --> 00:00:03.200
Welcome to today's tutorial.

00:00:03.500 --> 00:00:07.800
We'll be learning about neural networks and deep learning.

00:00:08.100 --> 00:00:12.400
Let's start with the basic concepts of artificial neurons.
```

### 9. Markdown Format Response

**Request:**

```bash
curl -X POST "http://localhost:8000/pipeline?formats=md&include=segments&ts=clock" \
  -F "audio_file=@conference.mp3" \
  -H "Accept: text/plain"
```

**Response:**

```md
# Conference Transcript

## 00:00:00 - 00:00:05

Ladies and gentlemen, welcome to our annual technology conference.

## 00:00:05 - 00:00:12

This year's theme is "Innovation in the Digital Age."

## 00:00:12 - 00:00:18

We have speakers from leading tech companies around the world.

# Summary

- Welcome to annual technology conference
- Theme is "Innovation in the Digital Age"
- Speakers from leading tech companies worldwide
```

### 10. Response with Quality Metrics

**Request:**

```bash
curl -X POST "http://localhost:8000/pipeline?include_quality_metrics=true" \
  -F "audio_file=@analysis.mp3" \
  -H "Accept: application/json"
```

**Response:**

```json
{
  "summary": {
    "bullets": [
      "Data analysis shows promising trends",
      "Machine learning models improved accuracy",
      "Recommendations for future work are provided"
    ],
    "abstract": "The analysis reveals promising trends in the dataset, with machine learning models showing significant accuracy improvements. Recommendations for future work include expanding the dataset and exploring advanced algorithms."
  },
  "segments": [],
  "transcript_punct": "Our data analysis shows promising trends. Machine learning models have improved accuracy by 15%. We recommend expanding the dataset and exploring advanced algorithms for future work.",
  "quality_metrics": {
    "wer": 0.045,
    "cer": 0.028,
    "per": 0.032,
    "uwer": 0.051,
    "fwer": 0.043,
    "alignment_confidence": 0.94
  }
}
```

### 11. Response with Quality Metrics and Diffs

**Request:**

```bash
curl -X POST "http://localhost:8000/pipeline?include_quality_metrics=true&include_diffs=true" \
  -F "audio_file=@research.mp3" \
  -H "Accept: application/json"
```

**Response:**

```json
{
  "summary": {
    "bullets": [
      "Research findings indicate positive results",
      "New methodology shows improved efficiency",
      "Future research directions are identified"
    ],
    "abstract": "The research findings indicate positive results with the new methodology showing improved efficiency. Several future research directions have been identified based on these outcomes."
  },
  "segments": [],
  "transcript_punct": "Our research findings indicate positive results. The new methodology shows improved efficiency. We have identified several future research directions based on these outcomes.",
  "quality_metrics": {
    "wer": 0.038,
    "cer": 0.025,
    "per": 0.021,
    "uwer": 0.042,
    "fwer": 0.035,
    "alignment_confidence": 0.96
  },
  "diffs": {
    "original_text": "our research findings indicate positive results the new methodology shows improved efficiency we have identified several future research directions based on these outcomes",
    "punctuated_text": "Our research findings indicate positive results. The new methodology shows improved efficiency. We have identified several future research directions based on these outcomes.",
    "diff_output": "Our research findings indicate positive results.\n+ The\n new methodology shows improved efficiency.\n+ We\n have identified several future research directions based on these outcomes.",
    "alignment_summary": "Successfully added punctuation and capitalization. 3 sentences were formed from the original text with perfect alignment."
  }
}
```

### 12. Multiple Output Formats

**Request:**

```bash
curl -X POST "http://localhost:8000/pipeline?formats=json,srt,vtt&include=segments&ts=ms" \
  -F "audio_file=@multimedia.mp3" \
  -H "Accept: application/json"
```

**Response:**

```json
{
  "summary": {
    "bullets": [
      "Multimedia content creation is evolving",
      "New tools enable better storytelling",
      "Audience engagement is key to success"
    ],
    "abstract": "The evolution of multimedia content creation is discussed, highlighting how new tools enable better storytelling and emphasizing that audience engagement remains key to success."
  },
  "segments": [
    {
      "start": 0.5,
      "end": 3.2,
      "text": "Multimedia content creation is evolving rapidly.",
      "confidence": 0.97
    },
    {
      "start": 3.5,
      "end": 7.8,
      "text": "New tools are enabling creators to tell better stories.",
      "confidence": 0.95
    }
  ],
  "transcript_punct": "Multimedia content creation is evolving rapidly. New tools are enabling creators to tell better stories. Audience engagement is key to success.",
  "formats": {
    "srt": "1\n00:00:00,500 --> 00:00:03,200\nMultimedia content creation is evolving rapidly.\n\n2\n00:00:03,500 --> 00:00:07,800\nNew tools are enabling creators to tell better stories.",
    "vtt": "WEBVTT\n\n00:00:00.500 --> 00:00:03.200\nMultimedia content creation is evolving rapidly.\n\n00:00:03.500 --> 00:00:07.800\nNew tools are enabling creators to tell better stories."
  }
}
```

## Advanced Usage

### Custom Summary Language

**Request:**

```bash
curl -X POST "http://localhost:8000/pipeline?summary_lang=en" \
  -F "audio_file=@vietnamese_audio.mp3" \
  -H "Accept: application/json"
```

**Response:**

```json
{
  "summary": {
    "bullets": [
      "The Vietnamese audio discusses cultural heritage",
      "Traditional practices are highlighted",
      "Modern influences are examined"
    ],
    "abstract": "The Vietnamese audio content discusses cultural heritage, highlighting traditional practices while examining modern influences on contemporary society."
  },
  "segments": [],
  "transcript_punct": "Audio tiếng Việt discusses di sản văn hóa, thực hành truyền thống được nhấn mạnh, ảnh hưởng hiện đại được xem xét."
}
```

### Clock Timestamp Format

**Request:**

```bash
curl -X POST "http://localhost:8000/pipeline?include=segments&ts=clock" \
  -F "audio_file=@long_form.mp3" \
  -H "Accept: application/json"
```

**Response:**

```json
{
  "summary": {
    "bullets": [
      "Long-form content requires careful planning",
      "Audience retention strategies are important",
      "Production quality impacts viewer experience"
    ],
    "abstract": "A discussion about long-form content creation, covering planning requirements, audience retention strategies, and the impact of production quality on viewer experience."
  },
  "segments": [
    {
      "start": 0.5,
      "end": 125.3,
      "text": "Long-form content requires careful planning and execution.",
      "confidence": 0.96
    },
    {
      "start": 126.0,
      "end": 248.7,
      "text": "Audience retention strategies are crucial for success.",
      "confidence": 0.94
    }
  ],
  "transcript_punct": "Long-form content requires careful planning and execution. Audience retention strategies are crucial for success. Production quality significantly impacts the overall viewer experience."
}
```

## Error Handling

### Invalid Audio File

**Request:**

```bash
curl -X POST "http://localhost:8000/pipeline" \
  -F "audio_file=@invalid_file.txt" \
  -H "Accept: application/json"
```

**Response:**

```json
{
  "status_code": 422,
  "detail": "Audio processing failed: Unsupported audio format",
  "extra": {
    "error_type": "AudioProcessingException",
    "supported_formats": ["mp3", "wav", "flac", "m4a"]
  }
}
```

### File Too Large

**Request:**

```bash
curl -X POST "http://localhost:8000/pipeline" \
  -F "audio_file=@huge_file.mp3" \
  -H "Accept: application/json"
```

**Response:**

```json
{
  "status_code": 413,
  "detail": "Request entity too large: File size exceeds maximum allowed size of 100MB",
  "extra": {
    "file_size": "150MB",
    "max_allowed": "100MB"
  }
}
```

## Best Practices

1. **Choose the right response format** for your use case:

   - Use `json` for programmatic integration
   - Use `text` for simple display
   - Use `srt` or `vtt` for video subtitles
   - Use `md` for documentation or reports

2. **Request only what you need** to minimize response size:

   - Use `include` parameter to specify required data
   - Use `summary` parameter to control summary type
   - Use `ts=none` if timestamps aren't needed

3. **Handle large files appropriately**:

   - Consider splitting very long audio files
   - Monitor processing time for files approaching size limits

4. **Use quality metrics for debugging**:

   - Enable `include_quality_metrics` during development
   - Use `include_diffs` to understand punctuation improvements

5. **Consider language settings**:
   - Set `summary_lang` appropriately for your content
   - Note that transcription language depends on the ASR model

# Progress Reporting

The OmoAI API provides a progress reporting feature for long-running asynchronous tasks. This allows you to monitor the progress of a task as it moves through the processing pipeline.

## How it Works

When you submit an asynchronous task to the `POST /pipeline` endpoint, you will receive a `task_id`. You can use this `task_id` to poll the `GET /pipeline/status/{task_id}` endpoint to get the status of your task.

The response from the status endpoint will include a `progress` field, which is a number from 0 to 100 that represents the estimated progress of the task.

## Progress Calculation

The progress percentage is calculated based on the current stage of the pipeline:

- **Pending:** 0%
- **Preprocessing:** 10%
- **ASR:** 80%
- **Postprocessing:** 90%
- **Success:** 100%

## Example

Here is an example of a response from the `GET /pipeline/status/{task_id}` endpoint showing the progress:

```json
{
  "task_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
  "status": "running",
  "progress": 80.0,
  "result": null,
  "errors": [],
  "submitted_at": "2024-09-09T09:03:00Z",
  "started_at": "2024-09-09T09:03:05Z",
  "completed_at": null
}
```

This indicates that the task is currently in the ASR stage.
