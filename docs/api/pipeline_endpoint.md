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

## Response Examples

### 1. Default Response (No Query Parameters)

When no query parameters are provided, the endpoint returns a minimal response with just the punctuated transcript and summary.

**Request:**

```bash
curl -X POST "http://localhost:8000/pipeline" \
  -F "audio_file=@podcast.mp3" \
  -H "Accept: application/json"
```

**Response:**

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

### 2. Full Response with Segments and Raw Transcript

**Request:**

```bash
curl -X POST "http://localhost:8000/pipeline?include=transcript_raw,segments" \
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

```json
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
  "segments": [],
  "transcript_punct": "Quantum computing represents a fundamental paradigm shift in computation. Unlike classical bits, qubits can exist in multiple states simultaneously through superposition. The phenomenon of entanglement enables quantum teleportation and quantum communication. Quantum algorithms can solve certain problems exponentially faster than classical algorithms. However, current quantum computers are in the NISQ era, facing significant challenges in error correction and scalability."
}
```

### 4. Abstract Summary Only

**Request:**

```bash
curl -X POST "http://localhost:8000/pipeline?summary=abstract" \
  -F "audio_file=@presentation.mp3" \
  -H "Accept: application/json"
```

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
