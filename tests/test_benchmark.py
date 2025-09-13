import os
import time

import pytest
from litestar.testing import TestClient

from omoai.api.app import create_app

app = create_app()
client = TestClient(app)


@pytest.fixture
def audio_file(test_mp3_path):
    # Prefer provided MP3; otherwise skip benchmark
    return str(test_mp3_path) if test_mp3_path is not None else None


@pytest.mark.slow
def test_benchmark_pipeline(audio_file):
    """
    Benchmarks the full /pipeline endpoint.
    """
    if not audio_file or not os.path.exists(audio_file):
        pytest.skip(
            "Benchmark audio not provided; set OMOAI_TEST_MP3 or place tests/assets/testaudio.mp3"
        )

    with open(audio_file, "rb") as f:
        files = {"audio_file": (os.path.basename(audio_file), f, "audio/mpeg")}

        start_time = time.time()
        response = client.post("/pipeline", files=files)
        total_time = time.time() - start_time

    # Debug: Print response details if there's an error
    if response.status_code != 201:
        print(f"Response status: {response.status_code}")
        print(f"Response content: {response.content}")
        try:
            error_data = response.json()
            print(f"Error details: {error_data}")
        except Exception:
            print("Could not parse error as JSON")

    # Allow 200 or 201 depending on controller return configuration
    assert response.status_code in (200, 201), (
        f"Expected 200/201, got {response.status_code}: {response.content}"
    )
    data = response.json()

    # Extract transcript from segments
    segments = data.get("segments", [])
    transcription = " ".join([segment.get("text", "") for segment in segments])

    # Get summary
    summary = data.get("summary", {})
    bullets = summary.get("bullets", [])
    abstract = summary.get("abstract", "")

    # Create benchmark report
    report = f"""# Benchmark Report

**Audio File:** `{os.path.basename(audio_file)}`

## Processing Time

| Stage             | Time (seconds) |
|-------------------|----------------|
| **Total Pipeline** | **{total_time:.4f}**       |

*Note: Individual stage timings are not currently exposed by the API endpoint.*

## Transcription

### Full Transcript
```
{transcription}
```

### Segments ({len(segments)} total)
"""

    for _i, segment in enumerate(segments[:5]):  # Show first 5 segments
        start = segment.get("start", 0)
        end = segment.get("end", 0)
        text = segment.get("text", "")
        report += f"- [{start:.2f}s - {end:.2f}s]: {text}\n"

    if len(segments) > 5:
        report += f"- ... and {len(segments) - 5} more segments\n"

    report += f"""
## Summary

### Bullet Points ({len(bullets)} total)
"""
    for bullet in bullets:
        report += f"- {bullet}\n"

    if abstract:
        report += f"""
### Abstract
{abstract}
"""

    report += """
## Quality Assessment

*TODO: Manually assess the quality of the transcription by listening to the audio file and comparing it to the text.*

### Metrics
- Total segments: {}
- Average segment length: {:.2f} seconds
- Transcription length: {} characters
""".format(
        len(segments),
        sum(segment.get("end", 0) - segment.get("start", 0) for segment in segments)
        / len(segments)
        if segments
        else 0,
        len(transcription),
    )

    with open("docs/BENCHMARK_REPORT.md", "w", encoding="utf-8") as f:
        f.write(report)

    print("\nBenchmark report generated at docs/BENCHMARK_REPORT.md")
    print(f"Total processing time: {total_time:.4f} seconds")
    print(f"Segments processed: {len(segments)}")
    print(f"Transcription length: {len(transcription)} characters")
