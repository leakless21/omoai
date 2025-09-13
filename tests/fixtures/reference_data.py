"""
Reference data and expected outputs for golden testing.

This module provides expected transcripts, confidence scores, and performance
benchmarks for validating system accuracy and regression testing.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ReferenceTranscript:
    """Reference transcript with expected outputs."""

    audio_fixture: str
    expected_transcript: str
    expected_word_count: int
    expected_confidence_min: float
    expected_confidence_avg: float
    expected_segments: list[dict[str, Any]]
    metadata: dict[str, Any]


@dataclass
class PerformanceBenchmark:
    """Performance benchmarks for regression testing."""

    audio_fixture: str
    max_processing_time_ms: float
    max_real_time_factor: float
    max_memory_usage_mb: float
    expected_gpu_memory_mb: float | None
    baseline_metrics: dict[str, float]


@dataclass
class QualityBenchmark:
    """Quality benchmarks for accuracy testing."""

    audio_fixture: str
    min_word_accuracy: float
    min_character_accuracy: float
    min_confidence_score: float
    max_word_error_rate: float
    expected_punctuation_accuracy: float


class ReferenceDataManager:
    """Manages reference data for comprehensive testing."""

    def __init__(self, reference_dir: Path | None = None):
        self.reference_dir = reference_dir or Path(__file__).parent / "reference_data"
        self.reference_dir.mkdir(parents=True, exist_ok=True)

        self.transcripts: dict[str, ReferenceTranscript] = {}
        self.performance_benchmarks: dict[str, PerformanceBenchmark] = {}
        self.quality_benchmarks: dict[str, QualityBenchmark] = {}

        self._load_reference_data()

    def _load_reference_data(self):
        """Load all reference data from files."""
        # Load transcripts
        transcripts_file = self.reference_dir / "transcripts.yaml"
        if transcripts_file.exists():
            with open(transcripts_file) as f:
                data = yaml.safe_load(f)
                for name, transcript_data in data.items():
                    self.transcripts[name] = ReferenceTranscript(**transcript_data)
        else:
            self._create_default_transcripts()

        # Load performance benchmarks
        performance_file = self.reference_dir / "performance.yaml"
        if performance_file.exists():
            with open(performance_file) as f:
                data = yaml.safe_load(f)
                for name, perf_data in data.items():
                    self.performance_benchmarks[name] = PerformanceBenchmark(
                        **perf_data
                    )
        else:
            self._create_default_performance_benchmarks()

        # Load quality benchmarks
        quality_file = self.reference_dir / "quality.yaml"
        if quality_file.exists():
            with open(quality_file) as f:
                data = yaml.safe_load(f)
                for name, quality_data in data.items():
                    self.quality_benchmarks[name] = QualityBenchmark(**quality_data)
        else:
            self._create_default_quality_benchmarks()

    def _create_default_transcripts(self):
        """Create default reference transcripts."""
        default_transcripts = {
            "simple_tone": ReferenceTranscript(
                audio_fixture="simple_tone",
                expected_transcript="",
                expected_word_count=0,
                expected_confidence_min=0.0,
                expected_confidence_avg=0.0,
                expected_segments=[],
                metadata={"type": "synthetic", "has_speech": False},
            ),
            "tone_sequence": ReferenceTranscript(
                audio_fixture="tone_sequence",
                expected_transcript="",
                expected_word_count=0,
                expected_confidence_min=0.0,
                expected_confidence_avg=0.0,
                expected_segments=[],
                metadata={"type": "synthetic", "has_speech": False},
            ),
            "short_audio": ReferenceTranscript(
                audio_fixture="short_audio",
                expected_transcript="",
                expected_word_count=0,
                expected_confidence_min=0.0,
                expected_confidence_avg=0.0,
                expected_segments=[],
                metadata={"type": "edge_case", "duration": "short"},
            ),
            # Example with speech (would be populated with real data)
            "sample_speech": ReferenceTranscript(
                audio_fixture="sample_speech",
                expected_transcript="hello world this is a test",
                expected_word_count=6,
                expected_confidence_min=0.8,
                expected_confidence_avg=0.92,
                expected_segments=[
                    {"start": 0.0, "end": 1.0, "text": "hello", "confidence": 0.95},
                    {"start": 1.0, "end": 2.0, "text": "world", "confidence": 0.93},
                    {"start": 2.0, "end": 3.0, "text": "this", "confidence": 0.90},
                    {"start": 3.0, "end": 4.0, "text": "is", "confidence": 0.88},
                    {"start": 4.0, "end": 5.0, "text": "a", "confidence": 0.85},
                    {"start": 5.0, "end": 6.0, "text": "test", "confidence": 0.91},
                ],
                metadata={"type": "speech", "speaker": "male", "accent": "american"},
            ),
        }

        self.transcripts.update(default_transcripts)
        self._save_transcripts()

    def _create_default_performance_benchmarks(self):
        """Create default performance benchmarks."""
        default_benchmarks = {
            "simple_tone": PerformanceBenchmark(
                audio_fixture="simple_tone",
                max_processing_time_ms=500.0,
                max_real_time_factor=0.1,  # 10x faster than real-time
                max_memory_usage_mb=100.0,
                expected_gpu_memory_mb=50.0,
                baseline_metrics={
                    "preprocessing_ms": 10.0,
                    "asr_ms": 150.0,
                    "postprocessing_ms": 50.0,
                },
            ),
            "tone_sequence": PerformanceBenchmark(
                audio_fixture="tone_sequence",
                max_processing_time_ms=500.0,
                max_real_time_factor=0.15,
                max_memory_usage_mb=100.0,
                expected_gpu_memory_mb=50.0,
                baseline_metrics={
                    "preprocessing_ms": 10.0,
                    "asr_ms": 150.0,
                    "postprocessing_ms": 50.0,
                },
            ),
            "short_audio": PerformanceBenchmark(
                audio_fixture="short_audio",
                max_processing_time_ms=300.0,
                max_real_time_factor=0.5,  # Shorter audio has higher relative overhead
                max_memory_usage_mb=80.0,
                expected_gpu_memory_mb=40.0,
                baseline_metrics={
                    "preprocessing_ms": 5.0,
                    "asr_ms": 100.0,
                    "postprocessing_ms": 30.0,
                },
            ),
            "long_audio": PerformanceBenchmark(
                audio_fixture="long_audio",
                max_processing_time_ms=5000.0,  # 5 seconds for 60s audio
                max_real_time_factor=0.08,  # 12x faster than real-time
                max_memory_usage_mb=200.0,
                expected_gpu_memory_mb=150.0,
                baseline_metrics={
                    "preprocessing_ms": 50.0,
                    "asr_ms": 3000.0,
                    "postprocessing_ms": 500.0,
                },
            ),
        }

        self.performance_benchmarks.update(default_benchmarks)
        self._save_performance_benchmarks()

    def _create_default_quality_benchmarks(self):
        """Create default quality benchmarks."""
        default_quality = {
            "sample_speech": QualityBenchmark(
                audio_fixture="sample_speech",
                min_word_accuracy=0.95,
                min_character_accuracy=0.98,
                min_confidence_score=0.85,
                max_word_error_rate=0.05,
                expected_punctuation_accuracy=0.90,
            ),
            "noisy_audio": QualityBenchmark(
                audio_fixture="noisy_audio",
                min_word_accuracy=0.80,
                min_character_accuracy=0.85,
                min_confidence_score=0.70,
                max_word_error_rate=0.20,
                expected_punctuation_accuracy=0.75,
            ),
        }

        self.quality_benchmarks.update(default_quality)
        self._save_quality_benchmarks()

    def _save_transcripts(self):
        """Save transcripts to file."""
        data = {
            name: asdict(transcript) for name, transcript in self.transcripts.items()
        }
        with open(self.reference_dir / "transcripts.yaml", "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def _save_performance_benchmarks(self):
        """Save performance benchmarks to file."""
        data = {
            name: asdict(benchmark)
            for name, benchmark in self.performance_benchmarks.items()
        }
        with open(self.reference_dir / "performance.yaml", "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def _save_quality_benchmarks(self):
        """Save quality benchmarks to file."""
        data = {
            name: asdict(benchmark)
            for name, benchmark in self.quality_benchmarks.items()
        }
        with open(self.reference_dir / "quality.yaml", "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def get_reference_transcript(self, fixture_name: str) -> ReferenceTranscript | None:
        """Get reference transcript for a fixture."""
        return self.transcripts.get(fixture_name)

    def get_performance_benchmark(
        self, fixture_name: str
    ) -> PerformanceBenchmark | None:
        """Get performance benchmark for a fixture."""
        return self.performance_benchmarks.get(fixture_name)

    def get_quality_benchmark(self, fixture_name: str) -> QualityBenchmark | None:
        """Get quality benchmark for a fixture."""
        return self.quality_benchmarks.get(fixture_name)

    def add_reference_transcript(self, transcript: ReferenceTranscript):
        """Add a new reference transcript."""
        self.transcripts[transcript.audio_fixture] = transcript
        self._save_transcripts()

    def add_performance_benchmark(self, benchmark: PerformanceBenchmark):
        """Add a new performance benchmark."""
        self.performance_benchmarks[benchmark.audio_fixture] = benchmark
        self._save_performance_benchmarks()

    def add_quality_benchmark(self, benchmark: QualityBenchmark):
        """Add a new quality benchmark."""
        self.quality_benchmarks[benchmark.audio_fixture] = benchmark
        self._save_quality_benchmarks()

    def update_benchmark_from_results(
        self,
        fixture_name: str,
        actual_results: dict[str, Any],
        performance_margin: float = 1.2,  # 20% margin for benchmark
    ):
        """Update benchmarks based on actual test results."""
        # Update performance benchmark
        if "duration_ms" in actual_results:
            current_benchmark = self.performance_benchmarks.get(fixture_name)
            if current_benchmark:
                # Update with some margin for performance regression testing
                new_max_time = actual_results["duration_ms"] * performance_margin
                current_benchmark.max_processing_time_ms = min(
                    current_benchmark.max_processing_time_ms,
                    new_max_time,
                )
            else:
                # Create new benchmark
                audio_duration = actual_results.get("audio_duration_seconds", 1.0)
                rtf = actual_results["duration_ms"] / 1000.0 / audio_duration

                self.performance_benchmarks[fixture_name] = PerformanceBenchmark(
                    audio_fixture=fixture_name,
                    max_processing_time_ms=actual_results["duration_ms"]
                    * performance_margin,
                    max_real_time_factor=rtf * performance_margin,
                    max_memory_usage_mb=actual_results.get("memory_usage_mb", 100.0)
                    * performance_margin,
                    expected_gpu_memory_mb=actual_results.get("gpu_memory_mb"),
                    baseline_metrics={},
                )

        self._save_performance_benchmarks()

    def validate_against_benchmark(
        self,
        fixture_name: str,
        actual_results: dict[str, Any],
    ) -> dict[str, bool]:
        """Validate actual results against benchmarks."""
        validation_results = {}

        # Check performance benchmark
        perf_benchmark = self.get_performance_benchmark(fixture_name)
        if perf_benchmark and "duration_ms" in actual_results:
            validation_results["performance_time"] = (
                actual_results["duration_ms"] <= perf_benchmark.max_processing_time_ms
            )

            if "real_time_factor" in actual_results:
                validation_results["performance_rtf"] = (
                    actual_results["real_time_factor"]
                    <= perf_benchmark.max_real_time_factor
                )

        # Check quality benchmark
        quality_benchmark = self.get_quality_benchmark(fixture_name)
        if quality_benchmark:
            if "confidence_avg" in actual_results:
                validation_results["quality_confidence"] = (
                    actual_results["confidence_avg"]
                    >= quality_benchmark.min_confidence_score
                )

            if "word_accuracy" in actual_results:
                validation_results["quality_accuracy"] = (
                    actual_results["word_accuracy"]
                    >= quality_benchmark.min_word_accuracy
                )

        return validation_results


def load_reference_transcripts() -> dict[str, str]:
    """Load reference transcripts from default location."""
    manager = ReferenceDataManager()
    return {
        name: transcript.expected_transcript
        for name, transcript in manager.transcripts.items()
    }
