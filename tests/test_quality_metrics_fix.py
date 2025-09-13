#!/usr/bin/env python3
"""
Unit test to verify the quality_metrics and diffs fix works correctly
and prevent regression of the bug where these fields were missing from API responses.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add src to path so we can import our modules
src_path = Path(__file__).parent.parent / "src"

sys.path.insert(0, str(src_path))

from omoai.api.models import HumanReadableDiff, PipelineResponse, QualityMetrics


class TestQualityMetricsFix:
    """Test suite for the quality_metrics and diffs fix"""

    def create_mock_pipeline_result_with_quality_metrics(self):
        """Create a mock pipeline result with quality_metrics and diffs"""
        result = MagicMock()
        result.transcript = "This is a test transcript"
        result.transcript_punct = "This is a test transcript."
        result.summary = {
            "bullets": ["Test point 1", "Test point 2"],
            "abstract": "This is a test summary",
        }
        result.segments = [
            {
                "start": 0.0,
                "end": 1.0,
                "text": "This is a test transcript",
            },
        ]
        result.quality_metrics = QualityMetrics(
            wer=0.05,
            cer=0.03,
            per=0.02,
            uwer=0.04,
            fwer=0.06,
            alignment_confidence=0.95,
        )
        result.diffs = HumanReadableDiff(
            original_text="bad sentence",
            punctuated_text="Good sentence.",
            diff_output="bad -> Good",
            alignment_summary="Punctuation improved",
        )
        result.metadata = {
            "duration_seconds": 1.0,
            "sample_rate": 16000,
            "processing_time": 0.5,
        }
        return result

    def create_mock_pipeline_result_without_quality_metrics(self):
        """Create a mock pipeline result without quality_metrics and diffs"""
        result = MagicMock()
        result.transcript = "This is a test transcript"
        result.transcript_punct = "This is a test transcript."
        result.summary = {
            "bullets": ["Test point 1", "Test point 2"],
            "abstract": "This is a test summary",
        }
        result.segments = [
            {
                "start": 0.0,
                "end": 1.0,
                "text": "This is a test transcript",
            },
        ]
        result.quality_metrics = None
        result.diffs = None
        result.metadata = {
            "duration_seconds": 1.0,
            "sample_rate": 16000,
            "processing_time": 0.5,
        }
        return result

    def test_quality_metrics_included_when_requested_and_available(self):
        """Test that quality_metrics and diffs are included when requested and available"""
        # Arrange
        mock_result = self.create_mock_pipeline_result_with_quality_metrics()

        # Act - Simulate the response construction with quality metrics and diffs requested
        response = PipelineResponse(
            summary=mock_result.summary,
            segments=mock_result.segments,
            transcript_punct=mock_result.transcript_punct,
            quality_metrics=mock_result.quality_metrics,
            diffs=mock_result.diffs,
        )

        # Assert
        assert response.quality_metrics is not None, (
            "Quality metrics should not be None when requested and available"
        )
        assert response.diffs is not None, (
            "Diffs should not be None when requested and available"
        )
        assert response.quality_metrics.wer == 0.05, "WER should be 0.05"
        assert response.quality_metrics.cer == 0.03, "CER should be 0.03"
        assert response.diffs.original_text == "bad sentence", (
            "Original text should match"
        )
        assert response.diffs.punctuated_text == "Good sentence.", (
            "Punctuated text should match"
        )

    def test_quality_metrics_excluded_when_not_requested(self):
        """Test that quality_metrics and diffs are excluded when not requested"""
        # Arrange
        mock_result = self.create_mock_pipeline_result_with_quality_metrics()

        # Act - Simulate the response construction without quality metrics and diffs requested
        response = PipelineResponse(
            summary=mock_result.summary,
            segments=mock_result.segments,
            transcript_punct=mock_result.transcript_punct,
            quality_metrics=None,
            diffs=None,
        )

        # Assert
        assert response.quality_metrics is None, (
            "Quality metrics should be None when not requested"
        )
        assert response.diffs is None, "Diffs should be None when not requested"

    def test_quality_metrics_none_when_not_available_in_data(self):
        """Test that quality_metrics and diffs are None when requested but not available in data"""
        # Arrange
        mock_result = self.create_mock_pipeline_result_without_quality_metrics()

        # Act - Simulate the response construction with quality metrics and diffs requested but no data
        response = PipelineResponse(
            summary=mock_result.summary,
            segments=mock_result.segments,
            transcript_punct=mock_result.transcript_punct,
            quality_metrics=mock_result.quality_metrics,
            diffs=mock_result.diffs,
        )

        # Assert
        assert response.quality_metrics is None, (
            "Quality metrics should be None when not available in data"
        )
        assert response.diffs is None, "Diffs should be None when not available in data"

    def test_quality_metrics_partial_data(self):
        """Test that partial quality metrics data is handled correctly"""
        # Arrange
        mock_result = MagicMock()
        mock_result.transcript = "This is a test transcript"
        mock_result.transcript_punct = "This is a test transcript."
        mock_result.summary = {"bullets": ["Test point 1"], "abstract": "Test summary"}
        mock_result.segments = [
            {"start": 0.0, "end": 1.0, "text": "This is a test transcript"}
        ]

        # Create partial quality metrics (only some fields)
        mock_result.quality_metrics = QualityMetrics(
            wer=0.05,
            cer=None,  # Missing CER
            per=None,  # Missing PER
            uwer=None,  # Missing UWER
            fwer=None,  # Missing FWER
            alignment_confidence=0.95,
        )
        mock_result.diffs = None  # No diffs

        # Act
        response = PipelineResponse(
            summary=mock_result.summary,
            segments=mock_result.segments,
            transcript_punct=mock_result.transcript_punct,
            quality_metrics=mock_result.quality_metrics,
            diffs=mock_result.diffs,
        )

        # Assert
        assert response.quality_metrics is not None, (
            "Quality metrics should not be None"
        )
        assert response.quality_metrics.wer == 0.05, "WER should be 0.05"
        assert response.quality_metrics.cer is None, "CER should be None"
        assert response.quality_metrics.alignment_confidence == 0.95, (
            "Alignment confidence should be 0.95"
        )
        assert response.diffs is None, "Diffs should be None"

    def test_diffs_partial_data(self):
        """Test that partial diffs data is handled correctly"""
        # Arrange
        mock_result = MagicMock()
        mock_result.transcript = "This is a test transcript"
        mock_result.transcript_punct = "This is a test transcript."
        mock_result.summary = {"bullets": ["Test point 1"], "abstract": "Test summary"}
        mock_result.segments = [
            {"start": 0.0, "end": 1.0, "text": "This is a test transcript"}
        ]

        # Create partial diffs (only some fields)
        mock_result.quality_metrics = None  # No quality metrics
        mock_result.diffs = HumanReadableDiff(
            original_text="bad sentence",
            punctuated_text="Good sentence.",
            diff_output=None,  # Missing diff output
            alignment_summary="Punctuation improved",
        )

        # Act
        response = PipelineResponse(
            summary=mock_result.summary,
            segments=mock_result.segments,
            transcript_punct=mock_result.transcript_punct,
            quality_metrics=mock_result.quality_metrics,
            diffs=mock_result.diffs,
        )

        # Assert
        assert response.diffs is not None, "Diffs should not be None"
        assert response.diffs.original_text == "bad sentence", (
            "Original text should match"
        )
        assert response.diffs.punctuated_text == "Good sentence.", (
            "Punctuated text should match"
        )
        assert response.diffs.diff_output is None, "Diff output should be None"
        assert response.diffs.alignment_summary == "Punctuation improved", (
            "Alignment summary should match"
        )
        assert response.quality_metrics is None, "Quality metrics should be None"


if __name__ == "__main__":
    pytest.main([__file__])
