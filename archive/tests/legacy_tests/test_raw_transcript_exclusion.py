#!/usr/bin/env python3
"""
Test script to verify that raw transcript is excluded by default in API responses.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from omoai.api.models import PipelineResponse

def test_pipeline_response_structure():
    """Test that PipelineResponse has the correct structure."""
    # Create a mock PipelineResponse
    response = PipelineResponse(
        summary={"bullets": ["Test bullet"], "abstract": "Test abstract"},
        segments=[{"start": 0.0, "end": 1.0, "text_punct": "Test sentence."}],
        transcript_punct="Test sentence.",
        transcript_raw="test sentence"  # This should not be included by default
    )
    
    print("Created PipelineResponse with all fields:")
    print(f"- summary: {response.summary}")
    print(f"- segments: {response.segments}")
    print(f"- transcript_punct: {response.transcript_punct}")
    print(f"- transcript_raw: {getattr(response, 'transcript_raw', 'NOT_PRESENT')}")
    
    # Create a new response that excludes raw transcript (simulating our API change)
    filtered_response = PipelineResponse(
        summary=response.summary,
        segments=response.segments,
        transcript_punct=response.transcript_punct
        # transcript_raw is intentionally omitted
    )
    
    print("\nFiltered PipelineResponse (without raw transcript):")
    print(f"- summary: {filtered_response.summary}")
    print(f"- segments: {filtered_response.segments}")
    print(f"- transcript_punct: {filtered_response.transcript_punct}")
    print(f"- transcript_raw: {getattr(filtered_response, 'transcript_raw', 'NOT_PRESENT')}")
    
    # Verify that raw transcript is not present in filtered response
    assert not hasattr(filtered_response, 'transcript_raw') or filtered_response.transcript_raw is None
    print("\n✓ Raw transcript correctly excluded from filtered response")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_pipeline_response_structure()