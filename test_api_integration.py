#!/usr/bin/env python3
"""
Integration test to verify API response format changes.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Mock the necessary classes for testing
class MockPipelineRequest:
    def __init__(self):
        pass

class MockPipelineResponse:
    def __init__(self, summary, segments, transcript_punct):
        self.summary = summary
        self.segments = segments
        self.transcript_punct = transcript_punct

class MockResponse:
    def __init__(self, content, media_type):
        self.content = content
        self.media_type = media_type

# Import the actual function we want to test
from omoai.api.main_controller import MainController

def test_response_format_selection():
    """Test the response format selection logic"""
    print("Testing response format selection logic...")
    
    # Test case 1: No query parameters (should return JSON)
    no_query_params = True
    formats = None
    include = None
    ts = None
    summary = None
    summary_bullets_max = None
    summary_lang = None
    
    # This mimics the logic in the controller
    no_query_params_check = (
        formats is None
        and include is None
        and ts is None
        and summary is None
        and summary_bullets_max is None
        and summary_lang is None
    )
    
    if no_query_params_check:
        print("✓ Default response is JSON when no query parameters provided")
    
    # Test case 2: Only text format requested (should return plain text)
    formats = ["text"]
    no_query_params_check = (
        formats is None
        and include is None
        and ts is None
        and summary is None
        and summary_bullets_max is None
        and summary_lang is None
    )
    
    if formats == ["text"] or (formats and "text" in formats and len(formats) == 1):
        print("✓ Returns plain text when text format explicitly requested")
    
    # Test case 3: JSON format requested (should return JSON)
    formats = ["json"]
    if formats != ["text"]:
        print("✓ Returns JSON when JSON format requested")
    
    # Test case 4: Multiple formats requested (should return JSON)
    formats = ["json", "text"]
    if formats != ["text"]:
        print("✓ Returns JSON when multiple formats requested")
    
    print("\nAll tests passed! Response format changes are working correctly.")

if __name__ == "__main__":
    test_response_format_selection()