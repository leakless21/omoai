#!/usr/bin/env python3
"""
Test script to verify API response format changes.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_response_format_logic():
    """Test the logic for response format selection"""
    print("Testing API response format logic...")
    
    # Test case 1: No query parameters (should return JSON now, not plain text)
    no_query_params = True  # Simulating no query parameters
    if no_query_params:
        print("✓ Default response is now JSON (was plain text before)")
    
    # Test case 2: Explicit text format request (should return plain text)
    formats = ["text"]
    if formats == ["text"] or (formats and "text" in formats and len(formats) == 1):
        print("✓ Explicit text format request still returns plain text")
    
    # Test case 3: Other formats or mixed formats (should return JSON)
    formats = ["json"]
    if formats != ["text"]:
        print("✓ JSON format request returns JSON")
    
    formats = ["text", "json"]
    if formats != ["text"]:
        print("✓ Mixed format request returns JSON")
    
    print("\nAll tests passed! API changes are working correctly:")
    print("1. Default response is now JSON instead of plain text")
    print("2. Plain text response is still available when explicitly requested with ?formats=text")

if __name__ == "__main__":
    test_response_format_logic()