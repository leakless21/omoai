#!/usr/bin/env python3
"""
Unit tests for Vietnamese text parsing in postprocess.py
"""

import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from omoai.pipeline.postprocess import _parse_vietnamese_labeled_text


class TestVietnameseParsing:
    """Test cases for Vietnamese text parsing function."""
    
    def test_complete_vietnamese_labeled_text(self):
        """Test parsing of complete Vietnamese-labeled text (the problematic case)."""
        text = """Tiêu đề: Giải thích và phương pháp tính tích phân đường loại một
Tóm tắt: Bài giảng trình bày cách tính tích phân đường loại một, ứng dụng trong bài toán tính khối lượng dây không đồng chất. Học sinh cần xác định loại đường cong (theo \\( t \\), \\( x \\), hoặc \\( y \\)), chọn biến tham số phù hợp, và áp dụng công thức tính \\( ds \\) tương ứng. Các ví dụ minh họa tính tích phân trên đường tròn, parabol, và đoạn thẳng.
Điểm chính:
- Cách xác định loại đường cong và chọn biến tham số
- Công thức tính \\( ds \\) cho từng loại đường cong
- Các ví dụ minh họa với đường tròn, parabol, đoạn thẳng"""
        
        result = _parse_vietnamese_labeled_text(text)
        
        assert result is not None, "Should successfully parse Vietnamese-labeled text"
        assert result["title"] == "Giải thích và phương pháp tính tích phân đường loại một"
        assert "Bài giảng trình bày cách tính tích phân" in result["abstract"]
        assert len(result["points"]) == 3
        assert "Cách xác định loại đường cong" in result["points"][0]
    
    def test_english_labeled_text(self):
        """Test parsing of English-labeled text."""
        text = """Title: Line Integral Type One - Calculation Methods and Examples
Summary: This lecture explains how to calculate line integrals of type one, with applications in calculating the mass of non-uniform wires.
Main Points:
- How to determine curve type and choose parameters
- \\( ds \\) formulas for each curve type
- Examples with circles, parabolas, line segments"""
        
        result = _parse_vietnamese_labeled_text(text)
        
        assert result is not None, "Should successfully parse English-labeled text"
        assert result["title"] == "Line Integral Type One - Calculation Methods and Examples"
        assert "This lecture explains how to calculate" in result["abstract"]
        assert len(result["points"]) == 3
    
    def test_mixed_vietnamese_english(self):
        """Test parsing of mixed Vietnamese and English labels."""
        text = """Tiêu đề: Hướng dẫn giải bài toán tích phân
Summary: Hướng dẫn chi tiết cách giải các bài toán tích phân đường loại một.
Điểm chính:
- Các bước giải bài toán
- Các công thức cần nhớ
- Ví dụ minh họa"""
        
        result = _parse_vietnamese_labeled_text(text)
        
        assert result is not None, "Should successfully parse mixed text"
        assert result["title"] == "Hướng dẫn giải bài toán tích phân"
        assert "Hướng dẫn chi tiết cách giải" in result["abstract"]
        assert len(result["points"]) == 3
    
    def test_text_without_labels(self):
        """Test that text without labels returns None."""
        text = "This is just a regular text without any labels. It should not be parsed as structured data."
        
        result = _parse_vietnamese_labeled_text(text)
        
        assert result is None, "Should return None for text without labels"
    
    def test_empty_text(self):
        """Test that empty text returns None."""
        result = _parse_vietnamese_labeled_text("")
        
        assert result is None, "Should return None for empty text"
    
    def test_only_title(self):
        """Test parsing of text with only title."""
        text = "Tiêu đề: Just a title"
        
        result = _parse_vietnamese_labeled_text(text)
        
        assert result is not None, "Should parse title-only text"
        assert result["title"] == "Just a title"
        assert result["abstract"] == ""
        assert result["points"] == []
    
    def test_unicode_normalization(self):
        """Test that Unicode normalization works correctly."""
        # Text with composed and decomposed characters
        text = "Tiêu đề: Test with unicode\nTóm tắt: Abstract here"
        
        result = _parse_vietnamese_labeled_text(text)
        
        assert result is not None, "Should handle Unicode normalization"
        assert result["title"] == "Test with unicode"
        assert result["abstract"] == "Abstract here"
    
    def test_case_insensitive_matching(self):
        """Test that pattern matching is case insensitive."""
        text = """TIÊU ĐỀ: Upper Case Title
TÓM TẮT: Upper Case Abstract
ĐIỂM CHÍNH:
- Point 1
- Point 2"""
        
        result = _parse_vietnamese_labeled_text(text)
        
        assert result is not None, "Should handle case insensitive matching"
        assert result["title"] == "Upper Case Title"
        assert result["abstract"] == "Upper Case Abstract"
        assert len(result["points"]) == 2
    
    def test_whitespace_handling(self):
        """Test that various whitespace patterns are handled correctly."""
        text = """Tiêu đề:    Title with spaces    
Tóm tắt:    Abstract with spaces    
Điểm chính:   
-   Point with spaces   
-   Another point   """
        
        result = _parse_vietnamese_labeled_text(text)
        
        assert result is not None, "Should handle whitespace correctly"
        assert result["title"] == "Title with spaces"
        assert result["abstract"] == "Abstract with spaces"
        assert len(result["points"]) == 2
        assert result["points"][0] == "Point with spaces"
        assert result["points"][1] == "Another point"


if __name__ == "__main__":
    # Run tests manually if pytest is not available
    test_instance = TestVietnameseParsing()
    
    print("Running Vietnamese text parsing tests...")
    
    tests = [
        "test_complete_vietnamese_labeled_text",
        "test_english_labeled_text", 
        "test_mixed_vietnamese_english",
        "test_text_without_labels",
        "test_empty_text",
        "test_only_title",
        "test_unicode_normalization",
        "test_case_insensitive_matching",
        "test_whitespace_handling"
    ]
    
    passed = 0
    failed = 0
    
    for test_name in tests:
        try:
            print(f"Running {test_name}...")
            getattr(test_instance, test_name)()
            print(f"✓ {test_name} PASSED")
            passed += 1
        except Exception as e:
            print(f"✗ {test_name} FAILED: {e}")
            failed += 1
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")
    
    if failed == 0:
        print("🎉 All tests passed! The Vietnamese parsing fix is working correctly.")
    else:
        print("❌ Some tests failed. Please review the implementation.")