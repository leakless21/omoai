#!/usr/bin/env python3
"""
Test the complete fix for the summary parsing issue.
"""

from src.omoai.api.services import _normalize_summary

def test_fix_validation():
    """Test the complete fix with the actual API flow."""
    
    print("=== Testing Complete Fix ===")
    
    # Test case 1: Dict with Vietnamese labels in summary field
    test_case_1 = {
        "title": "",
        "summary": "Tiêu đề: Tích phân đường loại một - Phương pháp tính và ví dụ minh họa\nTóm tắt: Bài giảng hướng dẫn cách tính tích phân đường loại một, ứng dụng trong bài toán tính khối lượng dây không đồng chất. Học sinh cần xác định loại đường cong (theo \\( t \\), \\( x \\), hoặc \\( y \\)), chọn biến tham số phù hợp, và áp dụng công thức tính \\( ds \\) tương ứng. Các ví dụ minh họa tính tích phân trên đường tròn, parabol, và đoạn thẳng.\nĐiểm chính:",
        "points": ["Point 1", "Point 2", "Point 3"]
    }
    
    print("Test Case 1: Dict with Vietnamese labels")
    result_1 = _normalize_summary(test_case_1)
    print(f"  Input title: '{test_case_1['title']}'")
    print(f"  Input summary: '{test_case_1['summary'][:50]}...'")
    print(f"  Result title: '{result_1['title']}'")
    print(f"  Result summary: '{result_1['summary'][:50]}...'")
    print(f"  Result points: {result_1['points']}")
    print(f"  Title extracted: {bool(result_1['title'])}")
    print(f"  Labels stripped: {not ('Tiêu đề:' in result_1['summary'] or 'Tóm tắt:' in result_1['summary'])}")
    print()
    
    # Test case 2: Dict with proper structure (should not need parsing)
    test_case_2 = {
        "title": "Tích phân đường loại một",
        "summary": "Bài giảng hướng dẫn cách tính tích phân đường loại một, ứng dụng trong bài toán tính khối lượng dây không đồng chất.",
        "points": ["Point 1", "Point 2", "Point 3"]
    }
    
    print("Test Case 2: Dict with proper structure")
    result_2 = _normalize_summary(test_case_2)
    print(f"  Input title: '{test_case_2['title']}'")
    print(f"  Input summary: '{test_case_2['summary'][:50]}...'")
    print(f"  Result title: '{result_2['title']}'")
    print(f"  Result summary: '{result_2['summary'][:50]}...'")
    print(f"  Result points: {result_2['points']}")
    print()
    
    # Test case 3: String input (direct text parsing)
    test_case_3 = """Tiêu đề: Tích phân đường loại một - Phương pháp tính và ví dụ minh họa
Tóm tắt: Bài giảng hướng dẫn cách tính tích phân đường loại một, ứng dụng trong bài toán tính khối lượng dây không đồng chất. Học sinh cần xác định loại đường cong (theo \\( t \\), \\( x \\), hoặc \\( y \\)), chọn biến tham số phù hợp, và áp dụng công thức tính \\( ds \\) tương ứng. Các ví dụ minh họa tính tích phân trên đường tròn, parabol, và đoạn thẳng.
Điểm chính:
- Điểm 1
- Điểm 2
- Điểm 3"""
    
    print("Test Case 3: String input with Vietnamese labels")
    result_3 = _normalize_summary(test_case_3)
    print(f"  Input: '{test_case_3[:50]}...'")
    print(f"  Result title: '{result_3['title']}'")
    print(f"  Result summary: '{result_3['summary'][:50]}...'")
    print(f"  Result points: {result_3['points']}")
    print(f"  Title extracted: {bool(result_3['title'])}")
    print(f"  Labels stripped: {not ('Tiêu đề:' in result_3['summary'] or 'Tóm tắt:' in result_3['summary'])}")
    
    # Overall validation
    print("\n=== Overall Validation ===")
    all_tests_pass = all([
        bool(result_1['title']),  # Title should be extracted
        not ('Tiêu đề:' in result_1['summary'] or 'Tóm tắt:' in result_1['summary']),  # Labels should be stripped
        result_2['title'] == "Tích phân đường loại một",  # Proper structure should be preserved
        result_2['summary'] == "Bài giảng hướng dẫn cách tính tích phân đường loại một, ứng dụng trong bài toán tính khối lượng dây không đồng chất.",
        bool(result_3['title']),  # String parsing should extract title
        not ('Tiêu đề:' in result_3['summary'] or 'Tóm tắt:' in result_3['summary']),  # String parsing should strip labels
        len(result_3['points']) > 0,  # String parsing should extract points
    ])
    
    print(f"All tests pass: {all_tests_pass}")
    
    if not all_tests_pass:
        print("❌ Some tests failed!")
        if not bool(result_1['title']):
            print("  - Title extraction from dict with labels failed")
        if 'Tiêu đề:' in result_1['summary'] or 'Tóm tắt:' in result_1['summary']:
            print("  - Label stripping from dict with labels failed")
        if result_2['title'] != "Tích phân đường loại một":
            print("  - Proper structure preservation failed")
        if not bool(result_3['title']):
            print("  - Title extraction from string failed")
        if 'Tiêu đề:' in result_3['summary'] or 'Tóm tắt:' in result_3['summary']:
            print("  - Label stripping from string failed")
        if len(result_3['points']) == 0:
            print("  - Points extraction from string failed")
    else:
        print("✅ All tests passed! The fix is working correctly.")

if __name__ == "__main__":
    test_fix_validation()