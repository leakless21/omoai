#!/usr/bin/env python3
"""
Debug the regex pattern matching issue.
"""

import re

def test_regex_patterns():
    """Test the regex patterns individually."""
    
    text = "Tiêu đề: Tích phân đường loại một - Phương pháp tính và ví dụ minh họa\nTóm tắt: Bài giảng hướng dẫn cách tính tích phân đường loại một, ứng dụng trong bài toán tính khối lượng dây không đồng chất. Học sinh cần xác định loại đường cong (theo \\( t \\), \\( x \\), hoặc \\( y \\)), chọn biến tham số phù hợp, và áp dụng công thức tính \\( ds \\) tương ứng. Các ví dụ minh họa tính tích phân trên đường tròn, parabol, và đoạn thẳng.\nĐiểm chính:"
    
    print("=== Testing Individual Regex Components ===")
    
    # Test each component of the pattern
    patterns = [
        r"(?i)T[oó]m\s*t[aá]t",  # Tóm tắt variations
        r"(?i)Đi[eê]m\s*ch[ií]nh",  # Điểm chính variations
        r"^\-",  # Lines starting with -
    ]
    
    for i, pattern in enumerate(patterns):
        matches = re.findall(pattern, text, flags=re.MULTILINE)
        print(f"Pattern {i+1}: {pattern}")
        print(f"  Matches: {matches}")
        print(f"  Any matches: {bool(matches)}")
        print()
    
    # Test the combined pattern
    combined_pattern = r"(?i)T[oó]m\s*t[aá]t|Đi[eê]m\s*ch[ií]nh|^\-"
    matches = re.findall(combined_pattern, text, flags=re.MULTILINE)
    print(f"Combined pattern: {combined_pattern}")
    print(f"  Matches: {matches}")
    print(f"  Any matches: {bool(matches)}")
    print()
    
    # Test simpler patterns
    simple_patterns = [
        r"Tóm tắt",
        r"Tôm tát",
        r"(?i)tóm tắt",
        r"(?i)tôm tát",
        r"(?i)t[oó]m t[aá]t",
    ]
    
    print("=== Testing Simpler Patterns ===")
    for pattern in simple_patterns:
        matches = re.findall(pattern, text, flags=re.MULTILINE)
        print(f"Pattern: {pattern}")
        print(f"  Matches: {matches}")
        print(f"  Any matches: {bool(matches)}")
        print()

if __name__ == "__main__":
    test_regex_patterns()
