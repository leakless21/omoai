#!/usr/bin/env python3
"""
Test the regex fix for Vietnamese labels.
"""

import re


def test_corrected_patterns():
    """Test the corrected regex patterns."""

    text = "Tiêu đề: Tích phân đường loại một - Phương pháp tính và ví dụ minh họa\nTóm tắt: Bài giảng hướng dẫn cách tính tích phân đường loại một, ứng dụng trong bài toán tính khối lượng dây không đồng chất. Học sinh cần xác định loại đường cong (theo \\( t \\), \\( x \\), hoặc \\( y \\)), chọn biến tham số phù hợp, và áp dụng công thức tính \\( ds \\) tương ứng. Các ví dụ minh họa tính tích phân trên đường tròn, parabol, và đoạn thẳng.\nĐiểm chính:"

    print("=== Testing Corrected Patterns ===")

    # Original problematic pattern
    original_pattern = r"(?i)T[oó]m\s*t[aá]t|Đi[eê]m\s*ch[ií]nh|^\-"
    matches = re.findall(original_pattern, text, flags=re.MULTILINE)
    print(f"Original pattern: {original_pattern}")
    print(f"  Matches: {matches}")
    print(f"  Any matches: {bool(matches)}")
    print()

    # Corrected patterns
    corrected_patterns = [
        r"(?i)T[oó]m\s*T[aá]t",  # Fixed case sensitivity
        r"(?i)T[oó]m\s*t[aá]t",  # Original with fixed case
        r"(?i)Đi[eê]m\s*ch[ií]nh",  # Original should work
        r"(?i)T[oó]m\s*t[aá]t|Đi[eê]m\s*ch[ií]nh|^\-",  # Combined corrected
    ]

    for i, pattern in enumerate(corrected_patterns):
        matches = re.findall(pattern, text, flags=re.MULTILINE)
        print(f"Corrected pattern {i + 1}: {pattern}")
        print(f"  Matches: {matches}")
        print(f"  Any matches: {bool(matches)}")
        print()

    # Test individual components
    print("=== Testing Individual Components ===")

    # Test title extraction
    title_pattern = r"(?i)Ti[eê]u\s*đ[eề]\s*:\s*(.+)"
    m_title = re.search(title_pattern, text, flags=re.IGNORECASE)
    if m_title:
        extracted_title = m_title.group(1).splitlines()[0].strip()
        print(f"Title extraction: '{extracted_title}'")

    # Test abstract extraction with corrected pattern
    abstract_pattern = r"(?i)T[oó]m\s*t[aá]t\s*:\s*([\s\S]*?)(?:\n\s*Đi[eê]m\s*ch[ií]nh\s*:|\n\s*Đi[eê]m\s*ch[ií]nh|\n-|\Z)"
    m_abstract = re.search(abstract_pattern, text, flags=re.IGNORECASE)
    if m_abstract:
        extracted_abstract = m_abstract.group(1).strip()
        print(f"Abstract extraction: '{extracted_abstract[:50]}...'")
    else:
        print("No abstract found")


if __name__ == "__main__":
    test_corrected_patterns()
