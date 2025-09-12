#!/usr/bin/env python3
"""
Test script to reproduce and debug the summary parsing issue.
"""

import re
from omoai.api.services import _normalize_summary

# Test case based on the problematic output from the issue
problematic_summary = {
    "title": "",
    "summary": "Tiêu đề: Tích phân đường loại một - Phương pháp tính và ví dụ minh họa\nTóm tắt: Bài giảng hướng dẫn cách tính tích phân đường loại một, ứng dụng trong bài toán tính khối lượng dây không đồng chất. Học sinh cần xác định loại đường cong (theo \\( t \\), \\( x \\), hoặc \\( y \\)), chọn biến tham số phù hợp, và áp dụng công thức tính \\( ds \\) tương ứng. Các ví dụ minh họa tính tích phân trên đường tròn, parabol, và đoạn thẳng.\nĐiểm chính:",
    "points": ["Point 1", "Point 2", "Point 3"]
}

def test_current_parsing():
    """Test the current parsing behavior."""
    print("=== Testing Current Parsing Behavior ===")
    
    result = _normalize_summary(problematic_summary)
    
    print("Input summary:")
    print(f"  title: '{problematic_summary['title']}'")
    print(f"  summary: '{problematic_summary['summary'][:100]}...'")
    print(f"  points: {problematic_summary['points']}")
    
    print("\nParsed result:")
    print(f"  title: '{result['title']}'")
    print(f"  summary: '{result['summary'][:100]}...'")
    print(f"  points: {result['points']}")
    
    return result

def test_individual_regex_patterns():
    """Test individual regex patterns used in the parsing."""
    print("\n=== Testing Individual Regex Patterns ===")
    
    text = problematic_summary['summary']
    
    # Test title pattern
    title_pattern = r"Ti[eê]u\s*đ[eề]\s*:\s*(.+)"
    m_title = re.search(title_pattern, text, flags=re.IGNORECASE)
    print(f"Title pattern '{title_pattern}':")
    print(f"  Match: {m_title.group(1) if m_title else 'None'}")
    
    # Test alternative title pattern
    title_pattern2 = r"Title\s*:\s*(.+)"
    m_title2 = re.search(title_pattern2, text, flags=re.IGNORECASE)
    print(f"Title pattern 2 '{title_pattern2}':")
    print(f"  Match: {m_title2.group(1) if m_title2 else 'None'}")
    
    # Test abstract pattern
    abstract_pattern = r"T[oó]m\s*t[aá]t\s*:\s*([\s\S]*?)(?:\n\s*Đi[eê]m\s*ch[ií]nh\s*:|\n\s*Đi[eê]m\s*ch[ií]nh|\n-|\Z)"
    m_abstract = re.search(abstract_pattern, text, flags=re.IGNORECASE)
    print(f"Abstract pattern '{abstract_pattern}':")
    print(f"  Match: {m_abstract.group(1) if m_abstract else 'None'}")
    
    # Test points pattern
    points_pattern = r"Đi[eê]m\s*ch[ií]nh\s*:\s*([\s\S]*)"
    m_points = re.search(points_pattern, text, flags=re.IGNORECASE)
    print(f"Points pattern '{points_pattern}':")
    print(f"  Match: {m_points.group(1) if m_points else 'None'}")

if __name__ == "__main__":
    result = test_current_parsing()
    test_individual_regex_patterns()
    
    print(f"\n=== Analysis ===")
    print(f"Title is empty: {result['title'] == ''}")
    print(f"Summary still contains labels: {'Tiêu đề:' in result['summary'] or 'Tóm tắt:' in result['summary']}")
