#!/usr/bin/env python3
"""
Test Unicode normalization for Vietnamese character handling.
"""

import unicodedata
import re

def test_unicode_normalization():
    """Test Unicode normalization for Vietnamese characters."""
    
    text = "Tiêu đề: Tích phân đường loại một - Phương pháp tính và ví dụ minh họa\nTóm tắt: Bài giảng hướng dẫn cách tính tích phân đường loại một, ứng dụng trong bài toán tính khối lượng dây không đồng chất. Học sinh cần xác định loại đường cong (theo \\( t \\), \\( x \\), hoặc \\( y \\)), chọn biến tham số phù hợp, và áp dụng công thức tính \\( ds \\) tương ứng. Các ví dụ minh họa tính tích phân trên đường tròn, parabol, và đoạn thẳng.\nĐiểm chính:"
    
    print("=== Unicode Normalization Testing ===")
    
    # Test different normalization forms
    forms = ['NFC', 'NFD', 'NFKC', 'NFKD']  # type: ignore
    
    for form in forms:
        normalized_text = unicodedata.normalize(form, text)
        print(f"\nNormalization form: {form}")
        
        # Test if patterns match after normalization
        patterns = [
            r"(?i)tiêu đề:",
            r"(?i)tóm tắt:",
            r"(?i)điểm chính",
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, normalized_text, flags=re.MULTILINE)
            print(f"  Pattern '{pattern}': {bool(matches)} - {matches[:2] if matches else 'None'}")
    
    # Test a more robust approach using case-insensitive matching with normalized text
    print("\n=== Robust Approach Testing ===")
    
    # Normalize and convert to lowercase for case-insensitive matching
    normalized_lower = unicodedata.normalize('NFC', text).lower()
    
    # Define patterns that work with normalized text
    robust_patterns = {
        'title': r'tiêu đề:\s*(.+?)(?=\n|$)',
        'abstract': r'tóm tắt:\s*(.+?)(?=\n\s*điểm chính|\n\s*điểm chính:|\n-|$)',
        'points_label': r'điểm chính:\s*(.*)',
    }
    
    for key, pattern in robust_patterns.items():
        match = re.search(pattern, normalized_lower, flags=re.MULTILINE | re.DOTALL)
        if match:
            content = match.group(1).strip()
            print(f"  {key}: '{content[:50]}...'")
        else:
            print(f"  {key}: Not found")
    
    # Test extracting all content
    print("\n=== Full Extraction Test ===")
    
    # Extract title
    title_match = re.search(r'tiêu đề:\s*(.+?)(?=\n|$)', normalized_lower, flags=re.MULTILINE | re.DOTALL)
    title = title_match.group(1).strip() if title_match else ""
    print(f"Title: '{title}'")
    
    # Extract abstract
    abstract_match = re.search(r'tóm tắt:\s*(.+?)(?=\n\s*điểm chính|\n\s*điểm chính:|\n-|$)', normalized_lower, flags=re.MULTILINE | re.DOTALL)
    abstract = abstract_match.group(1).strip() if abstract_match else ""
    print(f"Abstract: '{abstract[:100]}...'")
    
    # Extract points
    points_match = re.search(r'điểm chính:\s*(.*)', normalized_lower, flags=re.MULTILINE | re.DOTALL)
    if points_match:
        points_content = points_match.group(1).strip()
        points = [p.strip() for p in points_content.split('\n') if p.strip() and not p.strip().startswith('-')]
        print(f"Points: {points[:3]}")

if __name__ == "__main__":
    test_unicode_normalization()