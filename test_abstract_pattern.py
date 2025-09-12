#!/usr/bin/env python3
"""
Debug the abstract pattern specifically.
"""

import re

def debug_abstract_pattern():
    """Debug the abstract pattern step by step."""
    
    text = "Tiêu đề: Tích phân đường loại một - Phương pháp tính và ví dụ minh họa\nTóm tắt: Bài giảng hướng dẫn cách tính tích phân đường loại một, ứng dụng trong bài toán tính khối lượng dây không đồng chất. Học sinh cần xác định loại đường cong (theo \\( t \\), \\( x \\), hoặc \\( y \\)), chọn biến tham số phù hợp, và áp dụng công thức tính \\( ds \\) tương ứng. Các ví dụ minh họa tính tích phân trên đường tròn, parabol, và đoạn thẳng.\nĐiểm chính:"
    
    print("=== Debugging Abstract Pattern ===")
    print(f"Text: {text[:100]}...")
    print()
    
    # Test the current pattern
    current_pattern = r"T[oó]m\s*t[aá]t\s*:\s*([\s\S]*?)(?:\n\s*Đi[eê]m\s*ch[ií]nh\s*:|\n\s*Đi[eê]m\s*ch[ií]nh|\n-|\Z)"
    
    print(f"Current pattern: {current_pattern}")
    m_abstract = re.search(current_pattern, text, flags=re.IGNORECASE)
    if m_abstract:
        extracted_abstract = m_abstract.group(1).strip()
        print(f"  Match found: '{extracted_abstract[:50]}...'")
        print(f"  Full match: '{m_abstract.group()}'")
        print(f"  Group 1: '{m_abstract.group(1)}'")
    else:
        print("  No match found")
        print()
        
        # Try simpler patterns
        simple_patterns = [
            r"T[oó]m\s*t[aá]t\s*:\s*(.*)",
            r"Tóm tắt:\s*(.*)",
            r"(?i)t[oó]m\s*t[aá]t\s*:\s*(.*)",
            r"(?i)tóm tắt:\s*(.*)",
        ]
        
        print("=== Testing Simpler Patterns ===")
        for pattern in simple_patterns:
            m = re.search(pattern, text, flags=re.IGNORECASE)
            print(f"Pattern: {pattern}")
            if m:
                print(f"  Match: '{m.group(1)[:50]}...'")
            else:
                print("  No match")
            print()
    
    # Test the lookahead patterns individually
    print("=== Testing Lookahead Patterns ===")
    
    lookahead_patterns = [
        r"(?:\n\s*Đi[eê]m\s*ch[ií]nh\s*:)",
        r"(?:\n\s*Đi[eê]m\s*ch[ií]nh)",
        r"(?:\n-)",
        r"(?:\Z)",
    ]
    
    for pattern in lookahead_patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        print(f"Lookahead: {pattern}")
        if m:
            print(f"  Found at position: {m.start()}-{m.end()}")
            print(f"  Match: '{m.group()}'")
        else:
            print("  Not found")
        print()
    
    # Test the complete pattern step by step
    print("=== Step-by-step Pattern Construction ===")
    
    # Step 1: Find the label
    label_pattern = r"T[oó]m\s*t[aá]t\s*:\s*"
    label_match = re.search(label_pattern, text, flags=re.IGNORECASE)
    if label_match:
        print(f"Label found: '{label_match.group()}' at position {label_match.start()}-{label_match.end()}")
        
        # Step 2: Find the delimiter
        delimiter_pattern = r"\n\s*Đi[eê]m\s*ch[ií]nh"
        delimiter_match = re.search(delimiter_pattern, text, flags=re.IGNORECASE)
        if delimiter_match:
            print(f"Delimiter found: '{delimiter_match.group()}' at position {delimiter_match.start()}-{delimiter_match.end()}")
            
            # Step 3: Extract the content between label and delimiter
            start_pos = label_match.end()
            end_pos = delimiter_match.start()
            content = text[start_pos:end_pos].strip()
            print(f"Extracted content: '{content[:50]}...'")
        else:
            print("Delimiter not found")
    else:
        print("Label not found")

if __name__ == "__main__":
    debug_abstract_pattern()