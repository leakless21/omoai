#!/usr/bin/env python3
"""
Detailed test to understand the exact parsing logic issue.
"""

import re
from src.omoai.api.services import _normalize_summary

def test_step_by_step():
    """Test the parsing logic step by step."""
    
    # Test case based on the problematic output from the issue
    problematic_summary = {
        "title": "",
        "summary": "Tiêu đề: Tích phân đường loại một - Phương pháp tính và ví dụ minh họa\nTóm tắt: Bài giảng hướng dẫn cách tính tích phân đường loại một, ứng dụng trong bài toán tính khối lượng dây không đồng chất. Học sinh cần xác định loại đường cong (theo \\( t \\), \\( x \\), hoặc \\( y \\)), chọn biến tham số phù hợp, và áp dụng công thức tính \\( ds \\) tương ứng. Các ví dụ minh họa tính tích phân trên đường tròn, parabol, và đoạn thẳng.\nĐiểm chính:",
        "points": ["Point 1", "Point 2", "Point 3"]
    }
    
    print("=== Step-by-step analysis ===")
    
    # Step 1: Extract title, abstract, points from dict
    title = problematic_summary.get("title") or ""
    abstract = problematic_summary.get("summary") or ""
    points = problematic_summary.get("points") or []
    
    print(f"Step 1 - Extracted from dict:")
    print(f"  title: '{title}'")
    print(f"  abstract: '{abstract[:50]}...'")
    print(f"  points: {points}")
    
    # Step 2: Create combined_text
    combined_text = ""
    if title:
        combined_text += str(title).strip() + "\n\n"
    if abstract:
        combined_text += str(abstract).strip()
    
    print(f"\nStep 2 - Combined text:")
    print(f"  combined_text: '{combined_text[:100]}...'")
    
    # Step 3: Check if combined_text contains Vietnamese labels
    pattern = r"(?i)T[oó]m\s*t[aá]t|Đi[eê]m\s*ch[ií]nh|^-"
    has_vietnamese_labels = re.search(pattern, combined_text, flags=re.MULTILINE)
    
    print(f"\nStep 3 - Vietnamese labels check:")
    print(f"  Pattern: {pattern}")
    print(f"  Has Vietnamese labels: {bool(has_vietnamese_labels)}")
    print(f"  Match: {has_vietnamese_labels.group() if has_vietnamese_labels else 'None'}")
    
    # Step 4: If it has Vietnamese labels, parse text
    if has_vietnamese_labels:
        print(f"\nStep 4 - Parsing text with Vietnamese labels...")
        
        # Test the parsing logic manually
        text = combined_text
        
        # Test title extraction
        m_title = re.search(r"Ti[eê]u\s*đ[eề]\s*:\s*(.+)", text, flags=re.IGNORECASE)
        if m_title:
            extracted_title = m_title.group(1).splitlines()[0].strip()
            print(f"  Extracted title: '{extracted_title}'")
        else:
            print(f"  No title found")
        
        # Test abstract extraction
        m_abstract = re.search(
            r"T[oó]m\s*t[aá]t\s*:\s*([\s\S]*?)(?:\n\s*Đi[eê]m\s*ch[ií]nh\s*:|\n\s*Đi[eê]m\s*ch[ií]nh|\n-|\Z)",
            text,
            flags=re.IGNORECASE,
        )
        if m_abstract:
            extracted_abstract = m_abstract.group(1).strip()
            print(f"  Extracted abstract: '{extracted_abstract[:50]}...'")
        else:
            print(f"  No abstract found with Vietnamese pattern")
            
            # Try English pattern
            m_abstract2 = re.search(r"Summary\s*:\s*([\s\S]*?)(?:\n-|\Z)", text, flags=re.IGNORECASE)
            if m_abstract2:
                extracted_abstract = m_abstract2.group(1).strip()
                print(f"  Extracted abstract (English): '{extracted_abstract[:50]}...'")
            else:
                print(f"  No abstract found with English pattern")
    
    # Step 5: Final result
    final_result = _normalize_summary(problematic_summary)
    print(f"\nStep 5 - Final result:")
    print(f"  title: '{final_result['title']}'")
    print(f"  summary: '{final_result['summary'][:50]}...'")
    print(f"  points: {final_result['points']}")

if __name__ == "__main__":
    test_step_by_step()
