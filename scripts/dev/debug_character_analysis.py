#!/usr/bin/env python3
"""
Analyze the exact characters in the text.
"""


def analyze_characters():
    """Analyze the exact characters in the Vietnamese text."""

    text = "Tiêu đề: Tích phân đường loại một - Phương pháp tính và ví dụ minh họa\nTóm tắt: Bài giảng hướng dẫn cách tính tích phân đường loại một, ứng dụng trong bài toán tính khối lượng dây không đồng chất. Học sinh cần xác định loại đường cong (theo \\( t \\), \\( x \\), hoặc \\( y \\)), chọn biến tham số phù hợp, và áp dụng công thức tính \\( ds \\) tương ứng. Các ví dụ minh họa tính tích phân trên đường tròn, parabol, và đoạn thẳng.\nĐiểm chính:"

    print("=== Character Analysis ===")

    # Find the specific labels
    labels = [
        ("Tiêu đề:", text.find("Tiêu đề:")),
        ("Tóm tắt:", text.find("Tóm tắt:")),
        ("Điểm chính", text.find("Điểm chính")),
    ]

    for label, pos in labels:
        if pos != -1:
            print(f"Found '{label}' at position {pos}")
            # Show the exact characters
            for _i, char in enumerate(label):
                # actual_pos = pos + _i  # Unused variable removed
                print(f"  '{char}' - Unicode: U+{ord(char):04X}")
            print()

    # Test character matching
    print("=== Testing Character Matching ===")

    # Test specific character classes
    test_cases = [
        ("ó", "[oó]"),
        ("ắ", "[aá]"),
        ("ê", "[eê]"),
        ("í", "[ií]"),
    ]

    for char, char_class in test_cases:
        import re

        pattern = char_class
        matches = re.findall(pattern, char)
        print(
            f"Character '{char}' (U+{ord(char):04X}) matches pattern '{pattern}': {bool(matches)}"
        )

    # Test the exact label patterns
    print("\n=== Testing Exact Label Patterns ===")

    # Extract the exact characters from the text
    title_label = text[text.find("Tiêu đề:") : text.find("Tiêu đề:") + 8]
    abstract_label = text[text.find("Tóm tắt:") : text.find("Tóm tắt:") + 8]

    print(f"Title label: '{title_label}'")
    print(f"Abstract label: '{abstract_label}'")

    # Test if the patterns match the exact characters
    import re

    title_pattern = r"Tiêu đề:"
    abstract_pattern = r"Tóm tắt:"

    title_match = re.search(title_pattern, text, re.IGNORECASE)
    abstract_match = re.search(abstract_pattern, text, re.IGNORECASE)

    print(f"Title pattern '{title_pattern}' matches: {bool(title_match)}")
    print(f"Abstract pattern '{abstract_pattern}' matches: {bool(abstract_match)}")

    if title_match:
        print(f"  Match: '{title_match.group()}'")
    if abstract_match:
        print(f"  Match: '{abstract_match.group()}'")


if __name__ == "__main__":
    analyze_characters()
