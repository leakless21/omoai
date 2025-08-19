from pathlib import Path

import types

from scripts.post import _parse_time_to_seconds, _dedup_overlap, join_punctuated_segments


def test_parse_time_to_seconds():
    assert _parse_time_to_seconds("01:02:03:045") == 3723.045
    assert _parse_time_to_seconds("1:02") == 62.0
    assert _parse_time_to_seconds("00:00:01.500") == 1.5
    assert _parse_time_to_seconds(2.25) == 2.25
    assert _parse_time_to_seconds(None) is None
    assert _parse_time_to_seconds("bad") is None


def test_dedup_overlap():
    prev = "xin chao tat ca moi nguoi"
    nxt = "moi nguoi hom nay the nao"
    # overlap: "moi nguoi"
    result = _dedup_overlap(prev, nxt, max_tokens=4)
    assert result == "hom nay the nao"


def test_join_punctuated_segments_paragraphs():
    segs = [
        {"start": "0.0", "end": "1.0", "text_punct": "Xin chào."},
        {"start": "5.0", "end": "6.0", "text_punct": "Hôm nay khỏe không?"},
    ]
    out = join_punctuated_segments(segs, join_separator=" ", paragraph_gap_seconds=3.0)
    assert "Xin chào." in out and "Hôm nay khỏe không?" in out
    assert "\n\n" in out  # paragraph break due to gap >= 3.0




