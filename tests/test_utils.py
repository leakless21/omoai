from pathlib import Path
import types
import numpy as np

from omoai.pipeline.postprocess import _parse_time_to_seconds, _dedup_overlap, join_punctuated_segments


class MockTorchTensor:
    """Minimal mock object that mimics the parts of torch.Tensor used in tests.

    Supports:
    - construction from a shape-like list/tuple [channels, length] => zeros array
    - construction from an iterable of values => 1D array
    - .numpy() -> numpy.ndarray
    - .shape property
    - .to(device) -> self (no-op)
    - .unsqueeze(dim) -> returns self (no-op)
    """
    def __init__(self, shape_or_values):
        if isinstance(shape_or_values, (list, tuple)) and len(shape_or_values) == 2 and all(isinstance(x, int) for x in shape_or_values):
            self._arr = np.zeros(tuple(shape_or_values), dtype=np.float32)
        else:
            self._arr = np.array(shape_or_values, dtype=np.float32)

    def numpy(self):
        return self._arr

    @property
    def shape(self):
        return self._arr.shape

    def to(self, device):
        return self

    def unsqueeze(self, dim):
        return self


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




