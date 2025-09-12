import pytest

# Deprecated test module:
# In-process punctuation / summarization has been removed.
# This module is intentionally skipped. Historical content removed to
# avoid undefined name lint errors for legacy symbols.
pytest.skip(
    "Deprecated test: in-process punctuate_transcript / summarize_text removed. "
    "Use subprocess-based postprocess wrapper tests instead.",
    allow_module_level=True,
)



