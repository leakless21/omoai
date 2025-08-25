"""
Built-in output formatter plugins.
"""

from .text import TextFormatter
from .json import JsonFormatter
from .srt import SRTFormatter
from .vtt import VTTFormatter
from .markdown import MarkdownFormatter

# Register all built-in formatters
__all__ = ["TextFormatter", "JsonFormatter", "SRTFormatter", "VTTFormatter", "MarkdownFormatter"]
