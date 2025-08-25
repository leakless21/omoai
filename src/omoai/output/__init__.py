"""
OMOAI Output System

Pluggable output formatting and writing system for transcripts and summaries.
"""

from .formatter import OutputFormatter, register_formatter, get_formatter, list_formatters
from .writer import OutputWriter, write_outputs

__all__ = [
    "OutputFormatter",
    "register_formatter",
    "get_formatter",
    "list_formatters",
    "OutputWriter",
    "write_outputs",
]
