"""
Notion Parser - A comprehensive parser for Notion documents with OCR and PDF support
"""

from .parser import NotionParser
from .benchmark import benchmark_parsing, analyze_output, validate_parsing_quality

__version__ = "1.0.0"
__all__ = ["NotionParser", "benchmark_parsing", "analyze_output", "validate_parsing_quality"]