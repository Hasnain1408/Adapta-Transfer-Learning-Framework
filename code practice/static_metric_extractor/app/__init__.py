"""
Static Metric Extractor for C Source Code.

This package provides tools to extract various software metrics from C source code:
- Size metrics (LOC, comments, blank lines)
- Complexity metrics (cyclomatic complexity, nesting depth)
- Function metrics (function count, fan-out, coupling)
- Halstead metrics (operators, operands)

Main Components:
    Lexer: Tokenizes C source code
    ModuleMetrics: Stores all calculated metrics
    extract_metrics_from_tokens: Main function to calculate all metrics

Example:
    >>> from app import Lexer, extract_metrics_from_tokens, read_file
    >>> source = read_file("example.c")
    >>> lexer = Lexer(source)
    >>> tokens = lexer.tokenize()
    >>> metrics = extract_metrics_from_tokens(tokens, source, "example.c")
    >>> print(f"LOC: {metrics.loc}, CC: {metrics.cyclomatic_complexity}")
"""

from .lexer import Lexer
from .token import Token, TokenType
from .metrics import ModuleMetrics, extract_metrics_from_tokens
from .utils import read_file, is_stdlib_function
from .constants import (
    ALL_C_KEYWORDS,
    CONTROL_FLOW_KEYWORDS,
    DATA_TYPE_KEYWORDS,
    STDLIB_FUNCTIONS,
    CSV_HEADER,
)

# Expose main classes and functions
__all__ = [
    # Core classes
    'Lexer',
    'Token',
    'TokenType',
    'ModuleMetrics',
    
    # Main functions
    'extract_metrics_from_tokens',
    'read_file',
    'is_stdlib_function',
    
    # Constants
    'ALL_C_KEYWORDS',
    'CONTROL_FLOW_KEYWORDS',
    'DATA_TYPE_KEYWORDS',
    'STDLIB_FUNCTIONS',
    'CSV_HEADER',
]

__version__ = "1.0.0"
__author__ = "Static Metrics Team"

