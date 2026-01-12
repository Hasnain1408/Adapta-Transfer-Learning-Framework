"""
Main metrics module and data structures.

This module defines the ModuleMetrics dataclass and provides the main
function to extract all metrics from tokenized source code.
"""

from typing import List

from .token import Token
from .size_metrics import compute_loc, compute_comment_and_blank, compute_comment_density
from .complexity_metrics import compute_cyclomatic_complexity, compute_max_nesting
from .function_metrics import (
    compute_function_count, compute_fan_out, compute_function_calls
)
from .halstead_metrics import compute_variable_count, compute_halstead_metrics


class ModuleMetrics:
    """
    Container for all metrics extracted from a source file.
    
    This class holds all the calculated metrics and provides methods
    to export them in different formats (CSV, JSON, human-readable).
    
    Attributes:
        filename: Name of the source file
        
        Size metrics:
            total_lines: Total number of lines in file
            loc: Lines of code (excluding blanks and comments)
            blank_lines: Number of blank lines
            comment_lines: Number of comment lines
            comment_density: Ratio of comments to code
        
        Complexity metrics:
            cyclomatic_complexity: McCabe's cyclomatic complexity
            max_nesting_depth: Maximum depth of nested blocks
        
        Function metrics:
            function_count: Number of function definitions
            function_calls: Total number of function calls
            fan_out: Number of distinct external functions called
            called_functions: List of external functions called
        
        Variable metrics:
            variable_count: Number of variable declarations
        
        Halstead metrics:
            operators_count: Total operators
            operands_count: Total operands
            unique_operators: Count of unique operators
            unique_operands: Count of unique operands
    """
    
    def __init__(self, filename: str):
        """Initialize metrics container with filename."""
        self.filename = filename
        self.total_lines = 0
        self.loc = 0
        self.blank_lines = 0
        self.comment_lines = 0
        self.comment_density = 0.0
        self.cyclomatic_complexity = 1
        self.max_nesting_depth = 0
        self.function_count = 0
        self.function_calls = 0
        self.fan_out = 0
        self.called_functions: List[str] = []
        self.variable_count = 0
        self.operators_count = 0
        self.operands_count = 0
        self.unique_operators = 0
        self.unique_operands = 0

    def to_csv(self) -> str:
        """
        Export metrics as a CSV row.
        
        Returns:
            Comma-separated string of all metric values
        """
        return ",".join([
            self.filename,
            str(self.total_lines),
            str(self.loc),
            str(self.blank_lines),
            str(self.comment_lines),
            f"{self.comment_density:.2f}",
            str(self.cyclomatic_complexity),
            str(self.max_nesting_depth),
            str(self.function_count),
            str(self.function_calls),
            str(self.fan_out),
            str(self.variable_count),
            str(self.operators_count),
            str(self.operands_count),
            str(self.unique_operators),
            str(self.unique_operands),
        ])

    def to_json(self) -> dict:
        """
        Export metrics as a JSON-serializable dictionary.
        
        Organizes metrics into logical categories for easier consumption.
        
        Returns:
            Dictionary with categorized metrics
        """
        return {
            'filename': self.filename,
            'size_metrics': {
                'total_lines': self.total_lines,
                'loc': self.loc,
                'blank_lines': self.blank_lines,
                'comment_lines': self.comment_lines,
                'comment_density': round(self.comment_density, 2),
            },
            'complexity_metrics': {
                'cyclomatic_complexity': self.cyclomatic_complexity,
                'max_nesting_depth': self.max_nesting_depth,
            },
            'function_metrics': {
                'function_count': self.function_count,
                'function_calls': self.function_calls,
            },
            'coupling_metrics': {
                'fan_out': self.fan_out,
            },
            'variable_metrics': {
                'variable_count': self.variable_count,
            },
            'halstead_metrics': {
                'operators': self.operators_count,
                'operands': self.operands_count,
                'unique_operators': self.unique_operators,
                'unique_operands': self.unique_operands,
            }
        }


def extract_metrics_from_tokens(tokens: List[Token], source: str, 
                                filename: str) -> ModuleMetrics:
    """
    Extract all metrics from tokenized source code.
    
    This is the main function that orchestrates the calculation of all
    metrics by calling specialized functions from other modules.
    
    Args:
        tokens: List of tokens from lexical analysis
        source: Original source code string
        filename: Name of the source file
    
    Returns:
        ModuleMetrics object containing all calculated metrics
    
    Example:
        >>> from lexer import Lexer
        >>> source = read_file("example.c")
        >>> lexer = Lexer(source)
        >>> tokens = lexer.tokenize()
        >>> metrics = extract_metrics_from_tokens(tokens, source, "example.c")
        >>> print(f"LOC: {metrics.loc}, CC: {metrics.cyclomatic_complexity}")
    """
    m = ModuleMetrics(filename)
    
    # Size metrics
    m.total_lines = source.count('\n') + 1
    m.loc = compute_loc(source)
    m.comment_lines, m.blank_lines = compute_comment_and_blank(source)
    m.comment_density = compute_comment_density(m.comment_lines, m.loc)
    
    # Complexity metrics
    m.cyclomatic_complexity = compute_cyclomatic_complexity(tokens)
    m.max_nesting_depth = compute_max_nesting(tokens)
    
    # Function metrics
    m.function_count = compute_function_count(tokens)
    m.function_calls = compute_function_calls(tokens)
    m.fan_out, m.called_functions = compute_fan_out(tokens)
    
    # Variable metrics
    m.variable_count = compute_variable_count(tokens)
    
    # Halstead metrics
    (m.operators_count, m.operands_count, 
     m.unique_operators, m.unique_operands) = compute_halstead_metrics(tokens)
    
    return m

