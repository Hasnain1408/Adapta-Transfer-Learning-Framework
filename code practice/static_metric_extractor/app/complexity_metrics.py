"""
Complexity metrics calculation module.

This module provides functions to calculate complexity metrics
from tokenized C source code, including:
- Cyclomatic complexity
- Maximum nesting depth
"""

from typing import List

from .token import Token, TokenType


def compute_cyclomatic_complexity(tokens: List[Token]) -> int:
    """
    Calculate McCabe's Cyclomatic Complexity.
    
    Cyclomatic complexity measures the number of linearly independent
    paths through a program's source code. It's calculated as:
    CC = Number of decision points + 1
    
    Decision points include:
    - if, for, while, do statements
    - case statements in switch
    - && and || operators (short-circuit evaluation creates branches)
    - Ternary operator (?:)
    
    Args:
        tokens: List of tokens from lexical analysis
    
    Returns:
        Cyclomatic complexity value (minimum 1)
    
    Example:
        >>> # For code with 2 if statements and 1 for loop: CC = 4
        >>> complexity = compute_cyclomatic_complexity(tokens)
    """
    cc = 1  # Base complexity
    
    # Decision point token types
    decision_types = {
        TokenType.IF, TokenType.FOR, TokenType.WHILE, TokenType.DO,
        TokenType.CASE, TokenType.AND, TokenType.OR, TokenType.TERNARY
    }
    
    for token in tokens:
        if token.type in decision_types:
            cc += 1
    
    return cc


def compute_max_nesting(tokens: List[Token]) -> int:
    """
    Calculate maximum nesting depth of blocks.
    
    Nesting depth measures how deeply nested the control structures are
    in the code. Each opening brace '{' increases depth, and closing
    brace '}' decreases it. The maximum depth reached indicates the
    deepest level of nesting.
    
    Note: We subtract 1 from the result to exclude the top-level
    function body, focusing on nesting within functions.
    
    Args:
        tokens: List of tokens from lexical analysis
    
    Returns:
        Maximum nesting depth (0 for no nesting, higher values indicate
        more deeply nested code)
    
    Example:
        >>> # Code with nested if inside for loop inside function
        >>> # Function body (depth 1), for loop (depth 2), if (depth 3)
        >>> # Returns 2 (excluding function body)
        >>> max_depth = compute_max_nesting(tokens)
    """
    depth = 0
    max_depth = 0
    
    for token in tokens:
        if token.type == TokenType.LBRACE:
            depth += 1
            if depth > max_depth:
                max_depth = depth
        elif token.type == TokenType.RBRACE:
            depth = max(0, depth - 1)
    
    # Subtract one if top-level function body was counted
    return max_depth - 1 if max_depth > 0 else 0
