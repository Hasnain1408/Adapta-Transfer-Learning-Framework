"""
Coupling and function metrics calculation module.

This module provides functions to analyze function-related metrics:
- Function count (number of function definitions)
- Function calls (total calls made)
- Fan-out (number of distinct external functions called)
"""

from typing import List, Tuple, Set

from .token import Token, TokenType
from .utils import is_stdlib_function


def is_function_definition(tokens: List[Token], pos: int) -> bool:
    """
    Heuristic to detect if a token position is the start of a function definition.
    
    A function definition is identified by:
    1. An identifier (function name)
    2. Followed by opening parenthesis '('
    3. Parameter list
    4. Closing parenthesis ')'
    5. Opening brace '{' (function body)
    
    This distinguishes function definitions from declarations (which end with ';')
    and function calls.
    
    Args:
        tokens: List of all tokens
        pos: Current position to check
    
    Returns:
        True if position appears to be a function definition, False otherwise
    """
    if pos + 3 >= len(tokens):
        return False
    
    # Look for LPAREN after current position
    found_paren = -1
    paren_depth = 0
    
    # Search in a reasonable window (20 tokens) for LPAREN
    for j in range(pos, min(pos + 20, len(tokens))):
        if tokens[j].type == TokenType.LPAREN:
            found_paren = j
            paren_depth = 1
            
            # Find matching RPAREN
            for k in range(j + 1, len(tokens)):
                if tokens[k].type == TokenType.LPAREN:
                    paren_depth += 1
                elif tokens[k].type == TokenType.RPAREN:
                    paren_depth -= 1
                    
                    if paren_depth == 0:
                        # After RPAREN, look for LBRACE (definition) or SEMICOLON (declaration)
                        for m in range(k + 1, min(k + 8, len(tokens))):
                            if tokens[m].type == TokenType.LBRACE:
                                return True
                            if tokens[m].type == TokenType.SEMICOLON:
                                return False
                        return False
    
    return False


def compute_function_count(tokens: List[Token]) -> int:
    """
    Count the number of function definitions in the source code.
    
    This counts actual function definitions (with bodies), not forward
    declarations or function prototypes.
    
    Args:
        tokens: List of tokens from lexical analysis
    
    Returns:
        Number of function definitions
    
    Example:
        >>> # Code with 3 functions (main, helper1, helper2)
        >>> func_count = compute_function_count(tokens)
        >>> print(func_count)  # Returns 3
    """
    count = 0
    brace_depth = 0
    
    for i, token in enumerate(tokens):
        # Track brace depth to stay at file scope
        if token.type == TokenType.LBRACE:
            brace_depth += 1
        elif token.type == TokenType.RBRACE:
            brace_depth = max(0, brace_depth - 1)
        
        # Only check for function definitions at file scope (depth 0)
        if brace_depth == 0 and token.type == TokenType.IDENTIFIER:
            if is_function_definition(tokens, i):
                count += 1
    
    return count


def is_function_call(tokens: List[Token], pos: int) -> bool:
    """
    Check if an identifier at given position is a function call.
    
    A function call is identified by an identifier immediately
    followed by an opening parenthesis '('.
    
    Args:
        tokens: List of all tokens
        pos: Position to check
    
    Returns:
        True if it's a function call, False otherwise
    """
    if tokens[pos].type != TokenType.IDENTIFIER:
        return False
    
    # Check if next token is LPAREN
    if pos + 1 < len(tokens) and tokens[pos + 1].type == TokenType.LPAREN:
        return True
    
    return False


def compute_fan_out(tokens: List[Token]) -> Tuple[int, List[str]]:
    """
    Calculate fan-out metric (number of distinct external functions called).
    
    Fan-out measures coupling by counting how many different external
    functions are called within function bodies. Standard library functions
    are excluded as they don't represent module coupling.
    
    Args:
        tokens: List of tokens from lexical analysis
    
    Returns:
        Tuple of (fan_out_count, sorted_list_of_called_functions)
    
    Example:
        >>> fan_out, called = compute_fan_out(tokens)
        >>> print(f"Fan-out: {fan_out}")
        >>> print(f"Called functions: {', '.join(called[:5])}")
    """
    calls: Set[str] = set()
    brace_depth = 0
    in_function = False
    
    for i, token in enumerate(tokens):
        # Track when we're inside a function body
        if token.type == TokenType.LBRACE:
            brace_depth += 1
            if brace_depth == 1:
                in_function = True
        elif token.type == TokenType.RBRACE:
            if brace_depth > 0:
                brace_depth -= 1
            if brace_depth == 0:
                in_function = False
        
        # Count function calls inside functions
        if in_function and token.type == TokenType.IDENTIFIER:
            if is_function_call(tokens, i):
                name = token.value
                # Exclude standard library functions from fan-out
                if name and not is_stdlib_function(name):
                    calls.add(name)
    
    return len(calls), sorted(list(calls))


def compute_function_calls(tokens: List[Token]) -> int:
    """
    Count total number of function calls in the source code.
    
    This counts all function calls, including standard library functions
    and duplicate calls to the same function.
    
    Args:
        tokens: List of tokens from lexical analysis
    
    Returns:
        Total number of function calls
    
    Example:
        >>> total_calls = compute_function_calls(tokens)
        >>> print(f"Total function calls: {total_calls}")
    """
    total_calls = 0
    
    for i, token in enumerate(tokens):
        if token.type == TokenType.IDENTIFIER and is_function_call(tokens, i):
            total_calls += 1
    
    return total_calls
