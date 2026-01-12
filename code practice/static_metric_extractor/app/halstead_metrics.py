"""
Halstead metrics and variable counting module.

This module provides functions to calculate:
- Variable declarations count
- Halstead metrics (operators and operands)
"""

from typing import List, Set

from .token import Token, TokenType


def compute_variable_count(tokens: List[Token]) -> int:
    """
    Count variable declarations in the source code.
    
    This uses a simple heuristic: looks for type keywords followed by
    identifiers that are then followed by assignment, semicolon, comma,
    or array bracket. This distinguishes variables from function definitions
    (which have parentheses).
    
    Args:
        tokens: List of tokens from lexical analysis
    
    Returns:
        Approximate count of variable declarations
    
    Note:
        This is a heuristic and may not catch all variable declarations
        perfectly (e.g., in complex declarations or typedefs), but provides
        a reasonable approximation for metrics.
    """
    # Data type tokens that typically precede variable declarations
    type_tokens = {
        TokenType.INT, TokenType.CHAR, TokenType.FLOAT, TokenType.DOUBLE,
        TokenType.VOID, TokenType.LONG, TokenType.SHORT, 
        TokenType.UNSIGNED, TokenType.SIGNED, TokenType.STRUCT
    }
    
    count = 0
    
    for i, token in enumerate(tokens):
        # Check if this is a type keyword or struct/typedef identifier
        is_type = token.type in type_tokens or (
            token.type == TokenType.IDENTIFIER and 
            i + 1 < len(tokens) and 
            tokens[i + 1].type == TokenType.IDENTIFIER
        )
        
        if not is_type:
            continue
        
        # Skip past const, pointer, etc.
        j = i + 1
        while j < len(tokens) and tokens[j].type in (TokenType.CONST, TokenType.STAR):
            j += 1
        
        # Check if followed by identifier (variable name)
        if j < len(tokens) and tokens[j].type == TokenType.IDENTIFIER:
            # Look ahead for assignment, semicolon, comma, or bracket
            k = j + 1
            while k < len(tokens) and k <= j + 3:
                if tokens[k].type in (TokenType.ASSIGN, TokenType.SEMICOLON, 
                                     TokenType.COMMA, TokenType.LBRACKET):
                    count += 1
                    break
                # If we see LPAREN, it's a function, not a variable
                if tokens[k].type == TokenType.LPAREN:
                    break
                k += 1
    
    return count


def compute_halstead_metrics(tokens: List[Token]) -> tuple[int, int, int, int]:
    """
    Calculate Halstead metrics (operators and operands).
    
    Halstead metrics are based on counting operators and operands in code:
    - Operators: arithmetic, logical, relational operators
    - Operands: identifiers (variables, functions) and numeric literals
    
    These form the basis for calculating various Halstead complexity measures
    like volume, difficulty, and effort.
    
    Args:
        tokens: List of tokens from lexical analysis
    
    Returns:
        Tuple of (total_operators, total_operands, unique_operators, unique_operands)
    
    Example:
        >>> ops, opnds, u_ops, u_opnds = compute_halstead_metrics(tokens)
        >>> volume = (ops + opnds) * log2(u_ops + u_opnds)
    """
    # Operator token types (arithmetic, logical, relational, etc.)
    operator_types = {
        TokenType.AND, TokenType.OR, TokenType.TERNARY,
        TokenType.PLUS, TokenType.MINUS, TokenType.STAR, TokenType.SLASH,
        TokenType.PERCENT, TokenType.ASSIGN, TokenType.EQ, TokenType.NE,
        TokenType.LT, TokenType.GT, TokenType.LE, TokenType.GE,
        TokenType.NOT, TokenType.BITAND, TokenType.BITOR, TokenType.XOR,
        TokenType.LSHIFT, TokenType.RSHIFT
    }
    
    operators = 0
    operands = 0
    unique_ops: Set[str] = set()
    unique_opnds: Set[str] = set()
    
    for token in tokens:
        # Count operators
        if token.type in operator_types:
            operators += 1
            if token.value:
                unique_ops.add(token.value)
        
        # Count operands (identifiers and numbers)
        elif token.type in (TokenType.IDENTIFIER, TokenType.NUMBER):
            operands += 1
            if token.value:
                unique_opnds.add(token.value)
    
    return operators, operands, len(unique_ops), len(unique_opnds)
