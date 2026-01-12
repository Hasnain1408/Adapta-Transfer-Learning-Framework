"""
Token definitions for C language lexical analysis.

This module defines the token types and Token dataclass used
by the lexer to represent C language tokens.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional


class TokenType(Enum):
    """
    Enumeration of all token types in C language.
    
    Categories:
    - Control flow keywords (IF, FOR, WHILE, etc.)
    - Data type keywords (INT, CHAR, FLOAT, etc.)
    - Storage qualifiers (STATIC, CONST, EXTERN, etc.)
    - Operators (PLUS, MINUS, AND, OR, etc.)
    - Delimiters (LPAREN, RBRACE, SEMICOLON, etc.)
    - Literals (NUMBER, STRING, CHAR_LITERAL)
    - Special tokens (COMMENT, PREPROCESSOR, EOF, etc.)
    """
    IF = "if"
    ELSE = "else"
    FOR = "for"
    WHILE = "while"
    DO = "do"
    SWITCH = "switch"
    CASE = "case"
    DEFAULT = "default"
    RETURN = "return"
    BREAK = "break"
    CONTINUE = "continue"
    GOTO = "goto"

    INT = "int"
    CHAR = "char"
    FLOAT = "float"
    DOUBLE = "double"
    VOID = "void"
    LONG = "long"
    SHORT = "short"
    UNSIGNED = "unsigned"
    SIGNED = "signed"
    STRUCT = "struct"
    UNION = "union"
    ENUM = "enum"
    TYPEDEF = "typedef"
    CONST = "const"
    STATIC = "static"
    EXTERN = "extern"
    AUTO = "auto"
    REGISTER = "register"
    VOLATILE = "volatile"
    SIZEOF = "sizeof"

    AND = "&&"
    OR = "||"
    TERNARY = "?"

    PLUS = "+"
    MINUS = "-"
    STAR = "*"
    SLASH = "/"
    PERCENT = "%"
    ASSIGN = "="
    EQ = "=="
    NE = "!="
    LT = "<"
    GT = ">"
    LE = "<="
    GE = ">="
    NOT = "!"
    BITAND = "&"
    BITOR = "|"
    XOR = "^"
    LSHIFT = "<<"
    RSHIFT = ">>"
    INCREMENT = "++"
    DECREMENT = "--"
    ARROW = "->"
    DOT = "."

    LPAREN = "("
    RPAREN = ")"
    LBRACE = "{"
    RBRACE = "}"
    LBRACKET = "["
    RBRACKET = "]"
    SEMICOLON = ";"
    COMMA = ","
    COLON = ":"

    IDENTIFIER = "identifier"
    NUMBER = "number"
    STRING = "string"
    CHAR_LITERAL = "char_literal"

    COMMENT = "comment"
    NEWLINE = "newline"
    PREPROCESSOR = "preprocessor"
    EOF = "eof"
    UNKNOWN = "unknown"


@dataclass
class Token:
    """
    Represents a single token in the C source code.
    
    Attributes:
        type: The type of the token (from TokenType enum)
        value: The actual string value of the token (None for some token types)
        line: Line number where the token appears (1-indexed)
        column: Column number where the token starts (1-indexed)
    
    Example:
        >>> token = Token(type=TokenType.INT, value="int", line=1, column=1)
        >>> print(f"Token {token.type.name} at line {token.line}")
        Token INT at line 1
    """
    type: TokenType
    value: Optional[str]
    line: int
    column: int
    
    def __repr__(self) -> str:
        """Return a detailed string representation of the token."""
        return (f"Token(type={self.type.name}, value={repr(self.value)}, "
                f"line={self.line}, column={self.column})")

