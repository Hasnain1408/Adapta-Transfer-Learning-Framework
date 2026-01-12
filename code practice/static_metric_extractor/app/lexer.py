"""
Lexical analyzer for C source code.

This module implements a lexer that tokenizes C source code into a stream
of tokens. It handles:
- Keywords (if, while, for, etc.)
- Identifiers and literals (variables, numbers, strings)
- Operators and punctuation
- Comments (single-line and multi-line)
- Preprocessor directives
"""

from __future__ import annotations
from typing import List, Optional
import string

from .token import Token, TokenType
from .utils import read_file
from .constants import (
    ALL_C_KEYWORDS, TWO_CHAR_OPERATORS, SINGLE_CHAR_OPERATORS,
    DELIMITERS, WHITESPACE
)


class Lexer:
    """
    Lexical analyzer for C source code.
    
    The lexer scans through source code character by character and produces
    a stream of tokens. It maintains position information (line and column)
    for error reporting and analysis.
    
    Attributes:
        source: The source code string to tokenize
        length: Length of the source code
        pos: Current position in the source code (0-indexed)
        line: Current line number (1-indexed)
        column: Current column number (1-indexed)
    
    Example:
        >>> lexer = Lexer("int x = 5;")
        >>> tokens = lexer.tokenize()
        >>> for token in tokens:
        ...     print(token.type.name, token.value)
        INT int
        IDENTIFIER x
        ASSIGN =
        NUMBER 5
        SEMICOLON ;
        EOF None
    """
    
    def __init__(self, source: str):
        """
        Initialize the lexer with source code.
        
        Args:
            source: C source code string to tokenize
        """
        self.source = source
        self.length = len(source)
        self.pos = 0
        self.line = 1
        self.column = 1

    def peek(self, n: int = 0) -> str:
        """
        Look ahead at character without consuming it.
        
        Args:
            n: Number of characters to look ahead (default: 0 for current)
        
        Returns:
            Character at position (pos + n), or null character if past end
        """
        idx = self.pos + n
        return self.source[idx] if idx < self.length else "\0"

    def advance(self, n: int = 1) -> str:
        """
        Consume and return character(s), updating position tracking.
        
        Args:
            n: Number of characters to advance (default: 1)
        
        Returns:
            The first character at current position before advancing
        """
        ch = self.peek()
        for _ in range(n):
            c = self.peek()
            self.pos += 1
            if c == "\n":
                self.line += 1
                self.column = 1
            else:
                self.column += 1
        return ch

    def skip_whitespace(self) -> None:
        """Skip over whitespace characters (space, tab, carriage return)."""
        while self.peek() in WHITESPACE:
            self.advance()

    def make_token(self, ttype: TokenType, value: Optional[str], 
                   line: int, column: int) -> Token:
        """
        Create a token with the given attributes.
        
        Args:
            ttype: Type of the token
            value: String value of the token
            line: Line number where token appears
            column: Column number where token starts
        
        Returns:
            A new Token instance
        """
        return Token(type=ttype, value=value, line=line, column=column)

    def tokenize(self) -> List[Token]:
        """
        Tokenize the entire source code.
        
        Returns:
            List of tokens including a final EOF token
        """
        tokens: List[Token] = []
        while True:
            tok = self.next_token()
            tokens.append(tok)
            if tok.type == TokenType.EOF:
                break
        return tokens

    def next_token(self) -> Token:
        """
        Get the next token from the source code.
        
        This is the main tokenization method that identifies and returns
        the next token by examining the current character(s).
        
        Returns:
            The next Token in the source code
        """
        self.skip_whitespace()
        if self.pos >= self.length:
            return self.make_token(TokenType.EOF, None, self.line, self.column)

        c = self.peek()
        line = self.line
        column = self.column

        # Newline
        if c == "\n":
            self.advance()
            return self.make_token(TokenType.NEWLINE, "\n", line, column)

        # Preprocessor
        if c == '#' and (self.column == 1 or self.pos == 0 or 
                         self.source[self.pos-1] == '\n'):
            return self._tokenize_preprocessor(line, column)

        # Comments
        if c == '/' and (self.peek(1) == '/' or self.peek(1) == '*'):
            return self._tokenize_comment(line, column)

        # String/char literals
        if c in ('"', "'"):
            return self._tokenize_string_or_char(c, line, column)

        # Numbers
        if c.isdigit() or (c == '.' and self.peek(1).isdigit()):
            return self._tokenize_number(line, column)

        # Identifiers / keywords
        if c.isalpha() or c == '_':
            return self._tokenize_identifier_or_keyword(line, column)

        # Two-char operators
        two_char = c + self.peek(1)
        if two_char in TWO_CHAR_OPERATORS:
            self.advance(2)
            token_type = getattr(TokenType, TWO_CHAR_OPERATORS[two_char])
            return self.make_token(token_type, two_char, line, column)

        # Single-char tokens (operators and delimiters)
        self.advance()
        
        # Try operators first
        if c in SINGLE_CHAR_OPERATORS:
            token_type = getattr(TokenType, SINGLE_CHAR_OPERATORS[c])
            return self.make_token(token_type, c, line, column)
        
        # Then delimiters
        if c in DELIMITERS:
            token_type = getattr(TokenType, DELIMITERS[c])
            return self.make_token(token_type, c, line, column)
        
        # Unknown character
        return self.make_token(TokenType.UNKNOWN, c, line, column)

    def _tokenize_preprocessor(self, line: int, column: int) -> Token:
        """
        Tokenize a preprocessor directive.
        
        Preprocessor directives start with # and continue to end of line,
        handling line continuation with backslash-newline.
        """
        start = self.pos
        self.advance()  # Skip '#'
        
        while self.peek() != '\n' and self.peek() != '\0':
            # Handle backslash-newline line continuation
            if self.peek() == '\\' and self.peek(1) == '\n':
                self.advance(2)
            else:
                self.advance()
        
        val = self.source[start:self.pos]
        return self.make_token(TokenType.PREPROCESSOR, val, line, column)

    def _tokenize_comment(self, line: int, column: int) -> Token:
        """
        Tokenize a single-line or multi-line comment.
        
        Single-line comments start with // and go to end of line.
        Multi-line comments start with /* and end with */.
        """
        start = self.pos
        self.advance()  # Skip first '/'
        
        if self.peek() == '/':
            # Single-line comment
            self.advance()
            while self.peek() != '\n' and self.peek() != '\0':
                self.advance()
        elif self.peek() == '*':
            # Multi-line comment
            self.advance()
            while not (self.peek() == '*' and self.peek(1) == '/') and \
                  self.peek() != '\0':
                self.advance()
            
            # Consume closing */
            if self.peek() == '*':
                self.advance()
                if self.peek() == '/':
                    self.advance()
        
        val = self.source[start:self.pos]
        return self.make_token(TokenType.COMMENT, val, line, column)

    def _tokenize_string_or_char(self, quote: str, line: int, 
                                 column: int) -> Token:
        """
        Tokenize a string literal or character literal.
        
        Args:
            quote: The opening quote character (" or ')
            line: Line number where literal starts
            column: Column number where literal starts
        """
        self.advance()  # Skip opening quote
        start = self.pos
        
        while self.peek() != quote and self.peek() != '\0':
            if self.peek() == '\\' and self.peek(1) != '\0':
                self.advance()  # Skip escape character
            self.advance()
        
        val = self.source[start:self.pos]
        
        if self.peek() == quote:
            self.advance()  # Skip closing quote
        
        token_type = TokenType.STRING if quote == '"' else TokenType.CHAR_LITERAL
        return self.make_token(token_type, val, line, column)

    def _tokenize_number(self, line: int, column: int) -> Token:
        """
        Tokenize a numeric literal.
        
        Handles various number formats:
        - Decimal: 123, 123.45
        - Hexadecimal: 0x1A2B
        - Binary: 0b1010
        - Octal: 0755
        - Scientific notation: 1.23e-4
        - Suffixes: 123L, 3.14f
        """
        start = self.pos
        
        # Handle special prefixes (0x, 0b)
        if self.peek() == '0':
            self.advance()
            
            if self.peek() in ('x', 'X'):
                # Hexadecimal
                self.advance()
                while self.peek().isdigit() or self.peek().lower() in 'abcdef':
                    self.advance()
            elif self.peek() in ('b', 'B'):
                # Binary
                self.advance()
                while self.peek() in ('0', '1'):
                    self.advance()
            else:
                # Octal or decimal
                while self.peek().isdigit():
                    self.advance()
        else:
            # Decimal integer part
            while self.peek().isdigit():
                self.advance()
        
        # Fractional part
        if self.peek() == '.' and self.peek(1).isdigit():
            self.advance()  # Skip '.'
            while self.peek().isdigit():
                self.advance()
        
        # Exponent part
        if self.peek().lower() == 'e':
            self.advance()
            if self.peek() in ('+', '-'):
                self.advance()
            while self.peek().isdigit():
                self.advance()
        
        # Suffixes (L, LL, U, F, etc.)
        while self.peek().isalpha():
            self.advance()
        
        val = self.source[start:self.pos]
        return self.make_token(TokenType.NUMBER, val, line, column)

    def _tokenize_identifier_or_keyword(self, line: int, column: int) -> Token:
        """
        Tokenize an identifier or keyword.
        
        Identifiers start with letter or underscore and can contain
        letters, digits, and underscores. Keywords are special identifiers
        recognized by the language.
        """
        start = self.pos
        
        while self.peek().isalnum() or self.peek() == '_':
            self.advance()
        
        val = self.source[start:self.pos]
        
        # Map keywords to their token types
        keyword_map = {
            'if': TokenType.IF, 'else': TokenType.ELSE, 
            'for': TokenType.FOR, 'while': TokenType.WHILE,
            'do': TokenType.DO, 'switch': TokenType.SWITCH, 
            'case': TokenType.CASE, 'default': TokenType.DEFAULT,
            'return': TokenType.RETURN, 'break': TokenType.BREAK, 
            'continue': TokenType.CONTINUE, 'goto': TokenType.GOTO,
            'int': TokenType.INT, 'char': TokenType.CHAR, 
            'float': TokenType.FLOAT, 'double': TokenType.DOUBLE,
            'void': TokenType.VOID, 'long': TokenType.LONG, 
            'short': TokenType.SHORT,
            'unsigned': TokenType.UNSIGNED, 'signed': TokenType.SIGNED, 
            'struct': TokenType.STRUCT,
            'union': TokenType.UNION, 'enum': TokenType.ENUM, 
            'typedef': TokenType.TYPEDEF, 'const': TokenType.CONST,
            'static': TokenType.STATIC, 'extern': TokenType.EXTERN, 
            'auto': TokenType.AUTO, 'register': TokenType.REGISTER,
            'volatile': TokenType.VOLATILE, 'sizeof': TokenType.SIZEOF,
        }
        
        token_type = keyword_map.get(val, TokenType.IDENTIFIER)
        return self.make_token(token_type, val, line, column)

