"""
Constants and configuration for static metric extractor.

This module contains all magic strings, keyword definitions, and
configuration constants used throughout the application.
"""

from typing import Set, Dict

# ==============================================================================
# C LANGUAGE KEYWORDS AND OPERATORS
# ==============================================================================

# C Language Keywords - Control Flow
CONTROL_FLOW_KEYWORDS: Set[str] = {
    'if', 'else', 'for', 'while', 'do', 'switch', 
    'case', 'default', 'break', 'continue', 'return', 'goto'
}

# C Language Keywords - Data Types
DATA_TYPE_KEYWORDS: Set[str] = {
    'int', 'char', 'float', 'double', 'void', 'long', 'short',
    'unsigned', 'signed', 'struct', 'union', 'enum', 'typedef'
}

# C Language Keywords - Storage Classes and Qualifiers
STORAGE_QUALIFIERS: Set[str] = {
    'const', 'static', 'extern', 'auto', 'register', 'volatile'
}

# C Language Keywords - Other
OTHER_KEYWORDS: Set[str] = {
    'sizeof'
}

# All C Keywords
ALL_C_KEYWORDS: Set[str] = (
    CONTROL_FLOW_KEYWORDS | 
    DATA_TYPE_KEYWORDS | 
    STORAGE_QUALIFIERS | 
    OTHER_KEYWORDS
)

# ==============================================================================
# OPERATOR DEFINITIONS
# ==============================================================================

# Two-character operators (must be checked before single-char)
TWO_CHAR_OPERATORS: Dict[str, str] = {
    '&&': 'AND',
    '||': 'OR',
    '==': 'EQ',
    '!=': 'NE',
    '<=': 'LE',
    '>=': 'GE',
    '<<': 'LSHIFT',
    '>>': 'RSHIFT',
    '++': 'INCREMENT',
    '--': 'DECREMENT',
    '->': 'ARROW',
}

# Single-character operators
SINGLE_CHAR_OPERATORS: Dict[str, str] = {
    '+': 'PLUS',
    '-': 'MINUS',
    '*': 'STAR',
    '/': 'SLASH',
    '%': 'PERCENT',
    '=': 'ASSIGN',
    '<': 'LT',
    '>': 'GT',
    '!': 'NOT',
    '&': 'BITAND',
    '|': 'BITOR',
    '^': 'XOR',
    '?': 'TERNARY',
    '.': 'DOT',
}

# Punctuation and delimiters
DELIMITERS: Dict[str, str] = {
    '(': 'LPAREN',
    ')': 'RPAREN',
    '{': 'LBRACE',
    '}': 'RBRACE',
    '[': 'LBRACKET',
    ']': 'RBRACKET',
    ';': 'SEMICOLON',
    ',': 'COMMA',
    ':': 'COLON',
}

# ==============================================================================
# STANDARD LIBRARY FUNCTIONS
# ==============================================================================

# Standard C library functions to exclude from fan-out metrics
STDLIB_FUNCTIONS: Set[str] = {
    # stdio.h
    'printf', 'fprintf', 'sprintf', 'snprintf', 'scanf', 'fscanf', 'sscanf',
    'fopen', 'fclose', 'fread', 'fwrite', 'fgets', 'fputs', 'fgetc', 'fputc',
    'getchar', 'putchar', 'puts', 'gets', 'fflush', 'fseek', 'ftell', 'rewind',
    'perror', 'clearerr', 'feof', 'ferror', 'remove', 'rename', 'tmpfile',
    
    # stdlib.h
    'malloc', 'calloc', 'realloc', 'free', 'exit', 'abort', 'atexit',
    'atoi', 'atol', 'atof', 'strtol', 'strtoul', 'strtod', 'strtof',
    'rand', 'srand', 'abs', 'labs', 'qsort', 'bsearch', 'getenv', 'system',
    
    # string.h
    'strlen', 'strcpy', 'strncpy', 'strcat', 'strncat', 'strcmp', 'strncmp',
    'strchr', 'strrchr', 'strstr', 'strtok', 'memcpy', 'memmove', 'memset',
    'memcmp', 'strcoll', 'strxfrm', 'strerror',
    
    # ctype.h
    'isalpha', 'isdigit', 'isalnum', 'isspace', 'toupper', 'tolower',
    'isupper', 'islower', 'isprint', 'iscntrl', 'ispunct', 'isgraph', 'isxdigit',
    
    # math.h
    'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'atan2',
    'sinh', 'cosh', 'tanh', 'exp', 'log', 'log10', 'log2',
    'pow', 'sqrt', 'cbrt', 'ceil', 'floor', 'fabs', 'fmod',
    
    # time.h
    'time', 'clock', 'difftime', 'mktime', 'strftime', 'localtime', 'gmtime',
    
    # assert.h
    'assert',
}

# ==============================================================================
# METRIC CALCULATION CONSTANTS
# ==============================================================================

# Whitespace characters
WHITESPACE: Set[str] = {' ', '\t', '\r'}

# Quote characters
QUOTES: Set[str] = {'"', "'"}

# Comment start sequences
COMMENT_SEQUENCES: Set[str] = {'//', '/*'}

# Cyclomatic complexity decision points
COMPLEXITY_KEYWORDS: Set[str] = {
    'if', 'for', 'while', 'do', 'case', '&&', '||', '?'
}

# ==============================================================================
# CSV OUTPUT CONFIGURATION
# ==============================================================================

CSV_HEADER: str = (
    "filename,total_lines,loc,blank_lines,comment_lines,comment_density,"
    "cyclomatic_complexity,max_nesting_depth,function_count,function_calls,"
    "fan_out,variable_count,operators,operands,unique_operators,unique_operands"
)

# ==============================================================================
# FILE EXTENSIONS
# ==============================================================================

C_FILE_EXTENSIONS: Set[str] = {'.c', '.h'}
CPP_FILE_EXTENSIONS: Set[str] = {'.cpp', '.cc', '.cxx', '.hpp', '.hh', '.hxx'}
SUPPORTED_EXTENSIONS: Set[str] = C_FILE_EXTENSIONS | CPP_FILE_EXTENSIONS
