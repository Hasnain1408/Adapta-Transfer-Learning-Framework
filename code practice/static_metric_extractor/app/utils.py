"""
Utility functions for the static metric extractor.

This module provides helper functions for file I/O and checking
whether functions are part of the C standard library.
"""

import os
from typing import Optional

from .constants import STDLIB_FUNCTIONS


def read_file(path: str) -> Optional[str]:
    """
    Read a file and return its contents as a string.
    
    Attempts to decode the file as UTF-8 first, falling back to Latin-1
    if UTF-8 decoding fails. This ensures we can read most text files
    even with encoding issues.
    
    Args:
        path: Path to the file to read
    
    Returns:
        File contents as a string, or None if file doesn't exist
    
    Example:
        >>> content = read_file("/path/to/file.c")
        >>> if content:
        ...     print(f"File has {len(content)} characters")
    """
    if not os.path.exists(path):
        return None
    
    with open(path, "rb") as f:
        data = f.read()
    
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def is_stdlib_function(name: str) -> bool:
    """
    Check if a function name is part of the C standard library.
    
    This is used to exclude standard library functions from fan-out
    metrics, as they don't represent coupling to user-defined code.
    
    Args:
        name: Function name to check
    
    Returns:
        True if the function is in the standard library, False otherwise
    
    Example:
        >>> is_stdlib_function("printf")
        True
        >>> is_stdlib_function("my_custom_function")
        False
    """
    return name in STDLIB_FUNCTIONS

