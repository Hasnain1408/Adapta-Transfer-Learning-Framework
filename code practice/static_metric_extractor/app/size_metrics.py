"""
Size metrics calculation module.

This module provides functions to calculate size-related metrics
from C source code, including:
- Lines of Code (LOC)
- Comment lines
- Blank lines
- Comment density
"""

from typing import Tuple


def compute_loc(source: str) -> int:
    """
    Calculate Lines of Code (LOC).
    
    LOC counts non-blank, non-comment-only lines that contain executable
    code or declarations. This metric excludes:
    - Blank lines
    - Lines containing only comments
    - Lines inside multi-line comments
    
    Args:
        source: C source code as a string
    
    Returns:
        Number of lines of code
    
    Example:
        >>> code = '''
        ... int main() {
        ...     // This is a comment
        ...     int x = 5;  // Initialize x
        ...     return 0;
        ... }
        ... '''
        >>> loc = compute_loc(code)
        >>> print(loc)  # Returns 4 (main, x=5, return, closing brace)
    """
    lines = source.splitlines()
    loc = 0
    in_multiline = False
    
    for line in lines:
        s = line.strip()
        if not s:
            continue
        
        i = 0
        code = False
        
        while i < len(s):
            # Handle multi-line comments
            if in_multiline:
                if s[i:i+2] == '*/':
                    in_multiline = False
                    i += 2
                    continue
                i += 1
                continue
            
            if s[i:i+2] == '/*':
                in_multiline = True
                i += 2
                continue
            
            if s[i:i+2] == '//':
                # Rest of line is comment
                break
            
            # Handle strings (they count as code)
            if s[i] == '"':
                i += 1
                while i < len(s) and s[i] != '"':
                    if s[i] == '\\' and i+1 < len(s):
                        i += 2
                    else:
                        i += 1
                if i < len(s) and s[i] == '"':
                    i += 1
                code = True
                continue
            
            # Handle character literals
            if s[i] == "'":
                i += 1
                while i < len(s) and s[i] != "'":
                    if s[i] == '\\' and i+1 < len(s):
                        i += 2
                    else:
                        i += 1
                if i < len(s) and s[i] == "'":
                    i += 1
                code = True
                continue
            
            # Any non-whitespace character is code
            if not s[i].isspace():
                code = True
            
            i += 1
        
        # Count line if it had code and we're not in a comment
        if code and not in_multiline:
            loc += 1
    
    return loc


def compute_comment_and_blank(source: str) -> Tuple[int, int]:
    """
    Calculate comment lines and blank lines.
    
    Comment lines include:
    - Lines containing only comments (single-line or multi-line)
    - Lines inside multi-line comment blocks
    
    Blank lines are lines containing only whitespace.
    
    Args:
        source: C source code as a string
    
    Returns:
        Tuple of (comment_lines, blank_lines)
    
    Example:
        >>> code = '''
        ... int main() {
        ...     // This is a comment
        ...     
        ...     int x = 5;
        ... }
        ... '''
        >>> comments, blanks = compute_comment_and_blank(code)
        >>> print(f"Comments: {comments}, Blanks: {blanks}")
    """
    lines = source.splitlines()
    comment = 0
    blank = 0
    in_multiline = False
    
    for line in lines:
        s = line.strip()
        
        # Blank line
        if not s:
            blank += 1
            continue
        
        i = 0
        only_comment = True
        
        while i < len(s):
            # Inside multi-line comment
            if in_multiline:
                only_comment = True
                if s[i:i+2] == '*/':
                    in_multiline = False
                    i += 2
                else:
                    i += 1
                continue
            
            # Start of multi-line comment
            if s[i:i+2] == '/*':
                in_multiline = True
                only_comment = True
                i += 2
                continue
            
            # Single-line comment
            if s[i:i+2] == '//':
                # Rest of line is comment
                break
            
            # String or char literal means it's code
            if s[i] == '"' or s[i] == "'":
                only_comment = False
                break
            
            # Non-whitespace character means it's code
            if not s[i].isspace():
                only_comment = False
                break
            
            i += 1
        
        # Count as comment line if it only contains comments
        if only_comment:
            comment += 1
    
    return comment, blank


def compute_comment_density(comment_lines: int, loc: int) -> float:
    """
    Calculate comment density as a ratio.
    
    Comment density is the ratio of comment lines to lines of code.
    Higher values indicate more commented code.
    
    Args:
        comment_lines: Number of comment lines
        loc: Lines of code
    
    Returns:
        Comment density ratio (0.0 to 1.0+)
    
    Example:
        >>> density = compute_comment_density(10, 100)
        >>> print(f"Comment density: {density:.2%}")
        Comment density: 10.00%
    """
    return (comment_lines / loc) if loc > 0 else 0.0
