#!/usr/bin/env python3
"""
Main entry point for the Static Metric Extractor.

This module provides a command-line interface to analyze C source files
and extract various software metrics. It supports single file analysis,
directory batch processing, and multiple output formats (human-readable,
CSV, JSON).

Usage:
    python -m app.main <file.c>              # Analyze single file
    python -m app.main -d <directory> -csv   # Batch process to CSV
    python -m app.main <file.c> -json        # Output as JSON
    python -m app.main <file.c> --tokens     # Show token stream
"""

import argparse
import json
import os
import sys

from .lexer import Lexer
from .metrics import extract_metrics_from_tokens, ModuleMetrics
from .utils import read_file
from .constants import CSV_HEADER


def process_file(path: str, args: argparse.Namespace) -> ModuleMetrics:
    """
    Process a single C source file and extract metrics.
    
    Args:
        path: Path to the C source file
        args: Command-line arguments controlling output format
    
    Returns:
        ModuleMetrics object containing all calculated metrics
    
    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    source = read_file(path)
    if source is None:
        raise FileNotFoundError(f"File not found: {path}")
    
    # Tokenize source code
    lexer = Lexer(source)
    tokens = lexer.tokenize()
    
    # Show tokens if requested
    if args.tokens:
        print_tokens(tokens)
    
    # Extract metrics
    metrics = extract_metrics_from_tokens(tokens, source, path)
    
    # Output in requested format
    if args.json:
        print(json.dumps(metrics.to_json(), indent=2))
    elif args.csv:
        print(metrics.to_csv())
    else:
        print_human(metrics)
    
    return metrics


def print_tokens(tokens) -> None:
    """
    Print the token stream in a formatted table.
    
    Args:
        tokens: List of tokens to display
    """
    print("\n" + "="*70)
    print("TOKEN STREAM")
    print("="*70)
    print(f"{'Line':<6} {'Column':<8} {'Type':<18} {'Value':<30}")
    print('-'*70)
    
    for t in tokens:
        if t.type is not None:
            value_str = str(t.value)[:30] if t.value else ''
            print(f"{t.line:<6} {t.column:<8} {t.type.name:<18} {value_str:<30}")
    
    print('='*70 + '\n')


def print_human(metrics: ModuleMetrics) -> None:
    """
    Print metrics in a human-readable format.
    
    Args:
        metrics: ModuleMetrics object to display
    """
    print('\n' + '='*70)
    print(f"METRICS REPORT: {metrics.filename}")
    print('='*70)
    
    print('\n📏 SIZE METRICS')
    print('-'*70)
    print(f"{'Total Lines:':<30} {metrics.total_lines:>10}")
    print(f"{'Lines of Code (LOC):':<30} {metrics.loc:>10}")
    print(f"{'Blank Lines:':<30} {metrics.blank_lines:>10}")
    print(f"{'Comment Lines:':<30} {metrics.comment_lines:>10}")
    print(f"{'Comment Density:':<30} {metrics.comment_density:>10.2%}")
    
    print('\n🔄 COMPLEXITY METRICS')
    print('-'*70)
    print(f"{'Cyclomatic Complexity:':<30} {metrics.cyclomatic_complexity:>10}")
    print(f"{'Max Nesting Depth:':<30} {metrics.max_nesting_depth:>10}")
    
    print('\n🔧 FUNCTION METRICS')
    print('-'*70)
    print(f"{'Function Count:':<30} {metrics.function_count:>10}")
    print(f"{'Function Calls:':<30} {metrics.function_calls:>10}")
    
    print('\n🔗 COUPLING METRICS')
    print('-'*70)
    print(f"{'Fan-Out:':<30} {metrics.fan_out:>10}")
    
    if metrics.called_functions:
        print('\nCalled Functions (top 10):')
        for name in metrics.called_functions[:10]:
            print(f"  • {name}")
    
    print('\n📊 VARIABLE & HALSTEAD METRICS')
    print('-'*70)
    print(f"{'Variable Count:':<30} {metrics.variable_count:>10}")
    print(f"{'Total Operators:':<30} {metrics.operators_count:>10}")
    print(f"{'Total Operands:':<30} {metrics.operands_count:>10}")
    print(f"{'Unique Operators:':<30} {metrics.unique_operators:>10}")
    print(f"{'Unique Operands:':<30} {metrics.unique_operands:>10}")
    
    print('\n' + '='*70 + '\n')


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description='Python Static Metric Extractor for C Source Code',
        epilog='Examples:\n'
               '  %(prog)s file.c                    # Analyze single file\n'
               '  %(prog)s -d ./src -csv             # Batch to CSV\n'
               '  %(prog)s file.c -json              # JSON output\n'
               '  %(prog)s file.c --tokens           # Show tokens\n',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'path',
        nargs='?',
        help='C source file to analyze'
    )
    parser.add_argument(
        '-d', '--dir',
        help='Process all .c files in directory'
    )
    parser.add_argument(
        '-csv',
        action='store_true',
        help='Output in CSV format'
    )
    parser.add_argument(
        '-json',
        action='store_true',
        help='Output in JSON format'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--tokens',
        action='store_true',
        help='Show token stream'
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point for the application."""
    args = parse_args()
    
    try:
        if args.dir:
            # Process directory
            dir_path = os.path.abspath(args.dir)
            if not os.path.exists(dir_path):
                print(f"Error: Directory '{args.dir}' not found", file=sys.stderr)
                sys.exit(1)
            
            # Find all .c files in directory
            files = [
                os.path.join(dir_path, f) 
                for f in os.listdir(dir_path) 
                if f.endswith('.c')
            ]
            
            if not files:
                print(f"No .c files found in '{args.dir}'", file=sys.stderr)
                sys.exit(1)
            
            # Print CSV header once if needed
            if args.csv:
                print(CSV_HEADER)
            
            # Process JSON array
            if args.json:
                print('[')
            
            # Process each file
            for i, file_path in enumerate(files):
                try:
                    process_file(file_path, args)
                    
                    # JSON array separator
                    if args.json and i < len(files) - 1:
                        print(',')
                        
                except Exception as e:
                    print(f"Error processing {file_path}: {e}", file=sys.stderr)
            
            # Close JSON array
            if args.json:
                print(']')
                
        elif args.path:
            # Process single file
            file_path = os.path.abspath(args.path)
            
            if args.csv:
                print(CSV_HEADER)
            
            process_file(file_path, args)
            
        else:
            # No input specified
            print('Error: No input specified. Use -h for help.', file=sys.stderr)
            sys.exit(1)
            
    except KeyboardInterrupt:
        print('\n\nInterrupted by user', file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

