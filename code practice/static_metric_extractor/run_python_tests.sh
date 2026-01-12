#!/usr/bin/env bash
set -e
PYTHON=python3
PACKAGE=static_metric_extractor_py

echo "Running Python extractor tests..."
$PYTHON -m $PACKAGE.main ../test/sample2.c
$PYTHON -m $PACKAGE.main -csv ../test/sample2.c
$PYTHON -m $PACKAGE.main -json ../test/sample2.c
echo "All tests ran successfully."