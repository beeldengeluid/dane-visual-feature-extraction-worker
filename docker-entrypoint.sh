#!/bin/sh

echo "Starting DANE visual feature extraction worker"

poetry run python feature_extraction.py

echo The worker crashed, tailing /dev/null for debugging

# tail -f /dev/null
