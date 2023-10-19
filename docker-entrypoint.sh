#!/bin/sh

echo "Starting DANE visual feature extraction worker"

python3.10 worker.py "$@"

echo The worker crashed, tailing /dev/null for debugging

tail -f /dev/null
