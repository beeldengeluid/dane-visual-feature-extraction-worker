#!/bin/sh

echo "Starting DANE visual feature extraction worker"

python worker.py "$@"

echo "The worker has stopped"
