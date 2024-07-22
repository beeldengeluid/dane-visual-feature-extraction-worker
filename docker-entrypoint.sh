#!/bin/sh

echo "Starting DANE visual feature extraction worker"

poetry run python worker.py "$@"

echo "The worker has stopped"
