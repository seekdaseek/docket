#!/bin/bash
# Offline suite. No network, no database, no keys.
set -e
cd "$(dirname "$0")"
python3 -m unittest discover -s tests "$@"
