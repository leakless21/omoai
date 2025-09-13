#!/usr/bin/env bash
set -euo pipefail

# Simple test runner
# Usage: ./scripts/test.sh [pytest-args]

if command -v pytest >/dev/null 2>&1; then
  pytest -q "$@"
else
  echo "pytest not found. Activate your venv and install dev deps." >&2
  exit 1
fi

