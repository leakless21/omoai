#!/usr/bin/env bash
set -euo pipefail

if command -v pytest >/dev/null 2>&1; then
  PYTEST_ARGS=${PYTEST_ARGS:-"-q"}
  exec pytest ${PYTEST_ARGS}
else
  echo "pytest is not installed. Please install dev dependencies." >&2
  exit 127
fi

