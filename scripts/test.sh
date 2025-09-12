#!/usr/bin/env bash
set -euo pipefail

if rg -n "from src\\.omoai|import src\\.omoai" src tests scripts docs >/dev/null; then
  echo "Error: Legacy imports using 'src.omoai' detected. Please use 'omoai'." >&2
  rg -n "from src\\.omoai|import src\\.omoai" src tests scripts docs || true
  exit 1
fi

if command -v pytest >/dev/null 2>&1; then
  PYTEST_ARGS=${PYTEST_ARGS:-"-q"}
  exec pytest ${PYTEST_ARGS}
else
  echo "pytest is not installed. Please install dev dependencies." >&2
  exit 127
fi
