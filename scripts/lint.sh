#!/usr/bin/env bash
set -euo pipefail

# Ruff-based linting helper.
#
# Usage:
#   ./scripts/lint.sh           # run checks only
#   ./scripts/lint.sh --fix     # apply autofixes + formatting

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
cd "${REPO_DIR}"

FIX=0
if [[ "${1:-}" == "--fix" ]]; then
  FIX=1
fi

pick_ruff_runner() {
  local runner=""
  if command -v python3 >/dev/null 2>&1 && python3 -c "import ruff" >/dev/null 2>&1; then
    runner="python3 -m ruff"
  elif command -v ruff >/dev/null 2>&1; then
    runner="ruff"
  fi
  echo "${runner}"
}

RUFF_RUNNER="$(pick_ruff_runner)"
if [[ -z "${RUFF_RUNNER}" ]]; then
  echo "[lint] Ruff is not installed in the current environment." >&2
  echo "       Install it (e.g., 'python3 -m pip install ruff' or add to your venv)." >&2
  echo "       Once installed, re-run: ./scripts/lint.sh" >&2
  exit 127
fi

echo "[lint] Using: $(${RUFF_RUNNER} --version)"

# Build list of directories to check if they exist
declare -a CHECK_DIRS=()
for d in src tests scripts; do
  [[ -d "$d" ]] && CHECK_DIRS+=("$d")
done

if [[ ${#CHECK_DIRS[@]} -eq 0 ]]; then
  echo "[lint] No target directories found (src/tests/scripts missing)." >&2
  exit 0
fi

if [[ "${FIX}" -eq 1 ]]; then
  echo "[lint] Applying Ruff fixes..."
  ${RUFF_RUNNER} check "${CHECK_DIRS[@]}" --fix
  # Optionally format code (compatible with Black style)
  if ${RUFF_RUNNER} --help 2>/dev/null | grep -qE "^\s*format\s"; then
    echo "[lint] Running Ruff formatter..."
    ${RUFF_RUNNER} format "${CHECK_DIRS[@]}"
  fi
else
  echo "[lint] Running Ruff checks..."
  ${RUFF_RUNNER} check "${CHECK_DIRS[@]}"
fi

echo "[lint] Done."
