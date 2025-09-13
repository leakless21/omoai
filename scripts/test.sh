#!/usr/bin/env bash
set -euo pipefail

# Run test suite with sensible defaults for CI and local use.
# - By default, skip the heavy ASR config test that requires a local model checkpoint.
# - You can override selection by setting PYTEST_ARGS or OMOAI_CI_RUN_FULL=1.

PYTHON_BIN="${PYTHON_BIN:-python3}"
PYTEST_ARGS=${PYTEST_ARGS:-}
RUN_FULL=${OMOAI_CI_RUN_FULL:-0}

# Auto-enable full run if local ASR checkpoint is present
if [[ "${RUN_FULL}" != "1" ]]; then
  if [[ -f "models/chunkformer/chunkformer-large-vie/pytorch_model.bin" ]]; then
    RUN_FULL=1
  fi
fi

if [[ "${RUN_FULL}" != "1" && -z "${PYTEST_ARGS}" ]]; then
  # Exclude the one test that invokes the real ASR script with model weights
  PYTEST_ARGS="-k 'not test_asr_script_with_valid_config'"
fi

echo "[tests] Using PYTEST_ARGS: ${PYTEST_ARGS}"

# Enable coverage if requested
if [[ "${OMOAI_CI_COVERAGE:-0}" == "1" ]]; then
  echo "[tests] Coverage enabled"
  exec ${PYTHON_BIN} -m pytest -q \
    --cov=src --cov-report=xml --cov-report=term-missing ${PYTEST_ARGS}
else
  exec ${PYTHON_BIN} -m pytest -q ${PYTEST_ARGS}
fi
