#!/usr/bin/env bash
set -euo pipefail

# Minimal install for CI to run tests without GPU/vLLM.
# - Installs project in editable mode without deps
# - Installs a curated dep set excluding vLLM
# - Installs CPU wheels for torch/torchaudio

PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "[ci-install] Python: $($PYTHON_BIN --version)"

$PYTHON_BIN -m pip install -U pip wheel setuptools

# Install package without pulling heavy deps like vllm
$PYTHON_BIN -m pip install -e . --no-deps

# Base test tooling
$PYTHON_BIN -m pip install -U pytest pytest-cov

# Runtime deps used by tests (intentionally excludes vllm)
REQS=(
  colorama
  huggingface-hub
  ijson
  jiwer
  'litestar[standard]'
  loguru
  numba
  numpy
  pandas
  pillow
  pydub
  pydantic-settings
  pyyaml
  questionary
  sentencepiece
  textgrid
  tqdm
  nvidia-ml-py
  psutil
)

$PYTHON_BIN -m pip install "${REQS[@]}"

# Install CPU wheels for torch/torchaudio matching torchaudio==2.7.1
# Use torch index for reliable CPU artifacts.
$PYTHON_BIN -m pip install --index-url https://download.pytorch.org/whl/cpu \
  'torch==2.7.1' 'torchaudio==2.7.1'

echo "[ci-install] Done."
