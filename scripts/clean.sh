#!/usr/bin/env bash
set -euo pipefail

# Clean caches, build artifacts, and temp files safely.

echo "Cleaning Python caches (pyc, __pycache__)..."
find . -type d -name "__pycache__" -prune -exec rm -rf {} +
find . -type f -name "*.pyc" -delete || true

echo "Removing pytest cache..."
rm -rf .pytest_cache || true

echo "Removing build artifacts (egg-info)..."
rm -rf omoai.egg-info src/omoai.egg-info || true

echo "Removing tmp workspace and stray logs..."
rm -rf tmp || true
rm -f server.log || true

echo "Removing empty fixtures cache..."
rm -rf fixtures/__pycache__ || true

echo "Removing dead output package directory if present..."
rm -rf src/omoai/output || true

echo "Done."

