#!/usr/bin/env bash
# Run the same checks as .github/workflows/ci.yml (lint, format, mypy, pytest).
#
# One-time install (matches CI jobs; avoids installing torch/ultralytics for pytest):
#   pip install ruff "mypy>=1.9.0" "types-PyYAML>=6.0" -r requirements-ci.txt
#
# Full training/inference stack (not required for this script): pip install -r requirements.txt
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> ruff check ."
ruff check .

echo "==> ruff format --check ."
ruff format --check .

echo "==> mypy src/"
mypy src/

echo "==> pytest"
pytest

echo "OK: all CI checks passed locally."
