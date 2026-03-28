#!/usr/bin/env bash
# Run the same checks as .github/workflows/ci.yml (lint, format, mypy, pytest).
# Install tooling first: pip install -r requirements.txt -r requirements-dev.txt
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
