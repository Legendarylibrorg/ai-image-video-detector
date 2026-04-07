#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOCK_FILE="${LOCK_FILE:-$ROOT_DIR/requirements.lock}"
LOCK_MANIFEST_FILE="${LOCK_MANIFEST_FILE:-$ROOT_DIR/requirements.lock.json}"
PYPROJECT_FILE="${PYPROJECT_FILE:-$ROOT_DIR/pyproject.toml}"

python3 "$ROOT_DIR/scripts/update_deps_lock.py" update \
  --lock-file "$LOCK_FILE" \
  --manifest-file "$LOCK_MANIFEST_FILE" \
  --pyproject "$PYPROJECT_FILE"
