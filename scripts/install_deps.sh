#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOCK_FILE="${LOCK_FILE:-$ROOT_DIR/requirements.lock}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip

if [[ -s "$LOCK_FILE" ]]; then
  pip install -r "$LOCK_FILE"
  # Install the local package without re-resolving dependencies from pyproject.
  pip install -e . --no-deps
else
  echo "deps_lock=missing file=$LOCK_FILE fallback=pyproject_resolve"
  pip install -e .
fi

