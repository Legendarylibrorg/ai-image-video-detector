#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOCK_FILE="${LOCK_FILE:-$ROOT_DIR/requirements.lock}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
TMP_FILE="${LOCK_FILE}.tmp"

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
pip install -e .

# Keep lock portable: remove local editable path entry, keep fully pinned transitive set.
pip freeze \
  | grep -Ev '^(ai-image-detector @ file:|^-e )' \
  | sort -f > "$TMP_FILE"

mv "$TMP_FILE" "$LOCK_FILE"
echo "deps_lock=updated file=$LOCK_FILE"

