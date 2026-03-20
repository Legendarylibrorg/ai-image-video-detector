#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOCK_FILE="${LOCK_FILE:-$ROOT_DIR/requirements.lock}"
LOCK_VENV_DIR="${LOCK_VENV_DIR:-$ROOT_DIR/.venv.locktmp}"
TMP_FILE="${LOCK_FILE}.tmp"
KEEP_LOCK_VENV="${KEEP_LOCK_VENV:-0}"

rm -rf "$LOCK_VENV_DIR"
python3 -m venv "$LOCK_VENV_DIR"

# shellcheck disable=SC1091
source "$LOCK_VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel
pip install --upgrade --upgrade-strategy eager -e .
python -m pip check

# Keep lock portable: remove local editable path entry, keep fully pinned transitive set.
pip freeze \
  | grep -Ev '^(ai-image-detector @ file:|^-e |pip==|setuptools==|wheel==)' \
  | sort -f > "$TMP_FILE"

mv "$TMP_FILE" "$LOCK_FILE"
if [[ "$KEEP_LOCK_VENV" != "1" ]]; then
  deactivate || true
  rm -rf "$LOCK_VENV_DIR"
fi
echo "deps_lock=updated file=$LOCK_FILE"
