#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOCK_FILE="${LOCK_FILE:-$ROOT_DIR/requirements.lock}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
UPGRADE_TOOLCHAIN="${UPGRADE_TOOLCHAIN:-0}"
TORCH_CUDA_INDEX_URL="${TORCH_CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
if [[ "$UPGRADE_TOOLCHAIN" == "1" ]]; then
  if ! python -m pip install --upgrade pip setuptools wheel; then
    echo "warning_toolchain_upgrade_failed using_existing_versions=1"
  fi
fi

if [[ -s "$LOCK_FILE" ]]; then
  tmp_lock_no_torch="$(mktemp)"
  grep -Ev '^(torch==|torchvision==)' "$LOCK_FILE" > "$tmp_lock_no_torch"
  pip install -r "$tmp_lock_no_torch"
  rm -f "$tmp_lock_no_torch"

  torch_ver="$(grep -E '^torch==' "$LOCK_FILE" | head -n1 | cut -d= -f3 || true)"
  tv_ver="$(grep -E '^torchvision==' "$LOCK_FILE" | head -n1 | cut -d= -f3 || true)"
  if [[ -n "$torch_ver" && -n "$tv_ver" ]]; then
    if [[ "$(uname -s)" == "Linux" ]] && command -v nvidia-smi >/dev/null 2>&1; then
      pip install "torch==$torch_ver" "torchvision==$tv_ver" --index-url "$TORCH_CUDA_INDEX_URL"
    else
      pip install "torch==$torch_ver" "torchvision==$tv_ver"
    fi
  fi

  # Install the local package without re-resolving dependencies from pyproject.
  pip install -e . --no-deps --no-build-isolation
else
  echo "deps_lock=missing file=$LOCK_FILE fallback=pyproject_resolve"
  pip install --upgrade --upgrade-strategy eager -e .
fi
python -m pip check
