#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOCK_FILE="${LOCK_FILE:-$ROOT_DIR/requirements.lock}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
UPGRADE_TOOLCHAIN="${UPGRADE_TOOLCHAIN:-0}"
TORCH_CUDA_INDEX_URL="${TORCH_CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
DEPS_STAMP_FILE="${DEPS_STAMP_FILE:-$VENV_DIR/.deps_stamp}"
DEPS_EXTRA="${DEPS_EXTRA:-pipeline}"

hash_cmd() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum
    return
  fi
  shasum -a 256
}

pip_cmd() {
  local subcommand="$1"
  shift
  if [[ "$subcommand" == "install" ]]; then
    PIP_DISABLE_PIP_VERSION_CHECK=1 python -m pip install --progress-bar off "$@"
    return
  fi
  PIP_DISABLE_PIP_VERSION_CHECK=1 python -m pip "$subcommand" "$@"
}

deps_fingerprint() {
  {
    if [[ -f "$LOCK_FILE" ]]; then
      cat "$LOCK_FILE"
    fi
    if [[ -f "$ROOT_DIR/pyproject.toml" ]]; then
      cat "$ROOT_DIR/pyproject.toml"
    fi
  } | hash_cmd | awk '{print $1}'
}

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
deps_fp="$(deps_fingerprint)"
if [[ "$UPGRADE_TOOLCHAIN" != "1" && -f "$DEPS_STAMP_FILE" && "$(cat "$DEPS_STAMP_FILE")" == "$deps_fp" ]]; then
  if python - <<'PY' >/dev/null 2>&1
import ai_image_detector  # noqa: F401
import cv2  # noqa: F401
import datasets  # noqa: F401
import huggingface_hub  # noqa: F401
import numpy  # noqa: F401
import PIL  # noqa: F401
import piexif  # noqa: F401
import safetensors  # noqa: F401
import sklearn  # noqa: F401
import torch  # noqa: F401
import torchvision  # noqa: F401
PY
  then
    if command -v hf >/dev/null 2>&1 && command -v aid-train >/dev/null 2>&1 && command -v aid-video-train >/dev/null 2>&1; then
      echo "deps_status=up_to_date"
      exit 0
    fi
  fi
fi

if [[ "$UPGRADE_TOOLCHAIN" == "1" ]]; then
  if ! pip_cmd install --quiet --retries 1 --timeout 15 --upgrade pip setuptools wheel; then
    echo "warning_toolchain_upgrade_failed using_existing_versions=1"
  fi
fi

if [[ -s "$LOCK_FILE" ]]; then
  tmp_lock_no_torch="$(mktemp)"
  grep -Ev '^(torch==|torchvision==)' "$LOCK_FILE" > "$tmp_lock_no_torch"
  pip_cmd install -r "$tmp_lock_no_torch"
  rm -f "$tmp_lock_no_torch"

  torch_ver="$(grep -E '^torch==' "$LOCK_FILE" | head -n1 | cut -d= -f3 || true)"
  tv_ver="$(grep -E '^torchvision==' "$LOCK_FILE" | head -n1 | cut -d= -f3 || true)"
  if [[ -n "$torch_ver" && -n "$tv_ver" ]]; then
    if [[ "$(uname -s)" == "Linux" ]] && command -v nvidia-smi >/dev/null 2>&1; then
      pip_cmd install "torch==$torch_ver" "torchvision==$tv_ver" --index-url "$TORCH_CUDA_INDEX_URL"
    else
      pip_cmd install "torch==$torch_ver" "torchvision==$tv_ver"
    fi
  fi

  # Install the local package without re-resolving dependencies from pyproject.
  pip_cmd install -e . --no-deps --no-build-isolation
else
  echo "deps_lock=missing file=$LOCK_FILE fallback=pyproject_resolve"
  pip_cmd install --upgrade --upgrade-strategy eager -e ".[${DEPS_EXTRA}]"
fi

if ! python - <<'PY' >/dev/null 2>&1
import ai_image_detector  # noqa: F401
import cv2  # noqa: F401
import datasets  # noqa: F401
import huggingface_hub  # noqa: F401
import numpy  # noqa: F401
import PIL  # noqa: F401
import piexif  # noqa: F401
import safetensors  # noqa: F401
import sklearn  # noqa: F401
import torch  # noqa: F401
import torchvision  # noqa: F401
PY
then
  echo "deps_fail=core_python_deps_missing run=bash scripts/install_deps.sh" >&2
  exit 1
fi

if ! command -v hf >/dev/null 2>&1; then
  echo "deps_fail=huggingface_cli_missing run=bash scripts/install_deps.sh" >&2
  exit 1
fi

if ! command -v aid-train >/dev/null 2>&1; then
  echo "deps_fail=repo_cli_missing cli=aid-train run=bash scripts/install_deps.sh" >&2
  exit 1
fi

if ! command -v aid-video-train >/dev/null 2>&1; then
  echo "deps_fail=repo_cli_missing cli=aid-video-train run=bash scripts/install_deps.sh" >&2
  exit 1
fi

pip_cmd check
printf "%s\n" "$deps_fp" > "$DEPS_STAMP_FILE"
