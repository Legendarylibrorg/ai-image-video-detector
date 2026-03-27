#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/workspace"
cd "$ROOT_DIR"

export VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
export HF_HOME="${HF_HOME:-$ROOT_DIR/.local/hf}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$ROOT_DIR/.local/pip}"
export HOME="${HOME:-/tmp/aid-home}"

umask 077

mkdir -p "$ROOT_DIR/.local" "$HF_HOME" "$PIP_CACHE_DIR" "$HOME"

bash scripts/install_deps.sh

if [[ "$#" -eq 0 ]]; then
  exec ./local.sh help
fi

exec "$@"
