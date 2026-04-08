#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/workspace"
cd "$ROOT_DIR"

export VENV_DIR="${VENV_DIR:-/opt/aid-venv}"
export HF_HOME="${HF_HOME:-$ROOT_DIR/.local/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$ROOT_DIR/.local/pip}"
export HOME="${HOME:-/tmp/aid-home}"

umask 022

mkdir -p "$ROOT_DIR/.local" "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$PIP_CACHE_DIR" "$HOME"

bash scripts/install_deps.sh

if [[ "$#" -eq 0 ]]; then
  exec ./local.sh help
fi

exec "$@"
