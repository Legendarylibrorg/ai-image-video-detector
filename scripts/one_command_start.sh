#!/usr/bin/env bash
set -euo pipefail

# One-command Linux setup + broad data collection + full training + API/UI serve.
# Usage:
#   bash scripts/one_command_start.sh
# Optional:
#   HF_TOKEN=... HOST=0.0.0.0 PORT=8000 bash scripts/one_command_start.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

if command -v apt-get >/dev/null 2>&1; then
  if command -v sudo >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y python3 python3-venv python3-pip build-essential
  else
    apt-get update
    apt-get install -y python3 python3-venv python3-pip build-essential
  fi
fi

bash scripts/do.sh train-all-types
HOST="$HOST" PORT="$PORT" bash scripts/do.sh serve
