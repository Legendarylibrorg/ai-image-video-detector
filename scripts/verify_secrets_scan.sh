#!/usr/bin/env bash
# Ephemeral venv + detect-secrets scan (same baseline as CI / pre-commit).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

tmp="$(mktemp -d "${TMPDIR:-/tmp}/aid-detect-secrets.XXXXXX")"
cleanup() {
  rm -rf "$tmp"
}
trap cleanup EXIT

python3 -m venv "$tmp/v"
PIP_DISABLE_PIP_VERSION_CHECK=1 "$tmp/v/bin/python" -m pip install -q "detect-secrets==1.5.0"
exec "$tmp/v/bin/detect-secrets" scan --baseline .secrets.baseline "$@"
