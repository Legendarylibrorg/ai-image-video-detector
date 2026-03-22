#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SETUP_ENV_FILE="${SETUP_ENV_FILE:-${ENV_FILE:-$ROOT_DIR/.env}}"
SETUP_ENV_EXAMPLE_FILE="${SETUP_ENV_EXAMPLE_FILE:-${ENV_EXAMPLE_FILE:-$ROOT_DIR/.env.example}}"

source "$ROOT_DIR/scripts/setup_linux.sh"

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  ENV_FILE="${ENV_FILE:-$SETUP_ENV_FILE}"
  ENV_EXAMPLE_FILE="${ENV_EXAMPLE_FILE:-$SETUP_ENV_EXAMPLE_FILE}"
  SETUP_ENV_FILE="$ENV_FILE" SETUP_ENV_EXAMPLE_FILE="$ENV_EXAMPLE_FILE" SETUP_RUN_PIPELINE=0 main "$@"
fi
