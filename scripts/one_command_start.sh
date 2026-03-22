#!/usr/bin/env bash
set -euo pipefail

# One-command Linux setup + broad data collection + full training (pipeline-only).
# For environment bootstrap only, use:
#   ./local.sh setup
# Usage:
#   bash scripts/one_command_start.sh
# Optional:
#   HF_TOKEN=... bash scripts/one_command_start.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

HF_SETUP_REQUIRE_TOKEN="${HF_SETUP_REQUIRE_TOKEN:-0}" SETUP_RUN_PIPELINE=1 bash scripts/setup_linux.sh
