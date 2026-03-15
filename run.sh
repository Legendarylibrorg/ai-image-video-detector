#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
exec bash scripts/linux_service.sh "${1:-start}"
