#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
if [[ "$#" -eq 0 ]]; then
  exec bash scripts/linux_service.sh start
fi
exec bash scripts/linux_service.sh "$@"
