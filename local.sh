#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

cmd="${1:-help}"
shift || true

case "$cmd" in
  setup)
    bash scripts/setup_local.sh
    ;;
  setup-full)
    bash scripts/setup_linux.sh
    ;;
  start)
    bash scripts/do.sh start "$@"
    ;;
  collect)
    bash scripts/do.sh collect-diverse "$@"
    ;;
  collect-fast)
    bash scripts/do.sh collect-fast "$@"
    ;;
  train)
    bash scripts/do.sh train-all "$@"
    ;;
  doctor)
    bash scripts/do.sh doctor "$@"
    ;;
  scan)
    bash scripts/do.sh scan "$@"
    ;;
  deps-update)
    bash scripts/do.sh deps-update
    ;;
  status)
    bash scripts/do.sh status
    ;;
  help|*)
    echo "usage: ./local.sh [setup|setup-full|doctor|start|collect|collect-fast|train|scan|deps-update|status]"
    ;;
esac
