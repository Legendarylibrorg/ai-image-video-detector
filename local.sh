#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

cmd="${1:-help}"
shift || true

case "$cmd" in
  setup)
    bash scripts/one_command_4090.sh
    ;;
  collect)
    bash scripts/do.sh collect-diverse "$@"
    ;;
  train)
    bash scripts/do.sh train-all "$@"
    ;;
  serve)
    exec bash scripts/linux_service.sh start
    ;;
  full)
    exec bash scripts/linux_service.sh full-start
    ;;
  status|logs|pause|resume|stop|restart)
    exec bash scripts/linux_service.sh "$cmd"
    ;;
  help|*)
    echo "usage: ./local.sh [setup|collect|train|serve|full|status|logs|pause|resume|stop|restart]"
    ;;
esac
