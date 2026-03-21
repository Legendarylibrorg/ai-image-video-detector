#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

cmd="${1:-help}"
shift || true

run_do() {
  bash scripts/do.sh "$@"
}

print_usage() {
  cat <<'EOF'
usage: ./local.sh [setup|run|status|smoke|smoke-real]

linux quick start:
  sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip build-essential clamav clamav-daemon
  ./local.sh setup
  ./local.sh run

repo environment:
  ./local.sh setup creates or reuses ./.venv
  repo commands run inside that repo-local venv
  do not use sudo for repo commands

main pipeline commands:
  ./local.sh setup    # bootstrap deps and local env
  ./local.sh smoke    # quick collection sanity check
  ./local.sh smoke-real # tiny real HF + CUDA smoke on a tokenized GPU box
  ./local.sh run      # resumable collect + train pipeline
  ./local.sh status   # current pipeline and artifact summary

advanced aliases still work:
  init/bootstrap -> setup
  pipeline       -> run
  refresh        -> retrain
  autocollect    -> continuous
  quick          -> smoke
  doctor         -> check
  setup-full     -> bootstrap + full pipeline
  start          -> 4090 quality-first preset
EOF
}

case "$cmd" in
  setup|init|bootstrap)
    bash scripts/setup_local.sh
    ;;
  setup-full)
    bash scripts/setup_linux.sh
    ;;
  run|pipeline)
    run_do pipeline "$@"
    ;;
  smoke|quick|collect-fast)
    run_do smoke "$@"
    ;;
  smoke-real)
    run_do smoke-real "$@"
    ;;
  collect)
    run_do collect-diverse "$@"
    ;;
  collect-status|collection-status)
    run_do collection-status "$@"
    ;;
  train)
    run_do train-existing "$@"
    ;;
  retrain|refresh)
    run_do retrain "$@"
    ;;
  continuous|autocollect)
    run_do continuous "$@"
    ;;
  check|doctor)
    run_do doctor "$@"
    ;;
  start)
    run_do start "$@"
    ;;
  scan)
    run_do scan "$@"
    ;;
  deps-update)
    run_do deps-update
    ;;
  status)
    run_do status
    ;;
  help|*)
    print_usage
    ;;
esac
