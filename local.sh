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
usage: ./local.sh [setup|deps|doctor|run|status|smoke|smoke-real]

linux setup:
  1. sudo apt-get update
  2. sudo apt-get install -y python3 python3-venv python3-pip build-essential clamav clamav-daemon
  3. sudo freshclam || true
  4. ./local.sh setup
  5. ./local.sh run
  6. ./local.sh status

repo dependency install:
  ./local.sh setup
  creates or reuses ./.venv and installs pinned Python deps
  repo commands run inside that repo-local venv
  do not use sudo for repo commands

manual linux fallback:
  python3 -m venv .venv
  ./local.sh deps
  ./local.sh doctor
  printf "HF_TOKEN='your_token_here'\n" >> .env
  ./local.sh smoke
  ./local.sh run
  ./local.sh status

main pipeline commands:
  ./local.sh setup    # bootstrap deps and local env
  ./local.sh run      # resumable collect + train pipeline
  ./local.sh status   # current pipeline and artifact summary

optional validation:
  ./local.sh smoke      # quick collection sanity check
  ./local.sh smoke-real # tiny real HF + CUDA smoke on a tokenized GPU box
EOF
}

case "$cmd" in
  setup|init|bootstrap)
    SETUP_RUN_PIPELINE=0 bash scripts/setup_linux.sh
    ;;
  deps)
    bash scripts/install_deps.sh
    ;;
  setup-full)
    SETUP_RUN_PIPELINE=1 bash scripts/setup_linux.sh
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
