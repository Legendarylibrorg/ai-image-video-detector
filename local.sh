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
usage: ./local.sh [setup|run|smoke|collect|train|retrain|continuous|check|status|scan [paths...]|deps-update]

recommended:
  ./local.sh setup    # bootstrap deps and local env
  ./local.sh smoke    # quick collection sanity check
  ./local.sh run      # resumable collect + train pipeline
  ./local.sh train    # train on collected data already on disk
  ./local.sh retrain  # merge fresh data and benchmark-gate a retrain
  ./local.sh continuous # continuous collect + retrain loop
  ./local.sh check    # preflight validation

aliases:
  init/bootstrap -> setup
  pipeline       -> run
  refresh        -> retrain
  autocollect    -> continuous
  quick          -> smoke
  doctor         -> check
  setup-full     -> bootstrap + full pipeline
  start          -> advanced quality-first preset
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
  collect)
    run_do collect-diverse "$@"
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
