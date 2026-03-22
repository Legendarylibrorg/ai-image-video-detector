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

one-line install:
  curl -fsSL https://raw.githubusercontent.com/Legendarylibrorg/ai-image-video-detector/main/install.sh | bash

linux setup:
  fast path:
    ./local.sh setup
    printf "HF_TOKEN='your_token_here'\n" >> .env
    ./local.sh smoke
    ./local.sh run
    ./local.sh status

  step by step:
    1. sudo apt-get update
    2. sudo apt-get install -y curl ca-certificates git python3 python3-venv python3-pip build-essential clamav clamav-daemon
    3. sudo freshclam || true
    4. python3 -m venv .venv
    5. source .venv/bin/activate
    6. ./local.sh deps
    7. ./local.sh doctor
    8. printf "HF_TOKEN='your_token_here'\n" >> .env
    9. ./local.sh smoke
   10. ./local.sh run
   11. ./local.sh status

repo dependency install:
  ./local.sh setup
  creates or reuses ./.venv and installs pinned Python deps
  installs the repo CLI commands and Hugging Face CLI in that venv
  does not pause to prompt for HF_TOKEN by default
  repo commands run inside that repo-local venv
  do not use sudo for repo commands

main pipeline commands:
  ./local.sh setup    # bootstrap deps and local env
  ./local.sh run      # resumable collect + train pipeline
  ./local.sh status   # current pipeline and artifact summary

optional validation:
  ./local.sh smoke      # quick collection sanity check
  ./local.sh smoke-real # tiny real HF + CUDA smoke on a tokenized GPU box
EOF
}

create_repo_venv() {
  local root_dir
  local venv_dir
  root_dir="$(pwd)"
  venv_dir="${VENV_DIR:-$root_dir/.venv}"
  if [[ -x "$venv_dir/bin/python" ]]; then
    echo "venv_status=ready path=$venv_dir"
    return 0
  fi
  if ! command -v python3 >/dev/null 2>&1; then
    echo "venv_fail=python3_missing install_python3_and_retry=1" >&2
    return 1
  fi
  python3 -m venv "$venv_dir"
  echo "venv_status=created path=$venv_dir"
}

case "$cmd" in
  setup|init|bootstrap)
    SETUP_RUN_PIPELINE=0 bash scripts/setup_linux.sh
    ;;
  venv)
    create_repo_venv
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
