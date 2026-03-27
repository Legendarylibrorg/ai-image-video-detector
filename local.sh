#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

cmd="${1:-help}"
shift || true

APT_PACKAGES="${APT_PACKAGES:-curl ca-certificates git unzip python3 python3-venv python3-pip build-essential clamav clamav-daemon}"

run_do() {
  bash scripts/do.sh "$@"
}

print_usage() {
  cat <<EOF
usage: ./local.sh [setup|venv|deps|doctor|collect|run|status|smoke|smoke-real|collect-status|train|retrain|finetune|continuous]

one-line install:
  curl -fsSL https://raw.githubusercontent.com/Legendarylibrorg/ai-image-video-detector/main/install.sh | bash

basic linux commands inside the repo:
  1. sudo apt-get update
  2. sudo apt-get install -y $APT_PACKAGES
  3. sudo freshclam || true
  4. python3 -m venv .venv
  5. source .venv/bin/activate
  6. ./local.sh deps
  7. ./local.sh doctor
  8. printf "HF_TOKEN='your_token_here'\n" >> .env
  9. ./local.sh smoke
 10. ./local.sh run
 11. ./local.sh status

shortcut inside the repo:
  ./local.sh setup

repo dependency install:
  ./local.sh setup
  creates or reuses ./.venv and installs pinned Python deps
  installs the repo CLI commands and Hugging Face CLI in that venv
  does not pause to prompt for HF_TOKEN by default
  repo commands run inside that repo-local venv
  do not use sudo for repo commands

main pipeline commands:
  ./local.sh setup    # bootstrap deps and local env
  ./local.sh collect  # collect HF image/video data only
  ./local.sh run      # canonical HF collect + train pipeline
  ./local.sh retrain  # retrain on top of existing collected data
  ./local.sh finetune # metadata-aware finetune on top of an existing checkpoint
  ./local.sh continuous # continuous collection + retraining loop
  ./local.sh status   # current pipeline and artifact summary

optional validation:
  ./local.sh smoke      # tiny local end-to-end pipeline smoke
  ./local.sh smoke-real # tiny real HF + CUDA smoke on a tokenized GPU box

troubleshooting:
  ./local.sh collect        # collection only, no training
  ./local.sh collect-status # current dataset build and resume state
  ./local.sh train          # train only from data already collected
  ./local.sh retrain        # retrain with gating on existing data
  ./local.sh finetune       # finetune an existing model with metadata cues
  ./local.sh continuous     # repeat collection + retraining over time
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
  setup)
    SETUP_RUN_PIPELINE=0 bash scripts/setup_linux.sh
    ;;
  venv)
    create_repo_venv
    ;;
  deps)
    bash scripts/install_deps.sh
    ;;
  collect)
    run_do collect
    ;;
  run)
    run_do pipeline "$@"
    ;;
  smoke)
    run_do smoke "$@"
    ;;
  smoke-real)
    run_do smoke-real "$@"
    ;;
  collect-status)
    run_do collection-status "$@"
    ;;
  train)
    run_do train-existing "$@"
    ;;
  retrain)
    run_do retrain "$@"
    ;;
  finetune)
    run_do finetune "$@"
    ;;
  continuous)
    run_do continuous "$@"
    ;;
  doctor)
    run_do doctor "$@"
    ;;
  status)
    run_do status
    ;;
  help|*)
    print_usage
    ;;
esac
