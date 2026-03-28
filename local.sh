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
usage: ./local.sh [setup|deps|doctor|docker-doctor|collect|run|status|smoke|smoke-real|collect-status|train|retrain|finetune|continuous]

Linux bash commands:

Docker-first:
  docker compose build
  ./local.sh docker-doctor
  docker compose run --rm pipeline ./local.sh deps
  docker compose run --rm pipeline ./local.sh doctor
  docker compose run --rm pipeline-gpu ./local.sh doctor
  docker compose run --rm pipeline-gpu ./local.sh smoke
  docker compose run --rm pipeline-gpu ./local.sh run
  docker compose run --rm pipeline-gpu ./local.sh status
  container deps live in /opt/aid-venv
  Hugging Face caches live under ./.local/hf and are reused by Compose

one-line install:
  curl -fsSL https://raw.githubusercontent.com/Legendarylibrorg/ai-image-video-detector/main/install.sh | bash

Native Linux:
  sudo apt-get update
  sudo apt-get install -y $APT_PACKAGES
  sudo freshclam || true
  ./local.sh setup
  printf "HF_TOKEN='your_token_here'\n" >> .env
  ./local.sh smoke
  ./local.sh run
  ./local.sh status
  setup creates or reuses ./.venv, installs pinned deps, and runs doctor
  do not use sudo for repo commands

Main commands:
  ./local.sh setup    # bootstrap deps and local env
  ./local.sh docker-doctor # verify docker, compose, and repo docker files
  ./local.sh collect  # collect HF image/video data only
  ./local.sh run      # canonical HF collect + train pipeline
  ./local.sh train    # train only from data already collected
  ./local.sh retrain  # retrain on top of existing collected data
  ./local.sh finetune # metadata-aware finetune on top of an existing checkpoint
  ./local.sh continuous # continuous collection + retraining loop
  ./local.sh status   # current pipeline and artifact summary
  ./local.sh collect-status # current dataset build and resume state

Validation:
  ./local.sh smoke      # tiny local end-to-end pipeline smoke
  ./local.sh smoke-real # tiny real HF + CUDA smoke on a tokenized GPU box
EOF
}

case "$cmd" in
  setup)
    SETUP_RUN_PIPELINE=0 bash scripts/setup_linux.sh
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
  docker-doctor)
    DOCTOR_REQUIRE_DOCKER=1 bash scripts/doctor.sh "$@"
    ;;
  status)
    run_do status
    ;;
  help|*)
    print_usage
    ;;
esac
