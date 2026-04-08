#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

source "$ROOT_DIR/scripts/lib/env.sh"

cmd="${1:-help}"
shift || true

APT_PACKAGES="${APT_PACKAGES:-curl ca-certificates git python3 python3-venv python3-pip build-essential clamav clamav-daemon}"

run_do() {
  bash scripts/do.sh "$@"
}

print_usage() {
  cat <<EOF
usage: ./local.sh [setup|deps|doctor|docker-doctor|collect|run|train|retrain|finetune|continuous|status|collect-status|smoke|smoke-real]

Full command map and stage meanings: docs/COMMANDS.md
Docker VM walkthrough and native Linux bootstrap: docs/STARTUP.md

Typical native path:
  sudo apt-get update
  sudo apt-get install -y curl ca-certificates git python3 python3-venv python3-pip build-essential clamav clamav-daemon
  sudo freshclam || true
  ./local.sh setup
  printf "HF_TOKEN='your_token_here'\n" >> .env
  ./local.sh smoke && ./local.sh run
  (no sudo for repo commands after system packages)

Need the source first on Linux?
  git clone https://github.com/Legendarylibrorg/ai-image-video-detector.git
  cd ai-image-video-detector
  or use curl + tar from docs/STARTUP.md
  or use: curl -fsSL https://raw.githubusercontent.com/Legendarylibrorg/ai-image-video-detector/main/install.sh | bash

Typical Compose path (repo root after clone):
  ./local.sh docker-doctor && docker compose build
  then CPU/GPU steps in docs/STARTUP.md (container venv: /opt/aid-venv, active HF/pip caches in named volumes under /workspace/.local)

One-line Linux install from upstream:
  curl -fsSL https://raw.githubusercontent.com/Legendarylibrorg/ai-image-video-detector/main/install.sh | bash
EOF
}

case "$cmd" in
  setup)
    SETUP_RUN_PIPELINE=0 bash scripts/setup_linux.sh
    ;;
  deps)
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      echo "[DRY_RUN] $(deps_install_command)"
      echo "deps_status=dry_run"
    else
      run_deps_install
    fi
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
