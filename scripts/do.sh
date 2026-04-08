#!/usr/bin/env bash
set -euo pipefail

# Minimal command surface for everyday use.
# Examples:
#   bash scripts/do.sh pipeline
#   bash scripts/do.sh smoke
#   bash scripts/do.sh smoke-real
#   bash scripts/do.sh collect
#   bash scripts/do.sh collection-status
#   bash scripts/do.sh ingest
#   bash scripts/do.sh scan
#   bash scripts/do.sh retrain
#   bash scripts/do.sh continuous
#   bash scripts/do.sh doctor
#   bash scripts/do.sh preflight

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
TRAIN_LOCK="${TRAIN_LOCK:-$ROOT_DIR/.local/training.lock}"
ENV_FILE="${ENV_FILE:-$ROOT_DIR/.env}"
ENV_READY=0
PREPARED_IMAGE_DATA_DIR=""
PIPELINE_WAIT_FOR_TRAINING_SEC="${PIPELINE_WAIT_FOR_TRAINING_SEC:-600}"

source "$ROOT_DIR/scripts/lib/core.sh"
source "$ROOT_DIR/scripts/lib/collection.sh"
source "$ROOT_DIR/scripts/lib/training.sh"

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  return 0
fi

load_env_file
cmd="${1:-help}"
shift || true

case ",$GPU_REQUIRED_CMDS," in
  *,"$cmd",*)
    require_gpu_ready
    ;;
esac

case "$cmd" in
  pipeline)
    run_preflight_check
    run_full_pipeline
    ;;

  smoke)
    run_simple_collection_smoke
    ;;

  smoke-real)
    run_preflight_check
    bash scripts/smoke_real_stack.sh
    ;;

  preflight)
    run_preflight_check
    ;;

  doctor)
    run_doctor_check
    ;;

  collect)
    # Full collection cycle: image pull + new-output ingest + video pull.
    run_collection_command collect_full_cycle
    ;;

  collection-status)
    show_collection_status "$@"
    ;;

  ingest)
    run_collection_command ingest_outputs
    ;;

  scan)
    run_malware_scan "$@"
    ;;

  train-existing)
    # Train from collected image data already on disk, with video if available.
    with_training_lock train_existing_pipeline
    ;;

  retrain)
    wait_for_training_to_finish "retrain"
    run_retrain_pipeline
    ;;

  finetune)
    wait_for_training_to_finish "finetune"
    bash scripts/metadata_finetune_4090.sh "$@"
    ;;

  continuous)
    bash scripts/continuous_training.sh "$@"
    ;;

  status)
    show_status
    ;;

  *)
    print_usage
    exit 2
    ;;
esac
