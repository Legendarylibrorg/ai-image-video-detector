#!/usr/bin/env bash
set -euo pipefail

# Minimal command surface for everyday use.
# Examples:
#   bash scripts/do.sh pipeline
#   bash scripts/do.sh run
#   bash scripts/do.sh smoke
#   bash scripts/do.sh check
#   bash scripts/do.sh start
#   bash scripts/do.sh start-v2
#   bash scripts/do.sh collect
#   bash scripts/do.sh collect-diverse
#   bash scripts/do.sh collect-fast
#   bash scripts/do.sh collect-image
#   bash scripts/do.sh collect-video
#   bash scripts/do.sh collection-status
#   bash scripts/do.sh ingest
#   bash scripts/do.sh scan
#   bash scripts/do.sh train
#   bash scripts/do.sh retrain
#   bash scripts/do.sh continuous
#   bash scripts/do.sh train-all-types
#   bash scripts/do.sh deps-update
#   bash scripts/do.sh doctor

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
TRAIN_LOCK="${TRAIN_LOCK:-$ROOT_DIR/.local/training.lock}"
ENV_FILE="${ENV_FILE:-$ROOT_DIR/.env}"
ENV_READY=0
PREPARED_IMAGE_DATA_DIR=""
PIPELINE_STAGE_DIR="${PIPELINE_STAGE_DIR:-$ROOT_DIR/.local/pipeline}"
PIPELINE_FORCE_STAGES="${PIPELINE_FORCE_STAGES:-0}"
PIPELINE_MAX_ATTEMPTS="${PIPELINE_MAX_ATTEMPTS:-4}"
PIPELINE_RETRY_SLEEP_SEC="${PIPELINE_RETRY_SLEEP_SEC:-45}"
PIPELINE_WAIT_FOR_TRAINING_SEC="${PIPELINE_WAIT_FOR_TRAINING_SEC:-600}"
BEST_HF_QUERY_CSV_DEFAULT="real camera photo dataset,smartphone photo dataset,dslr photo dataset,webcam image dataset,cctv frame image dataset,meme image real vs ai,captioned image real ai,screenshot dataset image,chat ui screenshot,browser screenshot image,dashboard screenshot dataset,image poster infographic,logo brand image dataset,advertisement creative image,receipt scanned document image,id card document image,invoice form document scan,anime illustration real fake,digital art illustration dataset,3d render real fake,cgi synthetic image real,game render frame dataset,watermarked social media image,recompressed image dataset,heavily edited real photo,low resolution blurry image,extreme aspect ratio image,portrait selfie real fake,group photo real fake,deepfake face swap image,diffusion generated image latest"
DIVERSE_HF_QUERY_CSV_DEFAULT="$BEST_HF_QUERY_CSV_DEFAULT"

source "$ROOT_DIR/scripts/lib/core.sh"
source "$ROOT_DIR/scripts/lib/collection.sh"
source "$ROOT_DIR/scripts/lib/training.sh"

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  return 0
fi

load_env_file
cmd="${1:-start}"
shift || true

case "$cmd" in
  pipeline|run)
    run_full_pipeline
    ;;

  smoke)
    run_simple_collection_smoke
    ;;

  check|doctor)
    run_doctor_check
    ;;

  start)
    # Full, best-quality pipeline.
    with_training_lock bash scripts/max_quality_4090.sh
    ;;

  start-v2)
    # Max-accuracy v2 pipeline with domain calibration + refinement loops.
    with_training_lock bash scripts/max_accuracy_v2.sh
    ;;

  collect)
    # Full collection cycle: image pull + new-output ingest + video pull.
    run_collection_command collect_full_cycle
    ;;

  collect-diverse)
    run_collection_command collect_diverse_cycle
    ;;

  collect-fast)
    run_collection_command collect_fast_data
    ;;

  collect-image)
    run_collection_command collect_image_data
    ;;

  collect-video)
    run_collection_command collect_video_data
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

  train|train-existing)
    # Train from collected image data already on disk, with video if available.
    with_training_lock train_existing_pipeline
    ;;

  train-image)
    # Image pipeline only, assumes data already collected.
    with_training_lock train_image_pipeline
    ;;

  train-video)
    with_training_lock train_video_only
    ;;

  deps-update)
    bash scripts/update_deps_lock.sh
    ;;

  train-all)
    # Image + video training, assumes data already collected.
    with_training_lock train_all_pipeline
    ;;

  retrain)
    wait_for_training_to_finish "retrain"
    bash scripts/local_retrain_4090.sh "$@"
    ;;

  continuous)
    bash scripts/continuous_training.sh "$@"
    ;;

  train-all-types)
    # End-to-end: broad collection + image/video training + artifact checks.
    run_full_pipeline
    ;;

  status)
    show_status
    ;;

  *)
    print_usage
    exit 2
    ;;
esac
