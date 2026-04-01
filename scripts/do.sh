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
BEST_HF_QUERY_CSV_DEFAULT="real camera photo dataset,smartphone photo dataset,dslr photo dataset,webcam image dataset,cctv frame image dataset,portrait selfie real fake,group photo real fake,indoor room photo dataset,outdoor landscape photo dataset,product photo dataset,food photo dataset,animal photo dataset,night photo dataset,macro close up photo dataset,panorama photo dataset,high resolution photo dataset,low resolution image dataset,street photo dataset,travel photo dataset,architecture photo dataset,fashion photo dataset,sports action photo dataset,vehicle road photo dataset,drone aerial photo dataset,satellite image dataset,meme image real vs ai,captioned image real ai,screenshot dataset image,chat ui screenshot,browser screenshot image,dashboard screenshot dataset,mobile app screenshot image,website screenshot dataset,screen capture ui dataset,desktop screenshot dataset,tablet screenshot image,image poster infographic,logo brand image dataset,advertisement creative image,receipt scanned document image,id card document image,invoice form document scan,passport scan image,document camera capture dataset,newspaper scan image,textbook page image,old photo scan dataset,film scan photo dataset,raw photo dataset,anime illustration real fake,digital art illustration dataset,manga artwork dataset,comic panel image dataset,3d render real fake,cgi synthetic image real,game render frame dataset,watermarked social media image,recompressed image dataset,heavily edited real photo,jpeg photo dataset,png image dataset,webp image dataset,extreme aspect ratio image,deepfake face swap image,diffusion generated image latest,midjourney generated image dataset,dalle generated image dataset,flux generated image dataset,stable diffusion image dataset,stock photo real ai,image manipulation detection,synthetic portrait dataset"

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
