#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

STATE_DIR="${STATE_DIR:-$ROOT_DIR/.local/runtime}"
LOG_FILE="${LOG_FILE:-$STATE_DIR/runner.log}"
STOP_FILE="$STATE_DIR/stop"
PAUSE_FILE="$STATE_DIR/pause"
TRAIN_DONE_FILE="$STATE_DIR/training_done.flag"
CHILD_PID_FILE="$STATE_DIR/child.pid"
TRAIN_LOCK="${TRAIN_LOCK:-$ROOT_DIR/.local/training.lock}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
RETRY_SLEEP_SEC="${RETRY_SLEEP_SEC:-15}"
WORKER_MODE="${WORKER_MODE:-serve}"   # serve|full

mkdir -p "$STATE_DIR"

log() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG_FILE"
}

run_child() {
  "$@" &
  local child_pid=$!
  echo "$child_pid" > "$CHILD_PID_FILE"
  wait "$child_pid"
  local rc=$?
  rm -f "$CHILD_PID_FILE"
  return "$rc"
}

linux_setup() {
  if command -v apt-get >/dev/null 2>&1; then
    if command -v sudo >/dev/null 2>&1; then
      sudo apt-get update
      sudo apt-get install -y python3 python3-venv python3-pip build-essential
    else
      apt-get update
      apt-get install -y python3 python3-venv python3-pip build-essential
    fi
  fi
}

if [[ "$(uname -s)" != "Linux" ]]; then
  log "fatal platform=$(uname -s) expected=Linux"
  exit 1
fi

log "worker_start host=$HOST port=$PORT mode=$WORKER_MODE"
linux_setup

while true; do
  if [[ -f "$STOP_FILE" ]]; then
    log "worker_stop_requested"
    exit 0
  fi

  if [[ -f "$PAUSE_FILE" ]]; then
    log "worker_paused"
    sleep 5
    continue
  fi

  if [[ "$WORKER_MODE" == "full" && ! -f "$TRAIN_DONE_FILE" ]]; then
    log "phase=train_all_types start"
    if run_child bash scripts/do.sh train-all-types; then
      touch "$TRAIN_DONE_FILE"
      log "phase=train_all_types ok"
    else
      log "phase=train_all_types failed retry_in=${RETRY_SLEEP_SEC}s"
      sleep "$RETRY_SLEEP_SEC"
      continue
    fi
  fi

  if [[ -f "$TRAIN_LOCK" ]]; then
    log "training_lock_detected waiting"
    sleep 5
    continue
  fi

  log "phase=serve start"
  if run_child env HOST="$HOST" PORT="$PORT" bash scripts/do.sh serve; then
    log "phase=serve exited_cleanly restart_in=${RETRY_SLEEP_SEC}s"
  else
    log "phase=serve crashed restart_in=${RETRY_SLEEP_SEC}s"
  fi
  sleep "$RETRY_SLEEP_SEC"
done
