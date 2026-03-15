#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

STATE_DIR="${STATE_DIR:-$ROOT_DIR/.local/runtime}"
PID_FILE="$STATE_DIR/worker.pid"
CHILD_PID_FILE="$STATE_DIR/child.pid"
LOG_FILE="${LOG_FILE:-$STATE_DIR/runner.log}"
STOP_FILE="$STATE_DIR/stop"
PAUSE_FILE="$STATE_DIR/pause"
TRAIN_DONE_FILE="$STATE_DIR/training_done.flag"

mkdir -p "$STATE_DIR"

is_running() {
  [[ -f "$PID_FILE" ]] || return 1
  local pid
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  [[ -n "$pid" ]] || return 1
  kill -0 "$pid" 2>/dev/null
}

start() {
  local mode="${1:-serve}"
  if [[ "$(uname -s)" != "Linux" ]]; then
    echo "linux_service requires Linux (found: $(uname -s))"
    exit 1
  fi
  if is_running; then
    echo "already_running pid=$(cat "$PID_FILE")"
    exit 0
  fi
  rm -f "$STOP_FILE"
  nohup env WORKER_MODE="$mode" bash scripts/linux_worker.sh >> "$LOG_FILE" 2>&1 &
  echo "$!" > "$PID_FILE"
  echo "started pid=$! mode=$mode log=$LOG_FILE"
}

pause() {
  touch "$PAUSE_FILE"
  if [[ -f "$CHILD_PID_FILE" ]]; then
    kill -TERM "$(cat "$CHILD_PID_FILE")" 2>/dev/null || true
  fi
  echo "paused"
}

resume() {
  rm -f "$PAUSE_FILE"
  echo "resumed"
}

stop() {
  touch "$STOP_FILE"
  if [[ -f "$CHILD_PID_FILE" ]]; then
    kill -TERM "$(cat "$CHILD_PID_FILE")" 2>/dev/null || true
  fi
  if is_running; then
    kill -TERM "$(cat "$PID_FILE")" 2>/dev/null || true
    sleep 1
    if is_running; then
      kill -KILL "$(cat "$PID_FILE")" 2>/dev/null || true
    fi
  fi
  rm -f "$PID_FILE" "$CHILD_PID_FILE"
  echo "stopped"
}

status() {
  if is_running; then
    echo "worker=running pid=$(cat "$PID_FILE")"
  else
    echo "worker=stopped"
  fi
  if [[ -f "$PAUSE_FILE" ]]; then
    echo "pause=on"
  else
    echo "pause=off"
  fi
  if [[ -f "$TRAIN_DONE_FILE" ]]; then
    echo "training_done=yes"
  else
    echo "training_done=no"
  fi
  echo "log=$LOG_FILE"
}

restart() {
  stop
  start
}

reset_training() {
  rm -f "$TRAIN_DONE_FILE"
  echo "training_reset"
}

logs() {
  touch "$LOG_FILE"
  tail -n "${TAIL_N:-120}" "$LOG_FILE"
}

cmd="${1:-start}"
case "$cmd" in
  start) start "serve" ;;
  full-start) start "full" ;;
  pause) pause ;;
  resume) resume ;;
  stop) stop ;;
  restart) restart ;;
  status) status ;;
  logs) logs ;;
  reset-training) reset_training ;;
  *)
    echo "usage: bash scripts/linux_service.sh [start|full-start|pause|resume|stop|restart|status|logs|reset-training]"
    exit 2
    ;;
esac
