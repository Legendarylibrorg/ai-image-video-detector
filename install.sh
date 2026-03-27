#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/Legendarylibrorg/ai-image-video-detector.git}"
INSTALL_DIR="${INSTALL_DIR:-$PWD/ai-image-video-detector}"
INSTALL_SYSTEM_DEPS="${INSTALL_SYSTEM_DEPS:-1}"
DRY_RUN="${DRY_RUN:-0}"
INSTALL_ASSUME_LINUX="${INSTALL_ASSUME_LINUX:-0}"
APT_PACKAGES="${APT_PACKAGES:-curl ca-certificates git unzip python3 python3-venv python3-pip build-essential clamav clamav-daemon}"
ROOT_DIR=""
CALLER_DIR="$PWD"

display_path() {
  python3 - "$CALLER_DIR" "$1" <<'PY'
import os
import sys

base = sys.argv[1]
target = sys.argv[2]
print(os.path.relpath(target, base))
PY
}

run_cmd() {
  if [ "$DRY_RUN" = "1" ]; then
    printf '[DRY_RUN] %s\n' "$*"
  else
    bash -c "$*"
  fi
}

run_repo_cmd() {
  local display_root
  display_root="$(display_path "$ROOT_DIR")"
  if [ "$DRY_RUN" = "1" ]; then
    printf '[DRY_RUN] cd %q && %s\n' "$display_root" "$*"
  else
    (cd "$ROOT_DIR" && bash -c "$*")
  fi
}

ensure_linux() {
  if [ "$INSTALL_ASSUME_LINUX" = "1" ]; then
    return
  fi
  if [ "$(uname -s)" != "Linux" ]; then
    printf 'install_fail: linux_only\n' >&2
    exit 1
  fi
}

install_system_deps() {
  if [ "$INSTALL_SYSTEM_DEPS" != "1" ]; then
    printf 'install_stage=system_deps status=skip_opt_out\n'
    return
  fi
  if ! command -v apt-get >/dev/null 2>&1; then
    printf 'install_stage=system_deps status=skip_no_apt\n'
    return
  fi

  printf 'install_stage=system_deps status=run\n'
  if command -v sudo >/dev/null 2>&1; then
    run_cmd "sudo apt-get update"
    run_cmd "sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y $APT_PACKAGES"
    run_cmd "sudo freshclam || true"
  else
    run_cmd "apt-get update"
    run_cmd "env DEBIAN_FRONTEND=noninteractive apt-get install -y $APT_PACKAGES"
    run_cmd "freshclam || true"
  fi
  printf 'install_stage=system_deps status=done\n'
}

ensure_repo() {
  local display_root=""
  if [ -f "./local.sh" ] && [ -f "./scripts/install_deps.sh" ]; then
    ROOT_DIR=$(pwd)
    display_root="$(display_path "$ROOT_DIR")"
    printf 'install_stage=repo status=using_current repo=%s\n' "$display_root"
    return
  fi
  if [ -d "$INSTALL_DIR/.git" ]; then
    ROOT_DIR=$(cd "$INSTALL_DIR" && pwd)
    display_root="$(display_path "$ROOT_DIR")"
    printf 'install_stage=repo status=using_existing repo=%s\n' "$display_root"
    return
  fi
  if [ -f "$INSTALL_DIR/local.sh" ] && [ -f "$INSTALL_DIR/scripts/install_deps.sh" ]; then
    ROOT_DIR=$(cd "$INSTALL_DIR" && pwd)
    display_root="$(display_path "$ROOT_DIR")"
    printf 'install_stage=repo status=using_extracted repo=%s\n' "$display_root"
    return
  fi
  if [ -e "$INSTALL_DIR" ]; then
    printf 'install_fail: install_dir_exists_not_git path=%s\n' "$INSTALL_DIR" >&2
    exit 1
  fi
  if ! command -v git >/dev/null 2>&1; then
    printf 'install_fail: git_missing install git and retry\n' >&2
    exit 1
  fi
  run_cmd "git clone --depth 1 \"$REPO_URL\" \"$INSTALL_DIR\""
  if [ "$DRY_RUN" = "1" ]; then
    ROOT_DIR="$INSTALL_DIR"
  else
    ROOT_DIR=$(cd "$INSTALL_DIR" && pwd)
  fi
  display_root="$(display_path "$ROOT_DIR")"
  printf 'install_stage=repo status=cloned repo=%s\n' "$display_root"
}

ensure_venv() {
  local display_root
  display_root="$(display_path "$ROOT_DIR")"
  if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    printf 'install_stage=venv status=skip_exists path=%s/.venv\n' "$display_root"
    return
  fi
  printf 'install_stage=venv status=run\n'
  run_repo_cmd "python3 -m venv .venv"
  printf 'install_stage=venv status=done path=%s/.venv\n' "$display_root"
}

install_repo_deps() {
  printf 'install_stage=deps status=run\n'
  run_repo_cmd "./local.sh deps"
  printf 'install_stage=deps status=done\n'
}

run_repo_doctor() {
  printf 'install_stage=doctor status=run\n'
  run_repo_cmd "./local.sh doctor"
  printf 'install_stage=doctor status=done\n'
}

print_next_steps() {
  local display_root
  display_root="$(display_path "$ROOT_DIR")"
  printf 'install_status=ready repo=%s\n' "$display_root"
  printf 'next_steps:\n'
  printf '  cd %q\n' "$display_root"
  printf '  source .venv/bin/activate\n'
  printf "  printf \"HF_TOKEN='your_token_here'\\\\n\" >> .env\n"
  printf '  ./local.sh smoke\n'
  printf '  ./local.sh run\n'
  printf '  ./local.sh status\n'
}

ensure_linux
install_system_deps
ensure_repo
ensure_venv
install_repo_deps
run_repo_doctor
print_next_steps
