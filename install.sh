#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Supply-chain: default clone follows the repo default branch (a moving target). Prefer
# INSTALL_REV set to a release tag or known-good commit (see docs/STARTUP.md).
if [ -z "${INSTALL_REV:-}" ]; then
  printf '%s\n' "install_security_notice: INSTALL_REV is unset; clone will use the repository default branch. For a pinned checkout set INSTALL_REV to a tag or branch name (example: export INSTALL_REV=v0.1.0)." >&2
fi

# shellcheck source=scripts/lib/apt_packages_validate.sh
source "$SCRIPT_DIR/scripts/lib/apt_packages_validate.sh"

REPO_URL="${REPO_URL:-https://github.com/Legendarylibrorg/ai-image-video-detector.git}"
INSTALL_DIR="${INSTALL_DIR:-$PWD/ai-image-video-detector}"
INSTALL_SYSTEM_DEPS="${INSTALL_SYSTEM_DEPS:-1}"
DRY_RUN="${DRY_RUN:-0}"
INSTALL_ASSUME_LINUX="${INSTALL_ASSUME_LINUX:-0}"
INSTALL_ALLOW_CUSTOM_REPO="${INSTALL_ALLOW_CUSTOM_REPO:-0}"
APT_PACKAGES="${APT_PACKAGES:-curl ca-certificates git python3 python3-venv python3-pip build-essential clamav clamav-daemon}"
ROOT_DIR=""
CALLER_DIR="$PWD"

display_path() {
  python3 - "$CALLER_DIR" "$1" <<'PY'
import os
import sys

base = os.path.realpath(sys.argv[1])
target = os.path.realpath(sys.argv[2])
print(os.path.relpath(target, base))
PY
}

run_cmd() {
  if [ "$DRY_RUN" = "1" ]; then
    printf '[DRY_RUN]'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

validate_clone_parameters_or_exit() {
  if [ -n "${INSTALL_REV:-}" ]; then
    python3 "$SCRIPT_DIR/scripts/lib/install_validate.py" \
      --install-dir "$INSTALL_DIR" \
      --repo-url "$REPO_URL" \
      --allow-custom-repo "$INSTALL_ALLOW_CUSTOM_REPO" \
      --install-rev "$INSTALL_REV"
  else
    python3 "$SCRIPT_DIR/scripts/lib/install_validate.py" \
      --install-dir "$INSTALL_DIR" \
      --repo-url "$REPO_URL" \
      --allow-custom-repo "$INSTALL_ALLOW_CUSTOM_REPO"
  fi
}

run_repo_cmd() {
  local display_root
  display_root="$(display_path "$ROOT_DIR")"
  if [ "$DRY_RUN" = "1" ]; then
    if [ "$display_root" = "." ]; then
      printf '[DRY_RUN]'
      printf ' %q' "$@"
      printf '\n'
    else
      printf '[DRY_RUN] cd %q &&' "$display_root"
      printf ' %q' "$@"
      printf '\n'
    fi
  else
    (cd "$ROOT_DIR" && "$@")
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

run_apt_install() {
  validate_apt_package_tokens_or_exit "$APT_PACKAGES" || exit 1
  local -a apt_packages=()
  read -r -a apt_packages <<< "$APT_PACKAGES"
  if command -v sudo >/dev/null 2>&1; then
    run_cmd sudo apt-get update
    run_cmd sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y "${apt_packages[@]}"
    run_cmd sudo sh -c "freshclam || true"
  else
    run_cmd apt-get update
    run_cmd env DEBIAN_FRONTEND=noninteractive apt-get install -y "${apt_packages[@]}"
    run_cmd sh -c "freshclam || true"
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
  run_apt_install
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
  validate_clone_parameters_or_exit
  if [ "$DRY_RUN" = "1" ]; then
    if [ -n "${INSTALL_REV:-}" ]; then
      printf '[DRY_RUN] git clone --depth 1 --branch %q %q %q\n' "$INSTALL_REV" "$REPO_URL" "$INSTALL_DIR"
    else
      printf '[DRY_RUN] git clone --depth 1 %q %q\n' "$REPO_URL" "$INSTALL_DIR"
    fi
    ROOT_DIR="$INSTALL_DIR"
  else
    if [ -n "${INSTALL_REV:-}" ]; then
      git clone --depth 1 --branch "$INSTALL_REV" "$REPO_URL" "$INSTALL_DIR"
    else
      git clone --depth 1 "$REPO_URL" "$INSTALL_DIR"
    fi
    ROOT_DIR=$(cd "$INSTALL_DIR" && pwd)
  fi
  display_root="$(display_path "$ROOT_DIR")"
  printf 'install_stage=repo status=cloned repo=%s\n' "$display_root"
}

run_repo_setup() {
  printf 'install_stage=setup status=run\n'
  run_repo_cmd env SETUP_INSTALL_SYSTEM_DEPS=0 ./local.sh setup
  printf 'install_stage=setup status=done\n'
}

print_next_steps() {
  local display_root
  display_root="$(display_path "$ROOT_DIR")"
  printf 'install_status=ready repo=%s\n' "$display_root"
  printf 'next_steps:\n'
  if [ "$display_root" != "." ]; then
    printf '  cd %q\n' "$display_root"
  fi
  printf "  printf \"HF_TOKEN='your_token_here'\\\\n\" >> .env\n"
  printf '  ./local.sh smoke\n'
  printf '  ./local.sh run\n'
  printf '  ./local.sh status\n'
  printf '  (optional) source .venv/bin/activate   # ad-hoc python/pip; ./local.sh already uses .venv\n'
}

ensure_linux
install_system_deps
ensure_repo
run_repo_setup
print_next_steps
