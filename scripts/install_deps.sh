#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source "$ROOT_DIR/scripts/lib/env.sh"

LOCK_FILE="${LOCK_FILE:-$ROOT_DIR/requirements.lock}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
UPGRADE_TOOLCHAIN="${UPGRADE_TOOLCHAIN:-0}"
TORCH_CUDA_INDEX_URL="${TORCH_CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
DEPS_PROFILE_FILE="${DEPS_PROFILE_FILE:-$VENV_DIR/.deps_profile}"
DEPS_EXTRA="$(normalized_deps_extra "${DEPS_EXTRA:-pipeline}")"
DEPS_PROFILE_TAG="$(deps_extra_profile_tag "$DEPS_EXTRA")"
DEPS_STAMP_FILE="${DEPS_STAMP_FILE:-$VENV_DIR/.deps_stamp.${DEPS_PROFILE_TAG}}"

hash_cmd() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum
    return
  fi
  shasum -a 256
}

pip_cmd() {
  local subcommand="$1"
  shift
  if [[ "$subcommand" == "install" ]]; then
    PIP_DISABLE_PIP_VERSION_CHECK=1 python -m pip install --progress-bar off "$@"
    return
  fi
  PIP_DISABLE_PIP_VERSION_CHECK=1 python -m pip "$subcommand" "$@"
}

deps_fingerprint() {
  {
    printf "deps_extra=%s\n" "$DEPS_EXTRA"
    if [[ -f "$LOCK_FILE" ]]; then
      cat "$LOCK_FILE"
    fi
    if [[ -f "$ROOT_DIR/pyproject.toml" ]]; then
      cat "$ROOT_DIR/pyproject.toml"
    fi
  } | hash_cmd | awk '{print $1}'
}

needs_torch_stack() {
  deps_extra_enabled inference "$DEPS_EXTRA" || deps_extra_enabled training "$DEPS_EXTRA"
}

toolchain_supports_editable_install() {
  python - <<'PY'
from importlib.metadata import PackageNotFoundError, version
import sys


def parse_version(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for chunk in value.replace("-", ".").split("."):
        digits = ""
        for ch in chunk:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def at_least(distribution: str, minimum: str) -> bool:
    try:
        current = version(distribution)
    except PackageNotFoundError:
        return False
    current_parts = parse_version(current)
    minimum_parts = parse_version(minimum)
    if len(current_parts) < len(minimum_parts):
        current_parts = current_parts + (0,) * (len(minimum_parts) - len(current_parts))
    return current_parts >= minimum_parts


sys.exit(0 if at_least("pip", "21.3") and at_least("setuptools", "68") else 1)
PY
}

ensure_packaging_toolchain() {
  if [[ "$UPGRADE_TOOLCHAIN" == "1" ]] || ! toolchain_supports_editable_install; then
    if ! pip_cmd install --quiet --retries 1 --timeout 15 --upgrade "pip>=21.3" "setuptools>=68" wheel; then
      echo "warning_toolchain_upgrade_failed using_existing_versions=1"
    fi
  fi
  if ! toolchain_supports_editable_install; then
    echo "deps_fail=packaging_toolchain_too_old run=$(repair_install_command)" >&2
    exit 1
  fi
}

repair_install_command() {
  printf '%sbash scripts/install_deps.sh' "$(deps_extra_env_prefix "$DEPS_EXTRA")"
}

selected_lock_package_names() {
  local -a names=()
  if needs_torch_stack; then
    names+=(numpy pillow safetensors)
  fi
  if deps_extra_enabled training "$DEPS_EXTRA"; then
    names+=(piexif scikit-learn)
  fi
  if deps_extra_enabled collection "$DEPS_EXTRA"; then
    names+=(datasets huggingface_hub pillow)
  fi
  if deps_extra_enabled video "$DEPS_EXTRA"; then
    names+=(opencv-python-headless)
  fi
  printf '%s\n' "${names[@]}" | awk 'NF && !seen[$0]++'
}

install_selected_locked_packages() {
  local -a names=()
  local name=""
  while IFS= read -r name; do
    [[ -n "$name" ]] || continue
    names+=("$name")
  done < <(selected_lock_package_names)

  if [[ ${#names[@]} -eq 0 ]]; then
    echo "deps_lock=subset extra=$DEPS_EXTRA packages=none"
    return 0
  fi

  local regex=""
  local tmp_lock=""
  local IFS='|'
  regex="^(${names[*]})=="
  tmp_lock="$(mktemp)"
  grep -E "$regex" "$LOCK_FILE" > "$tmp_lock" || true
  if [[ -s "$tmp_lock" ]]; then
    echo "deps_lock=subset extra=$DEPS_EXTRA packages=${names[*]}"
    pip_cmd install -r "$tmp_lock"
  else
    echo "deps_lock=subset_empty extra=$DEPS_EXTRA packages=${names[*]}"
  fi
  rm -f "$tmp_lock"
}

install_locked_torch_stack() {
  needs_torch_stack || return 0
  local torch_ver=""
  local tv_ver=""
  torch_ver="$(grep -E '^torch==' "$LOCK_FILE" | head -n1 | cut -d= -f3 || true)"
  tv_ver="$(grep -E '^torchvision==' "$LOCK_FILE" | head -n1 | cut -d= -f3 || true)"
  if [[ -z "$torch_ver" || -z "$tv_ver" ]]; then
    return 0
  fi
  if [[ "$(uname -s)" == "Linux" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    pip_cmd install "torch==$torch_ver" "torchvision==$tv_ver" --index-url "$TORCH_CUDA_INDEX_URL"
  else
    pip_cmd install "torch==$torch_ver" "torchvision==$tv_ver"
  fi
}

verify_python_deps() {
  python "$ROOT_DIR/scripts/deps_profile.py" --extras "$DEPS_EXTRA" --check-imports
}

venv_command_path() {
  printf '%s/bin/%s\n' "$VENV_DIR" "$1"
}

verify_required_commands() {
  if deps_extra_enabled collection "$DEPS_EXTRA" && [[ ! -x "$(venv_command_path hf)" ]]; then
    echo "deps_fail=huggingface_cli_missing extra=$DEPS_EXTRA run=$(repair_install_command)" >&2
    return 1
  fi
  if deps_extra_enabled training "$DEPS_EXTRA" && [[ ! -x "$(venv_command_path aid-train)" ]]; then
    echo "deps_fail=repo_cli_missing cli=aid-train extra=$DEPS_EXTRA run=$(repair_install_command)" >&2
    return 1
  fi
  if deps_extra_enabled training "$DEPS_EXTRA" && deps_extra_enabled video "$DEPS_EXTRA" && [[ ! -x "$(venv_command_path aid-video-train)" ]]; then
    echo "deps_fail=repo_cli_missing cli=aid-video-train extra=$DEPS_EXTRA run=$(repair_install_command)" >&2
    return 1
  fi
}

install_local_package() {
  pip_cmd install -e "$(deps_extra_install_target "$DEPS_EXTRA")" --no-deps --no-build-isolation
}

install_repo_cli_wrappers() {
  local train_wrapper="$VENV_DIR/bin/aid-train"
  local video_wrapper="$VENV_DIR/bin/aid-video-train"
  mkdir -p "$VENV_DIR/bin"
  if deps_extra_enabled training "$DEPS_EXTRA"; then
    cat > "$train_wrapper" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/python" -c 'from ai_image_detector.cli import train_main; raise SystemExit(train_main())' "$@"
EOF
    chmod +x "$train_wrapper"
  else
    rm -f "$train_wrapper"
  fi
  if deps_extra_enabled training "$DEPS_EXTRA" && deps_extra_enabled video "$DEPS_EXTRA"; then
    cat > "$video_wrapper" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/python" -c 'from ai_image_detector.cli import video_train_main; raise SystemExit(video_train_main())' "$@"
EOF
    chmod +x "$video_wrapper"
  else
    rm -f "$video_wrapper"
  fi
}

write_deps_profile_file() {
  printf "%s\n" "$DEPS_EXTRA" > "$DEPS_PROFILE_FILE"
}

ensure_virtualenv_ready() {
  if [[ -x "$VENV_DIR/bin/python" && -f "$VENV_DIR/bin/activate" ]]; then
    return 0
  fi
  python3 -m venv "$VENV_DIR"
}

ensure_virtualenv_ready

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
deps_fp="$(deps_fingerprint)"
if [[ "$UPGRADE_TOOLCHAIN" != "1" && -f "$DEPS_STAMP_FILE" && "$(cat "$DEPS_STAMP_FILE")" == "$deps_fp" ]]; then
  if verify_python_deps >/dev/null 2>&1 && verify_required_commands >/dev/null 2>&1; then
    install_repo_cli_wrappers
    write_deps_profile_file
    echo "deps_status=up_to_date"
    exit 0
  fi
fi

ensure_packaging_toolchain

if [[ -s "$LOCK_FILE" && "$DEPS_EXTRA" == "pipeline" ]]; then
  tmp_lock_no_torch="$(mktemp)"
  grep -Ev '^(torch==|torchvision==)' "$LOCK_FILE" > "$tmp_lock_no_torch"
  pip_cmd install -r "$tmp_lock_no_torch"
  rm -f "$tmp_lock_no_torch"

  install_locked_torch_stack

  # Install the local package without re-resolving dependencies from pyproject.
  install_local_package
else
  if [[ -s "$LOCK_FILE" ]]; then
    install_selected_locked_packages
    install_locked_torch_stack
    install_local_package
  else
    echo "deps_lock=missing file=$LOCK_FILE fallback=pyproject_resolve"
    pip_cmd install --upgrade --upgrade-strategy eager -e "$(deps_extra_install_target "$DEPS_EXTRA")"
  fi
fi

install_repo_cli_wrappers

if ! verify_python_deps >/dev/null 2>&1; then
  echo "deps_fail=core_python_deps_missing extra=$DEPS_EXTRA run=$(repair_install_command)" >&2
  exit 1
fi

if ! verify_required_commands; then
  exit 1
fi

pip_cmd check
write_deps_profile_file
printf "%s\n" "$deps_fp" > "$DEPS_STAMP_FILE"
