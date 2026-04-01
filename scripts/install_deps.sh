#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOCK_FILE="${LOCK_FILE:-$ROOT_DIR/requirements.lock}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
UPGRADE_TOOLCHAIN="${UPGRADE_TOOLCHAIN:-0}"
TORCH_CUDA_INDEX_URL="${TORCH_CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
DEPS_EXTRA="${DEPS_EXTRA:-pipeline}"
DEPS_PROFILE_TAG="$(printf '%s' "$DEPS_EXTRA" | tr ',/' '__')"
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

extra_enabled() {
  local wanted="$1"
  local extra=""
  local trimmed=""
  local -a extras=()
  IFS=',' read -r -a extras <<< "$DEPS_EXTRA"
  for extra in "${extras[@]}"; do
    trimmed="${extra#"${extra%%[![:space:]]*}"}"
    trimmed="${trimmed%"${trimmed##*[![:space:]]}"}"
    if [[ "$trimmed" == "pipeline" || "$trimmed" == "$wanted" ]]; then
      return 0
    fi
  done
  return 1
}

verify_python_deps() {
  python - "$DEPS_EXTRA" <<'PY'
import importlib
import sys

raw_extra = sys.argv[1]
extras = {item.strip() for item in raw_extra.split(",") if item.strip()}
if "pipeline" in extras:
    extras.update({"inference", "training", "collection", "video"})

modules = {"ai_image_detector"}
if "inference" in extras or "training" in extras:
    modules.update({"numpy", "PIL", "safetensors", "torch", "torchvision"})
if "training" in extras:
    modules.update({"piexif", "sklearn"})
if "collection" in extras:
    modules.update({"datasets", "huggingface_hub", "PIL"})
if "video" in extras:
    modules.add("cv2")

for name in sorted(modules):
    importlib.import_module(name)
PY
}

verify_required_commands() {
  if extra_enabled collection && ! command -v hf >/dev/null 2>&1; then
    echo "deps_fail=huggingface_cli_missing extra=$DEPS_EXTRA run=bash scripts/install_deps.sh" >&2
    return 1
  fi
  if extra_enabled training && ! command -v aid-train >/dev/null 2>&1; then
    echo "deps_fail=repo_cli_missing cli=aid-train extra=$DEPS_EXTRA run=bash scripts/install_deps.sh" >&2
    return 1
  fi
  if extra_enabled training && extra_enabled video && ! command -v aid-video-train >/dev/null 2>&1; then
    echo "deps_fail=repo_cli_missing cli=aid-video-train extra=$DEPS_EXTRA run=bash scripts/install_deps.sh" >&2
    return 1
  fi
}

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
deps_fp="$(deps_fingerprint)"
if [[ "$UPGRADE_TOOLCHAIN" != "1" && -f "$DEPS_STAMP_FILE" && "$(cat "$DEPS_STAMP_FILE")" == "$deps_fp" ]]; then
  if verify_python_deps >/dev/null 2>&1 && verify_required_commands >/dev/null 2>&1; then
    echo "deps_status=up_to_date"
    exit 0
  fi
fi

if [[ "$UPGRADE_TOOLCHAIN" == "1" ]]; then
  if ! pip_cmd install --quiet --retries 1 --timeout 15 --upgrade pip setuptools wheel; then
    echo "warning_toolchain_upgrade_failed using_existing_versions=1"
  fi
fi

if [[ -s "$LOCK_FILE" && "$DEPS_EXTRA" == "pipeline" ]]; then
  tmp_lock_no_torch="$(mktemp)"
  grep -Ev '^(torch==|torchvision==)' "$LOCK_FILE" > "$tmp_lock_no_torch"
  pip_cmd install -r "$tmp_lock_no_torch"
  rm -f "$tmp_lock_no_torch"

  torch_ver="$(grep -E '^torch==' "$LOCK_FILE" | head -n1 | cut -d= -f3 || true)"
  tv_ver="$(grep -E '^torchvision==' "$LOCK_FILE" | head -n1 | cut -d= -f3 || true)"
  if [[ -n "$torch_ver" && -n "$tv_ver" ]]; then
    if [[ "$(uname -s)" == "Linux" ]] && command -v nvidia-smi >/dev/null 2>&1; then
      pip_cmd install "torch==$torch_ver" "torchvision==$tv_ver" --index-url "$TORCH_CUDA_INDEX_URL"
    else
      pip_cmd install "torch==$torch_ver" "torchvision==$tv_ver"
    fi
  fi

  # Install the local package without re-resolving dependencies from pyproject.
  pip_cmd install -e . --no-deps --no-build-isolation
else
  if [[ -s "$LOCK_FILE" ]]; then
    echo "deps_lock=skip extra=$DEPS_EXTRA fallback=pyproject_resolve"
  else
    echo "deps_lock=missing file=$LOCK_FILE fallback=pyproject_resolve"
  fi
  pip_cmd install --upgrade --upgrade-strategy eager -e .
fi

if ! verify_python_deps >/dev/null 2>&1; then
  echo "deps_fail=core_python_deps_missing extra=$DEPS_EXTRA run=bash scripts/install_deps.sh" >&2
  exit 1
fi

if ! verify_required_commands; then
  exit 1
fi

pip_cmd check
printf "%s\n" "$deps_fp" > "$DEPS_STAMP_FILE"
