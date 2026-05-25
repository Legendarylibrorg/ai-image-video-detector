parse_env_assignment() {
  local line="${1%$'\r'}"
  [[ "$line" =~ ^[[:space:]]*$ ]] && return 1
  [[ "$line" =~ ^[[:space:]]*# ]] && return 1
  [[ "$line" =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || return 1

  ENV_ASSIGN_NAME="${BASH_REMATCH[2]}"
  local raw_value="${BASH_REMATCH[3]}"
  local trimmed_leading="${raw_value#"${raw_value%%[![:space:]]*}"}"
  local parsed=""
  local char=""
  local prev_char=""
  local in_single=0
  local in_double=0
  local i=0

  for ((i = 0; i < ${#trimmed_leading}; i++)); do
    char="${trimmed_leading:i:1}"
    if [[ "$char" == "'" && "$in_double" == "0" ]]; then
      if [[ "$in_single" == "1" ]]; then
        in_single=0
      else
        in_single=1
      fi
      parsed+="$char"
      prev_char="$char"
      continue
    fi
    if [[ "$char" == '"' && "$in_single" == "0" ]]; then
      if [[ "$in_double" == "1" ]]; then
        in_double=0
      else
        in_double=1
      fi
      parsed+="$char"
      prev_char="$char"
      continue
    fi
    if [[ "$char" == "#" && "$in_single" == "0" && "$in_double" == "0" ]]; then
      if [[ -z "$parsed" || "$prev_char" =~ [[:space:]] ]]; then
        break
      fi
    fi
    parsed+="$char"
    prev_char="$char"
  done

  parsed="${parsed%"${parsed##*[![:space:]]}"}"
  ENV_ASSIGN_VALUE="$parsed"

  if [[ "$ENV_ASSIGN_VALUE" =~ ^\'(.*)\'$ ]]; then
    ENV_ASSIGN_VALUE="${BASH_REMATCH[1]}"
  elif [[ "$ENV_ASSIGN_VALUE" =~ ^\"(.*)\"$ ]]; then
    ENV_ASSIGN_VALUE="${BASH_REMATCH[1]}"
  fi
  return 0
}

set_env_var_literal() {
  local name="$1"
  local value="$2"
  printf -v "$name" '%s' "$value"
  export "$name"
}

normalize_hf_token_env_aliases() {
  local token="${HF_TOKEN:-}"
  if [[ -z "$token" && -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    token="$HUGGING_FACE_HUB_TOKEN"
  fi
  if [[ -z "$token" && -n "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
    token="$HUGGINGFACE_HUB_TOKEN"
  fi
  if [[ -n "$token" ]]; then
    export HF_TOKEN="$token"
  fi
}

trim_env_value() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf "%s\n" "$value"
}

read_deps_profile_file() {
  local profile_file="${1:-}"
  [[ -n "$profile_file" && -f "$profile_file" ]] || return 1
  local profile=""
  profile="$(tr -d '\r' < "$profile_file" | head -n 1)"
  profile="$(trim_env_value "$profile")"
  [[ -n "$profile" ]] || return 1
  printf "%s\n" "$profile"
}

resolve_deps_extra() {
  local explicit="${1:-}"
  local profile_file="${2:-}"
  if [[ -n "$explicit" ]]; then
    normalized_deps_extra "$explicit"
    return $?
  fi
  local stored=""
  stored="$(read_deps_profile_file "$profile_file" || true)"
  if [[ -n "$stored" ]]; then
    normalized_deps_extra "$stored"
    return $?
  fi
  printf "pipeline\n"
}

# Must match ``scripts/deps_profile.py`` ``ALLOWED_DEPS_EXTRAS`` / ``pyproject.toml``
# ``[project.optional-dependencies]`` (see ``tests/test_dependency_metadata_surface.py``).
deps_extra_token_allowed() {
  case "$1" in
    pipeline|training|collection|video|inference) return 0 ;;
    *) return 1 ;;
  esac
}

normalized_deps_extra() {
  local extras_csv="${1:-pipeline}"
  local extra=""
  local trimmed=""
  local -a extras=()
  local -a normalized=()
  IFS=',' read -r -a extras <<< "$extras_csv"
  for extra in "${extras[@]}"; do
    trimmed="$(trim_env_value "$extra")"
    [[ -n "$trimmed" ]] || continue
    if ! deps_extra_token_allowed "$trimmed"; then
      echo "deps_fail=invalid_deps_extra_token token=${trimmed} allowed=pipeline,training,collection,video,inference" >&2
      return 1
    fi
    if [[ "$trimmed" == "pipeline" ]]; then
      printf "pipeline\n"
      return 0
    fi
    normalized+=("$trimmed")
  done
  if [[ ${#normalized[@]} -eq 0 ]]; then
    printf "pipeline\n"
    return 0
  fi
  printf '%s\n' "${normalized[@]}" | awk 'NF && !seen[$0]++' | paste -sd, -
}

deps_extra_enabled() {
  local wanted="$1"
  local extras_csv=""
  extras_csv="$(normalized_deps_extra "${2:-pipeline}")"
  local extra=""
  local trimmed=""
  local -a extras=()
  IFS=',' read -r -a extras <<< "$extras_csv"
  for extra in "${extras[@]}"; do
    trimmed="$(trim_env_value "$extra")"
    if [[ "$trimmed" == "pipeline" || "$trimmed" == "$wanted" ]]; then
      return 0
    fi
  done
  return 1
}

deps_extra_env_prefix() {
  local deps_extra=""
  deps_extra="$(normalized_deps_extra "${1:-pipeline}")"
  if [[ "$deps_extra" == "pipeline" ]]; then
    printf "%s" ""
    return 0
  fi
  printf "env DEPS_EXTRA=%q " "$deps_extra"
}

deps_extra_profile_tag() {
  local deps_extra=""
  deps_extra="$(normalized_deps_extra "${1:-pipeline}")"
  printf '%s\n' "$(printf '%s' "$deps_extra" | tr ',/' '__')"
}

deps_extra_install_target() {
  local deps_extra=""
  deps_extra="$(normalized_deps_extra "${1:-pipeline}")"
  printf '.[%s]\n' "$deps_extra"
}

deps_profile_file() {
  printf '%s\n' "${DEPS_PROFILE_FILE:-${VENV_DIR:-$ROOT_DIR/.venv}/.deps_profile}"
}

resolved_deps_extra() {
  resolve_deps_extra "${DEPS_EXTRA:-}" "$(deps_profile_file)"
}

deps_install_command() {
  local deps_extra=""
  deps_extra="$(normalized_deps_extra "${1:-$(resolved_deps_extra)}")"
  printf '%sbash scripts/install_deps.sh\n' "$(deps_extra_env_prefix "$deps_extra")"
}

run_deps_install() {
  local deps_extra=""
  deps_extra="$(normalized_deps_extra "${1:-$(resolved_deps_extra)}")"
  if [[ "$deps_extra" == "pipeline" ]]; then
    bash scripts/install_deps.sh
    return 0
  fi
  DEPS_EXTRA="$deps_extra" bash scripts/install_deps.sh
}

load_env_file() {
  local env_file="${1:-${ENV_FILE:-}}"
  if [[ -n "$env_file" && -f "$env_file" ]]; then
    local line=""
    while IFS= read -r line || [[ -n "$line" ]]; do
      if ! parse_env_assignment "$line"; then
        continue
      fi
      if [[ "${!ENV_ASSIGN_NAME+x}" == "x" && -n "${!ENV_ASSIGN_NAME}" ]]; then
        continue
      fi
      set_env_var_literal "$ENV_ASSIGN_NAME" "$ENV_ASSIGN_VALUE"
    done < "$env_file"
  fi
  normalize_hf_token_env_aliases
}

hf_token_path() {
  if [[ -n "${HF_TOKEN_PATH:-}" ]]; then
    printf "%s\n" "$HF_TOKEN_PATH"
    return
  fi
  if [[ -n "${HF_HOME:-}" ]]; then
    printf "%s/token\n" "$HF_HOME"
    return
  fi
  if [[ -n "${XDG_CACHE_HOME:-}" ]]; then
    printf "%s/huggingface/token\n" "$XDG_CACHE_HOME"
    return
  fi
  if [[ -n "${HOME:-}" ]]; then
    printf "%s/.cache/huggingface/token\n" "$HOME"
    return
  fi
  printf "\n"
}

read_hf_token_file() {
  local token_path=""
  token_path="$(hf_token_path)"
  [[ -n "$token_path" && -f "$token_path" ]] || return 1

  local token=""
  IFS= read -r token < "$token_path" || true
  token="${token#"${token%%[![:space:]]*}"}"
  token="${token%"${token##*[![:space:]]}"}"
  [[ -n "$token" ]] || return 1
  printf "%s\n" "$token"
}

CURRENT_HF_TOKEN=""
CURRENT_HF_TOKEN_SOURCE=""

resolve_current_hf_token() {
  CURRENT_HF_TOKEN=""
  CURRENT_HF_TOKEN_SOURCE=""
  normalize_hf_token_env_aliases
  if [[ -n "${HF_TOKEN:-}" ]]; then
    CURRENT_HF_TOKEN="$HF_TOKEN"
    CURRENT_HF_TOKEN_SOURCE="env"
    return 0
  fi
  local token=""
  token="$(read_hf_token_file || true)"
  if [[ -n "$token" ]]; then
    CURRENT_HF_TOKEN="$token"
    CURRENT_HF_TOKEN_SOURCE="hf_token_file"
  fi
  return 0
}

set_hf_token_vars() {
  local token="$1"
  export HF_TOKEN="$token"
  export HUGGING_FACE_HUB_TOKEN="$token"
  export HUGGINGFACE_HUB_TOKEN="$token"
}

save_hf_token_env() {
  local token="$1"
  local env_file="${2:-${ENV_FILE:-}}"
  [[ -n "$env_file" ]] || return 1
  if [[ "$token" == *$'\n'* || "$token" == *"'"* ]]; then
    echo "hf_token_save_unsupported_chars=1" >&2
    return 1
  fi

  mkdir -p "$(dirname "$env_file")"
  local line=""
  local found=0
  local tmp_file="${env_file}.tmp"
  if [[ -f "$env_file" ]]; then
    : > "$tmp_file"
    while IFS= read -r line || [[ -n "$line" ]]; do
      if parse_env_assignment "$line" && [[ "$ENV_ASSIGN_NAME" == "HF_TOKEN" ]]; then
        printf "HF_TOKEN='%s'\n" "$token" >> "$tmp_file"
        found=1
      else
        printf "%s\n" "$line" >> "$tmp_file"
      fi
    done < "$env_file"
    if [[ "$found" != "1" ]]; then
      printf "\nHF_TOKEN='%s'\n" "$token" >> "$tmp_file"
    fi
    mv "$tmp_file" "$env_file"
    return 0
  fi
  printf "HF_TOKEN='%s'\n" "$token" > "$env_file"
}

ensure_env_file() {
  local env_file="${1:-${ENV_FILE:-}}"
  local env_example_file="${2:-${ENV_EXAMPLE_FILE:-}}"
  [[ -n "$env_file" ]] || return 1

  if [[ -f "$env_file" ]]; then
    echo "setup_stage=env_file status=skip_exists file=$env_file"
    return 0
  fi
  if [[ -z "$env_example_file" || ! -f "$env_example_file" ]]; then
    echo "setup_stage=env_file status=skip_missing_example file=$env_example_file"
    return 0
  fi
  cp "$env_example_file" "$env_file"
  echo "setup_stage=env_file status=done file=$env_file"
}
