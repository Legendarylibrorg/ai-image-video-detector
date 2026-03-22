load_env_file() {
  local env_file="${1:-${ENV_FILE:-}}"
  [[ -n "$env_file" && -f "$env_file" ]] || return 0

  local line=""
  local name=""
  local restore_script=""
  while IFS= read -r line; do
    case "$line" in
      ''|\#*) continue ;;
    esac
    if [[ "$line" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
      name="${line%%=*}"
      if eval '[[ ${'"$name"'+x} && -n "${'"$name"'}" ]]'; then
        restore_script+="$(eval "printf '%s=%q\n' '$name' \"\${$name}\"")"$'\n'
      fi
    fi
  done < "$env_file"

  set -a
  # shellcheck disable=SC1090
  source "$env_file"
  set +a
  [[ -z "$restore_script" ]] || eval "$restore_script"
}

current_hf_token() {
  if [[ -n "${HF_TOKEN:-}" ]]; then
    printf "%s\n" "$HF_TOKEN"
    return
  fi
  if [[ -n "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
    printf "%s\n" "$HUGGINGFACE_HUB_TOKEN"
  fi
}

set_hf_token_vars() {
  local token="$1"
  export HF_TOKEN="$token"
  export HUGGINGFACE_HUB_TOKEN="$token"
}

save_hf_token_env() {
  local token="$1"
  local env_file="${2:-${ENV_FILE:-}}"
  [[ -n "$env_file" ]] || return 1

  mkdir -p "$(dirname "$env_file")"
  if [[ -f "$env_file" ]]; then
    if grep -q '^HF_TOKEN=' "$env_file"; then
      sed -i.bak "s|^HF_TOKEN=.*$|HF_TOKEN='$token'|" "$env_file"
      rm -f "${env_file}.bak"
    else
      printf "\nHF_TOKEN='%s'\n" "$token" >> "$env_file"
    fi
  else
    printf "HF_TOKEN='%s'\n" "$token" > "$env_file"
  fi
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
