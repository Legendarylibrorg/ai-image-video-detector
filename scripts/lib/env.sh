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

load_env_file() {
  local env_file="${1:-${ENV_FILE:-}}"
  [[ -n "$env_file" && -f "$env_file" ]] || return 0

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
