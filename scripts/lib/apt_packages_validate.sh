#!/usr/bin/env bash
# Validate APT_PACKAGES tokens (Debian package names) before apt-get install to avoid
# shell command injection when package lists are expanded into a command line.

validate_apt_package_tokens_or_exit() {
  local raw="$1"
  local -a pkgs=()
  read -r -a pkgs <<< "$raw"
  local p
  for p in "${pkgs[@]}"; do
    [[ -n "$p" ]] || continue
    if [[ ! "$p" =~ ^[a-zA-Z0-9+.-]+$ ]]; then
      printf 'install_fail: invalid_apt_package_token token=%s\n' "$p" >&2
      return 1
    fi
  done
  return 0
}
