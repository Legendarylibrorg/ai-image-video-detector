#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOCK_FILE="${LOCK_FILE:-$ROOT_DIR/requirements.lock}"
LOCK_VENV_DIR="${LOCK_VENV_DIR:-$ROOT_DIR/.venv.locktmp}"
TMP_FILE="${LOCK_FILE}.tmp"
KEEP_LOCK_VENV="${KEEP_LOCK_VENV:-0}"
rm -rf "$LOCK_VENV_DIR"
python3 -m venv "$LOCK_VENV_DIR"

# shellcheck disable=SC1091
source "$LOCK_VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel
pip install --upgrade --upgrade-strategy eager -e .
if ! python - <<'PY' >/dev/null 2>&1
import tomllib
PY
then
  python -m pip install tomli
fi
python -m pip check

# Keep the lock small and maintainable: pin only the project's direct runtime
# dependencies and let pip resolve their transitives during install.
python - <<'PY' "$ROOT_DIR/pyproject.toml" > "$TMP_FILE"
import importlib.metadata as md
from importlib.metadata import PackageNotFoundError
from pathlib import Path
import re
import sys
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib

project = tomllib.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
project_cfg = project.get("project", {})
deps = list(project_cfg.get("dependencies", []))
rows: list[str] = []

for dep in deps:
    spec = dep.split(";", 1)[0].strip()
    match = re.match(r"([A-Za-z0-9_.-]+)", spec)
    if not match:
        continue
    name = match.group(1)
    candidates = (name, name.replace("_", "-"), name.replace("-", "_"))
    version = None
    for candidate in candidates:
        try:
            version = md.version(candidate)
            break
        except PackageNotFoundError:
            continue
    if version is None:
        raise SystemExit(f"missing_installed_dependency={name}")
    rows.append(f"{name}=={version}")

for row in sorted(dict.fromkeys(rows), key=str.lower):
    print(row)
PY

mv "$TMP_FILE" "$LOCK_FILE"
if [[ "$KEEP_LOCK_VENV" != "1" ]]; then
  deactivate || true
  rm -rf "$LOCK_VENV_DIR"
fi
echo "deps_lock=updated file=$LOCK_FILE"
