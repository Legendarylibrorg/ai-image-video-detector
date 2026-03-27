from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any


def read_json_dict(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        return {}
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def write_json_dict(path: str | Path, payload: dict[str, Any], *, indent: int = 2) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=indent), encoding="utf-8")


def read_nonempty_lines(path: str | Path) -> list[str]:
    target = Path(path)
    if not target.exists():
        return []
    return [line.strip() for line in target.read_text(encoding="utf-8").splitlines() if line.strip()]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"
