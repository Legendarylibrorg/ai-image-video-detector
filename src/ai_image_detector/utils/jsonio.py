from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from typing import Any


def write_json_atomic(
    path: str | Path,
    payload: Any,
    *,
    indent: int = 2,
    sort_keys: bool = False,
) -> None:
    """Serialize JSON to ``path`` via a same-directory temp file and ``os.replace`` (crash-safe)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=indent, sort_keys=sort_keys)
    tmp = target.with_name(target.name + f".tmp.{os.getpid()}")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def read_json_dict(path: str | Path) -> dict[str, Any]:
    from ..io_limits import read_json_file_limited

    try:
        return read_json_file_limited(path)
    except Exception:
        return {}


def write_json_dict(path: str | Path, payload: dict[str, Any], *, indent: int = 2) -> None:
    """Write a JSON object; uses atomic replace so readers never see a half-written file."""
    write_json_atomic(path, payload, indent=indent, sort_keys=False)


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
