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
    """Load a JSON object from ``path`` using ``read_json_file_limited`` (size/symlink guards).

    Missing or zero-byte files return ``{}``. Invalid UTF-8, non-JSON, JSON that is not
    a single object, oversize files, or symlink leaves propagate the same errors as
    ``read_json_file_limited`` (typically ``ValueError``).
    """
    from ..io_limits import read_json_file_limited

    return read_json_file_limited(path)


def write_json_dict(path: str | Path, payload: dict[str, Any], *, indent: int = 2) -> None:
    """Write a JSON object; uses atomic replace so readers never see a half-written file."""
    write_json_atomic(path, payload, indent=indent, sort_keys=False)


def read_nonempty_lines(path: str | Path) -> list[str]:
    """Read non-empty trimmed lines with byte and line caps (see ``io_limits`` env defaults).

    Only the first ``AID_MAX_NONEMPTY_LINES_FILE_BYTES`` of the file are read; the symlink
    leaf is rejected. Files larger than the byte cap therefore contribute lines only from
    that prefix (intentional DoS bound, not a full-file manifest guarantee).
    """
    from ..io_limits import (
        MAX_NONEMPTY_LINES_COUNT,
        MAX_NONEMPTY_LINES_FILE_BYTES,
        read_bytes_limited,
        reject_symlink,
    )

    target = Path(path)
    if not target.exists():
        return []
    reject_symlink(target)
    raw = read_bytes_limited(target, max_bytes=MAX_NONEMPTY_LINES_FILE_BYTES)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"invalid_utf8_text path={target}") from exc
    out: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        out.append(s)
        if len(out) > MAX_NONEMPTY_LINES_COUNT:
            raise ValueError(f"nonempty_lines_too_many path={target} max_lines={MAX_NONEMPTY_LINES_COUNT}")
    return out


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except (FileNotFoundError, subprocess.CalledProcessError, OSError):
        return "unknown"
