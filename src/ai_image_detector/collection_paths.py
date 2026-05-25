from __future__ import annotations

"""Path confinement for local dataset collection (ingest, HF caches, outputs)."""

import os
from pathlib import Path
from typing import Any

from .io_limits import reject_symlink


def collection_workspace_root() -> Path:
    """Directory that all collection inputs and outputs must stay under.

    Defaults to the process current working directory (repo root in normal runs).
    Override with ``AID_WORKSPACE_ROOT`` for stricter deployments.
    """
    raw = (os.environ.get("AID_WORKSPACE_ROOT") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return Path.cwd().resolve()


def require_under_collection_workspace(path: str | Path, workspace: Path | None = None) -> Path:
    """Resolve ``path`` and require it lies under the collection workspace.

    Workspace is ``AID_WORKSPACE_ROOT`` (resolved) when set, otherwise ``cwd``. Both
    relative paths (including ``..`` tricks) and absolute paths must resolve inside
    this anchor so collection cannot be pointed at arbitrary filesystem locations
    unless the process cwd or ``AID_WORKSPACE_ROOT`` already covers them.
    """
    w = workspace if workspace is not None else collection_workspace_root()
    p = Path(path).expanduser()
    if p.exists():
        reject_symlink(p)
    resolved = p.resolve()
    try:
        resolved.relative_to(w)
    except ValueError as exc:
        raise ValueError(f"collection_path_escapes_workspace path={path!r} workspace={w}") from exc
    return resolved


def validate_collection_io_paths(
    *,
    workspace: Path | None = None,
    out: str | Path | None = None,
    sources_file: str | Path | None = None,
    hf_cache_file: str | Path | None = None,
    hf_audit_file: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> None:
    """Ensure configured collection paths do not escape the workspace (no ``..`` tricks)."""
    w = workspace if workspace is not None else collection_workspace_root()
    checks: tuple[tuple[str, Any], ...] = (
        ("out", out),
        ("sources_file", sources_file),
        ("hf_cache_file", hf_cache_file),
        ("hf_audit_file", hf_audit_file),
        ("cache_dir", cache_dir),
    )
    for label, raw in checks:
        if raw is None or raw == "":
            continue
        require_under_collection_workspace(raw, w)

