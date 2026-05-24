from __future__ import annotations

from pathlib import Path
import sys


def ensure_src_path() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src"
    src_entry = str(src_root)
    if src_entry not in sys.path:
        sys.path.insert(0, src_entry)


def _import_utils():
    from ai_image_detector.utils import (
        git_commit,
        read_json_dict,
        read_nonempty_lines,
        write_json_dict,
    )

    return git_commit, read_json_dict, read_nonempty_lines, write_json_dict


try:
    git_commit, read_json_dict, read_nonempty_lines, write_json_dict = _import_utils()
except ModuleNotFoundError:
    ensure_src_path()
    git_commit, read_json_dict, read_nonempty_lines, write_json_dict = _import_utils()


__all__ = [
    "ensure_src_path",
    "git_commit",
    "iter_member_dirs",
    "read_json_dict",
    "read_nonempty_lines",
    "resolve_checkpoint",
    "resolve_preferred_checkpoint",
    "write_json_dict",
]


def iter_member_dirs(ens_out: Path) -> list[Path]:
    return sorted((path for path in ens_out.glob("m*") if path.is_dir()), key=lambda path: path.name)


def _checkpoint_candidate(path: Path) -> Path:
    safe = path.with_suffix(".safetensors")
    if safe.exists():
        return safe
    return path


def resolve_checkpoint(path: Path) -> Path | None:
    candidate = _checkpoint_candidate(path)
    if candidate.exists():
        return candidate
    return None


def resolve_preferred_checkpoint(path: Path) -> Path:
    return _checkpoint_candidate(path)
