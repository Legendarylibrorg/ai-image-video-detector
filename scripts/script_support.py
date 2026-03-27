from __future__ import annotations

from pathlib import Path
import sys


def _ensure_src_path() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src"
    src_entry = str(src_root)
    if src_entry not in sys.path:
        sys.path.insert(0, src_entry)


try:
    from ai_image_detector.utils import git_commit, read_json_dict, read_nonempty_lines, write_json_dict
except ModuleNotFoundError:
    _ensure_src_path()
    from ai_image_detector.utils import git_commit, read_json_dict, read_nonempty_lines, write_json_dict


def iter_member_dirs(ens_out: Path) -> list[Path]:
    return sorted((path for path in ens_out.glob("m*") if path.is_dir()), key=lambda path: path.name)


def resolve_checkpoint(path: Path) -> Path | None:
    if path.exists():
        return path
    safe = path.with_suffix(".safetensors")
    if safe.exists():
        return safe
    return None


def resolve_preferred_checkpoint(path: Path) -> Path:
    safe = path.with_suffix(".safetensors")
    if safe.exists():
        return safe
    return path
