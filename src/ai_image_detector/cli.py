from __future__ import annotations

from importlib import import_module
from pathlib import Path
import shlex
import sys


def _repo_root() -> Path | None:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "local.sh").is_file() and (parent / "pyproject.toml").is_file():
            return parent
    return None


def _pip_extra_install_command(extra: str) -> str:
    return f'python -m pip install --upgrade "ai-image-video-detector[{extra}]"'


def _run_entrypoint(module_name: str, attr_name: str, *, extra: str) -> int:
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        missing_name = getattr(exc, "name", "") or "unknown"
        if missing_name.startswith("ai_image_detector"):
            raise
        repo_root = _repo_root()
        if repo_root is not None:
            hint = (
                f"run=(cd {shlex.quote(str(repo_root))} && "
                f"env DEPS_EXTRA={shlex.quote(extra)} ./local.sh deps)"
            )
        else:
            hint = f'run={_pip_extra_install_command(extra)}'
        print(
            f"missing_dependency={missing_name} hint_extra={extra} {hint}",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc

    func = getattr(module, attr_name)
    result = func()
    if isinstance(result, int):
        return result
    return 0


def train_main() -> int:
    return _run_entrypoint("ai_image_detector.train", "main", extra="training")


def video_train_main() -> int:
    return _run_entrypoint("ai_image_detector.video_temporal", "train_main", extra="training,video")
