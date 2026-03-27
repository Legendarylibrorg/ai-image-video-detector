from __future__ import annotations

from importlib import import_module
import sys


def _run_entrypoint(module_name: str, attr_name: str, *, extra: str) -> int:
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        missing_name = getattr(exc, "name", "") or "unknown"
        if missing_name.startswith("ai_image_detector"):
            raise
        print(
            f"missing_dependency={missing_name} install_extra={extra} "
            f"run=pip install -e '.[{extra}]'",
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
