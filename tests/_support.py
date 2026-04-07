from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def source_tree_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    pythonpath_parts = [str(SRC), str(SCRIPTS)]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    if extra:
        env.update(extra)
    return env


def write_rgb_image(path: Path, color: tuple[int, int, int] = (64, 128, 192), size: tuple[int, int] = (24, 24)) -> None:
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise unittest.SkipTest("requires Pillow") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path, quality=92)
