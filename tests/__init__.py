from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]

for extra in (ROOT / "src", ROOT / "scripts", ROOT / "tests"):
    extra_str = str(extra)
    if extra.is_dir() and extra_str not in sys.path:
        sys.path.insert(0, extra_str)
