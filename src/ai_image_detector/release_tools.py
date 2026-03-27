from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
import shutil


def write_timestamped_release(
    out_dir: str | Path,
    artifact_names: Iterable[str],
    *,
    preferred_artifact: str | Path | None = None,
) -> Path:
    root = Path(out_dir)
    release_dir = root / "releases" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    release_dir.mkdir(parents=True, exist_ok=True)

    copied_names = set()
    for name in artifact_names:
        copied_names.add(str(name))
        src = root / name
        if src.exists():
            shutil.copy2(src, release_dir / name)

    if preferred_artifact is not None:
        preferred = Path(preferred_artifact)
        if not preferred.is_absolute():
            preferred = root / preferred
        if preferred.exists() and preferred.name not in copied_names:
            shutil.copy2(preferred, release_dir / preferred.name)

    (root / "latest_release.txt").write_text(str(release_dir), encoding="utf-8")
    return release_dir
