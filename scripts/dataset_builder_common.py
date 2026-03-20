from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Sequence


def configure_hf_cache_env(cache_dir: str) -> Path | None:
    if not cache_dir:
        return None
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_path))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_path / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_path / "datasets"))
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    return cache_path


def count_existing_split_classes(
    out: Path,
    splits: Sequence[str],
    classes: Sequence[str],
    pattern: str,
) -> Dict[str, Dict[str, int]]:
    have: Dict[str, Dict[str, int]] = {split: {cls: 0 for cls in classes} for split in splits}
    for split in splits:
        for cls in classes:
            target_dir = out / split / cls
            have[split][cls] = len(list(target_dir.glob(pattern))) if target_dir.exists() else 0
    return have


def targets_met(
    have: Dict[str, Dict[str, int]],
    need: Dict[str, Dict[str, int]],
    splits: Sequence[str],
    classes: Sequence[str],
) -> bool:
    return all(have[split][cls] >= need[split][cls] for split in splits for cls in classes)
