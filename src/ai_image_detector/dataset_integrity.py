"""Dataset preflight: symlink rejection, optional content manifests, train/val leakage checks."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Iterable

from .dataset_layout import IMAGE_EXTS
from .io_limits import MAX_IMAGE_FILE_BYTES, check_file_size
from .utils.jsonio import write_json_atomic


def preflight_dataset_tree(data_root: Path, *, splits: tuple[str, ...] = ("train", "val")) -> None:
    """Fail fast if split roots, class buckets, or image leaves are symlinks."""
    if os.environ.get("AID_SKIP_DATA_PREFLIGHT", "").strip().lower() in {"1", "true", "yes"}:
        return
    root = data_root.resolve()
    for split in splits:
        split_dir = root / split
        if not split_dir.exists():
            continue
        if split_dir.is_symlink():
            raise ValueError(f"dataset_split_symlink_not_allowed path={split_dir}")
        for cls_dir in split_dir.iterdir():
            if not cls_dir.is_dir():
                continue
            if cls_dir.is_symlink():
                raise ValueError(f"dataset_class_symlink_not_allowed path={cls_dir}")
            for p in cls_dir.rglob("*"):
                if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
                    continue
                if p.is_symlink():
                    raise ValueError(f"dataset_image_symlink_not_allowed path={p}")


def _rel_str(data_root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(data_root.resolve()))


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    check_file_size(path, max_bytes=MAX_IMAGE_FILE_BYTES)
    return sha256_path(path)


def build_manifest_records(
    samples: Iterable[tuple[str, int]],
    class_names: list[str],
    data_root: Path,
    *,
    hash_files: bool,
) -> list[dict[str, Any]]:
    root = data_root.resolve()
    out: list[dict[str, Any]] = []
    for path_str, target in samples:
        p = Path(path_str)
        st = p.stat()
        rel = _rel_str(root, p)
        label = class_names[int(target)] if 0 <= int(target) < len(class_names) else str(int(target))
        digest: str | None = None
        if hash_files:
            digest = sha256_file(p)
        rec: dict[str, Any] = {
            "rel": rel,
            "label": label,
            "size": int(st.st_size),
            "mtime_ns": int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))),
        }
        if digest is not None:
            rec["sha256"] = digest
        out.append(rec)
    out.sort(key=lambda x: x["rel"])
    return out


def assert_no_train_val_hash_overlap(train_records: list[dict[str, Any]], val_records: list[dict[str, Any]]) -> None:
    """Require every record to include sha256; error if any hash appears in both splits."""
    train_h: set[str] = set()
    val_h: set[str] = set()
    for r in train_records:
        h = r.get("sha256")
        if not h:
            raise ValueError("strict_dataset_requires_sha256_on_train manifest_records")
        train_h.add(str(h))
    for r in val_records:
        h = r.get("sha256")
        if not h:
            raise ValueError("strict_dataset_requires_sha256_on_val manifest_records")
        val_h.add(str(h))
    overlap = train_h & val_h
    if overlap:
        sample = next(iter(overlap))
        raise ValueError(f"train_val_content_overlap count={len(overlap)} example_sha256={sample}")


def write_dataset_manifest(
    path: Path,
    *,
    data_root: Path,
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
    schema: str = "ai-image-detector-dataset-manifest-v1",
) -> None:
    payload = {
        "schema": schema,
        "data_root": str(data_root.resolve()),
        "train_count": len(train_records),
        "val_count": len(val_records),
        "train": train_records,
        "val": val_records,
    }
    write_json_atomic(path, payload, indent=2, sort_keys=False)
