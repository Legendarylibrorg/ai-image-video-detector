from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from PIL import Image

from ai_image_detector.collection_paths import collection_workspace_root, require_under_collection_workspace
from ai_image_detector.io_limits import (
    MAX_IMAGE_FILE_BYTES,
    MAX_JSON_CONFIG_BYTES,
    check_file_size,
    configure_pil_limits,
    path_must_be_under,
    read_json_file_limited,
    reject_symlink,
)


def _iter_queue_jpgs(class_dir: Path) -> list[Path]:
    if not class_dir.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(class_dir.iterdir(), key=lambda item: item.name):
        if not p.is_file():
            continue
        if p.suffix.lower() != ".jpg":
            continue
        out.append(p)
    return out


def _validate_image(path: Path) -> bool:
    try:
        reject_symlink(path)
        check_file_size(path, max_bytes=MAX_IMAGE_FILE_BYTES)
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img.convert("RGB")
        return True
    except (OSError, ValueError):
        return False


def _validate_json_sidecar(path: Path) -> bool:
    if not path.exists():
        return True
    try:
        reject_symlink(path)
        read_json_file_limited(path, max_bytes=MAX_JSON_CONFIG_BYTES)
        return True
    except (OSError, ValueError):
        return False


def _transfer_file(src: Path, dst: Path, *, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copy2(src, dst)
    else:
        os.replace(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser(description="Move reviewed queue items into labeled dataset folders")
    ap.add_argument("--queue", default="./incoming_review_queue")
    ap.add_argument("--dst", default="./data_new/train")
    ap.add_argument("--archive", default="./incoming_review_queue/_processed")
    ap.add_argument("--copy", action="store_true", default=False, help="Copy instead of move")
    args = ap.parse_args()
    configure_pil_limits()

    workspace = collection_workspace_root()
    queue_root = require_under_collection_workspace(args.queue, workspace)
    dst_root = require_under_collection_workspace(args.dst, workspace)
    archive_root = require_under_collection_workspace(args.archive, workspace)

    ingested = 0
    for cls in ("ai", "real"):
        src_dir = queue_root / cls
        dst_cls = dst_root / cls
        dst_cls.mkdir(parents=True, exist_ok=True)
        archive_cls = archive_root / cls

        for p in _iter_queue_jpgs(src_dir):
            try:
                path_must_be_under(p, queue_root)
            except (ValueError, FileNotFoundError):
                continue
            if not _validate_image(p):
                continue
            sidecar = p.with_suffix(".json")
            if not _validate_json_sidecar(sidecar):
                continue

            out = dst_cls / p.name
            _transfer_file(p, out, copy=bool(args.copy))
            ingested += 1

            if sidecar.exists():
                archive_cls.mkdir(parents=True, exist_ok=True)
                arch_json = archive_cls / sidecar.name
                _transfer_file(sidecar, arch_json, copy=bool(args.copy))

    print(f"review_queue_ingested dst={dst_root} ingested={ingested}")


if __name__ == "__main__":
    main()
