from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from pathlib import Path
from typing import Iterable

from PIL import Image

from ai_image_detector.collection_paths import collection_workspace_root, require_under_collection_workspace
from ai_image_detector.dataset_layout import IMAGE_EXTS
from ai_image_detector.io_limits import MAX_IMAGE_FILE_BYTES, check_file_size, configure_pil_limits, path_must_be_under, reject_symlink


def iter_images_under(root: Path) -> Iterable[Path]:
    """List image files under ``root`` without following directory symlinks."""
    root_r = root.resolve()
    if not root_r.is_dir():
        return
    for dirpath, _dirnames, filenames in os.walk(root_r, topdown=True, followlinks=False):
        base = Path(dirpath)
        for name in filenames:
            p = base / name
            if not p.is_file():
                continue
            if p.suffix.lower() not in IMAGE_EXTS:
                continue
            yield p


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_hashes(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def save_hashes(path: Path, hashes: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(sorted(hashes)) + ("\n" if hashes else ""), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest fresh model outputs into incremental training data")
    ap.add_argument("--src", default="./incoming_model_outputs", help="Expected class folders: <src>/ai and <src>/real")
    ap.add_argument("--dst", default="./data_new/train", help="Destination with class folders")
    ap.add_argument("--archive", default="./incoming_model_outputs/_processed")
    ap.add_argument("--jpeg-quality", type=int, default=92)
    ap.add_argument("--min-side", type=int, default=128)
    args = ap.parse_args()
    configure_pil_limits()

    workspace = collection_workspace_root()
    src_root = require_under_collection_workspace(args.src, workspace)
    dst_root = require_under_collection_workspace(args.dst, workspace)
    archive_root = require_under_collection_workspace(args.archive, workspace)

    hash_manifest = dst_root / ".hashes.txt"
    seen = load_hashes(hash_manifest)
    start_seen = len(seen)
    ingested = 0

    for cls in ("ai", "real"):
        src_cls = src_root / cls
        if not src_cls.exists():
            continue
        dst_cls = dst_root / cls
        dst_cls.mkdir(parents=True, exist_ok=True)
        archive_cls = archive_root / cls
        archive_cls.mkdir(parents=True, exist_ok=True)

        for p in iter_images_under(src_cls):
            try:
                path_must_be_under(p, src_root)
            except (ValueError, FileNotFoundError):
                continue
            try:
                reject_symlink(p)
            except ValueError:
                continue
            try:
                check_file_size(p, max_bytes=MAX_IMAGE_FILE_BYTES)
                raw = p.read_bytes()
            except (OSError, ValueError):
                continue

            h = hash_bytes(raw)
            if h in seen:
                try:
                    p.unlink()
                except OSError:
                    pass
                continue

            try:
                with Image.open(p) as img:
                    rgb = img.convert("RGB")
                    if min(rgb.size) < args.min_side:
                        continue
                    out = dst_cls / f"source=model_output__{cls}_{h[:16]}.jpg"
                    rgb.save(out, quality=args.jpeg_quality)
            except OSError:
                continue

            seen.add(h)
            ingested += 1
            try:
                arch = archive_cls / p.name
                if arch.exists():
                    arch = archive_cls / f"{p.stem}_{h[:8]}{p.suffix.lower()}"
                shutil.move(str(p), str(arch))
            except OSError:
                pass

    save_hashes(hash_manifest, seen)
    print(
        f"model_output_ingest src={src_root} dst={dst_root} ingested={ingested} "
        f"known_before={start_seen} known_after={len(seen)}"
    )


if __name__ == "__main__":
    main()
