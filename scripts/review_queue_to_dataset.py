from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from ai_image_detector.collection_paths import validate_review_queue_paths
from ai_image_detector.io_limits import reject_symlink


def _ingest_file(src: Path, dst: Path, *, copy: bool) -> None:
    reject_symlink(src)
    if dst.exists():
        reject_symlink(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copy2(src, dst)
    else:
        src.rename(dst)


def main() -> None:
    ap = argparse.ArgumentParser(description="Move reviewed queue items into labeled dataset folders")
    ap.add_argument("--queue", default="./incoming_review_queue")
    ap.add_argument("--dst", default="./data_new/train")
    ap.add_argument("--archive", default="./incoming_review_queue/_processed")
    ap.add_argument("--copy", action="store_true", default=False, help="Copy instead of move")
    args = ap.parse_args()

    q, dst_root, archive_root = validate_review_queue_paths(
        queue=args.queue,
        dst=args.dst,
        archive=args.archive,
    )
    ingested = 0
    for cls in ("ai", "real"):
        (dst_root / cls).mkdir(parents=True, exist_ok=True)
        src_dir = q / cls
        if not src_dir.is_dir():
            continue
        for p in sorted(src_dir.iterdir()):
            if not p.is_file() or p.is_symlink():
                continue
            if p.suffix.lower() != ".jpg":
                continue
            out = dst_root / cls / p.name
            _ingest_file(p, out, copy=bool(args.copy))
            ingested += 1
            j = p.with_suffix(".json")
            if j.is_file() and not j.is_symlink():
                aj = archive_root / cls
                aj.mkdir(parents=True, exist_ok=True)
                _ingest_file(j, aj / j.name, copy=bool(args.copy))
    print(f"review_queue_ingested dst={dst_root} count={ingested}")


if __name__ == "__main__":
    main()
