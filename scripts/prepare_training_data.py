from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

from script_support import ensure_src_path

ensure_src_path()

from ai_image_detector.dataset_integrity import sha256_file
from ai_image_detector.dataset_layout import CLASSES, IMAGE_EXTS, IMAGE_SPLITS, image_counts, iter_bucket_files
from ai_image_detector.utils import write_json_dict


def _same_content(a: Path, b: Path) -> bool:
    if not a.exists() or not b.exists():
        return False
    if a.stat().st_size != b.stat().st_size:
        return False
    try:
        if os.path.samefile(a, b):
            return True
    except OSError:
        pass
    return sha256_file(a) == sha256_file(b)


def _link_or_copy(src: Path, dst: Path, copy_only: bool) -> str:
    if not copy_only:
        try:
            os.link(src, dst)
            return "linked"
        except OSError:
            pass
    shutil.copy2(src, dst)
    return "copied"


def _materialize_bucket(files: list[Path], out_dir: Path, copy_only: bool) -> dict[str, int]:
    stats = {
        "added": 0,
        "linked": 0,
        "copied": 0,
        "skipped_existing": 0,
        "renamed_conflicts": 0,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for src in files:
        dst = out_dir / src.name
        if dst.exists():
            if _same_content(dst, src):
                stats["skipped_existing"] += 1
                continue
            idx = 1
            while True:
                candidate = out_dir / f"{src.stem}__merged_{idx:04d}{src.suffix.lower()}"
                if candidate.exists():
                    if _same_content(candidate, src):
                        stats["skipped_existing"] += 1
                        dst = candidate
                        break
                    idx += 1
                    continue
                dst = candidate
                stats["renamed_conflicts"] += 1
                break
            if dst.exists():
                continue
        mode = _link_or_copy(src, dst, copy_only=copy_only)
        stats["added"] += 1
        stats[mode] += 1
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description="Prepare additive training data from collected base + incremental buckets")
    ap.add_argument("--base", default="./data_best")
    ap.add_argument("--incremental", default="./data_new")
    ap.add_argument("--out", default="./.local/training_data")
    ap.add_argument("--copy", action="store_true", default=False, help="Copy files instead of hard-linking when possible")
    args = ap.parse_args()

    base = Path(args.base)
    incremental = Path(args.incremental)
    out = Path(args.out)

    if out.resolve() == base.resolve():
        raise SystemExit(f"output_must_not_equal_base path={out}")
    if incremental.exists() and out.resolve() == incremental.resolve():
        raise SystemExit(f"output_must_not_equal_incremental path={out}")

    summary: dict[str, object] = {
        "base": str(base.resolve()),
        "incremental": str(incremental.resolve()),
        "out": str(out.resolve()),
        "copy_only": bool(args.copy),
        "bucket_stats": {split: {cls: {} for cls in CLASSES} for split in IMAGE_SPLITS},
    }

    for split in IMAGE_SPLITS:
        for cls in CLASSES:
            base_files = iter_bucket_files(base, split, cls, exts=IMAGE_EXTS, allow_train_root_alias=True, include_symlinks=False)
            incremental_files = iter_bucket_files(
                incremental,
                split,
                cls,
                exts=IMAGE_EXTS,
                allow_train_root_alias=True,
                include_symlinks=False,
            )
            bucket_files = [*base_files, *incremental_files]
            stats = _materialize_bucket(bucket_files, out / split / cls, copy_only=bool(args.copy))
            stats["base_candidates"] = len(base_files)
            stats["incremental_candidates"] = len(incremental_files)
            summary["bucket_stats"][split][cls] = stats

    final_counts = image_counts(out, allow_train_root_alias=True, include_symlinks=False)
    summary["final_counts"] = final_counts

    missing_buckets: list[str] = []
    for split in IMAGE_SPLITS:
        for cls in CLASSES:
            if final_counts[split][cls] <= 0:
                missing_buckets.append(f"{split}/{cls}")

    summary["complete_image_dataset"] = len(missing_buckets) == 0
    summary["missing_buckets"] = missing_buckets

    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "training_data_report.json"
    write_json_dict(report_path, summary)
    print(json.dumps(summary, indent=2))

    if missing_buckets:
        print("training_data_invalid missing_buckets=" + ",".join(missing_buckets), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
