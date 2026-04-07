from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence


IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"})
VIDEO_EXTS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"})
IMAGE_SPLITS = ("train", "val", "test")
VIDEO_SPLITS = ("train", "val")
CLASSES = ("ai", "real")


def resolve_split_root(
    root: str | Path,
    split: str,
    *,
    classes: Sequence[str] = CLASSES,
    allow_train_root_alias: bool = False,
) -> Path | None:
    dataset_root = Path(root)
    if (dataset_root / split).is_dir():
        return dataset_root / split
    if allow_train_root_alias and split == "train" and any((dataset_root / cls).is_dir() for cls in classes):
        return dataset_root
    return None


def bucket_path(
    root: str | Path,
    split: str,
    cls: str,
    *,
    classes: Sequence[str] = CLASSES,
    allow_train_root_alias: bool = False,
) -> Path:
    split_root = resolve_split_root(root, split, classes=classes, allow_train_root_alias=allow_train_root_alias)
    if split_root is None:
        return Path(root) / split / cls
    return split_root / cls


def iter_bucket_files(
    root: str | Path,
    split: str,
    cls: str,
    *,
    exts: Iterable[str],
    classes: Sequence[str] = CLASSES,
    allow_train_root_alias: bool = False,
    include_symlinks: bool = True,
) -> list[Path]:
    bucket = bucket_path(root, split, cls, classes=classes, allow_train_root_alias=allow_train_root_alias)
    if not bucket.exists():
        return []
    normalized_exts = {str(ext).lower() for ext in exts}
    return sorted(
        path
        for path in bucket.iterdir()
        if path.is_file()
        and path.suffix.lower() in normalized_exts
        and (include_symlinks or not path.is_symlink())
    )


def count_split_class_files(
    root: str | Path,
    *,
    splits: Sequence[str],
    classes: Sequence[str],
    exts: Iterable[str],
    allow_train_root_alias: bool = False,
    include_symlinks: bool = True,
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {
        split: {cls: 0 for cls in classes}
        for split in splits
    }
    for split in splits:
        for cls in classes:
            counts[split][cls] = len(
                iter_bucket_files(
                    root,
                    split,
                    cls,
                    exts=exts,
                    classes=classes,
                    allow_train_root_alias=allow_train_root_alias,
                    include_symlinks=include_symlinks,
                )
            )
    return counts


def complete_split_class_counts(
    counts: dict[str, dict[str, int]],
    *,
    splits: Sequence[str],
    classes: Sequence[str],
    minimum: int = 1,
) -> bool:
    return all(int(counts.get(split, {}).get(cls, 0)) >= int(minimum) for split in splits for cls in classes)


def split_class_shortfalls(
    counts: dict[str, dict[str, int]],
    *,
    splits: Sequence[str],
    classes: Sequence[str],
    required_by_split: dict[str, int],
) -> list[dict[str, int | str]]:
    shortfalls: list[dict[str, int | str]] = []
    for split in splits:
        need = max(0, int(required_by_split.get(split, 0)))
        for cls in classes:
            have = int(counts.get(split, {}).get(cls, 0))
            if have >= need:
                continue
            shortfalls.append(
                {
                    "split": split,
                    "class": cls,
                    "have": have,
                    "need": need,
                }
            )
    return shortfalls


def image_counts(root: str | Path, *, allow_train_root_alias: bool = False, include_symlinks: bool = True) -> dict[str, dict[str, int]]:
    return count_split_class_files(
        root,
        splits=IMAGE_SPLITS,
        classes=CLASSES,
        exts=IMAGE_EXTS,
        allow_train_root_alias=allow_train_root_alias,
        include_symlinks=include_symlinks,
    )


def video_counts(root: str | Path, *, allow_train_root_alias: bool = False, include_symlinks: bool = True) -> dict[str, dict[str, int]]:
    return count_split_class_files(
        root,
        splits=VIDEO_SPLITS,
        classes=CLASSES,
        exts=VIDEO_EXTS,
        allow_train_root_alias=allow_train_root_alias,
        include_symlinks=include_symlinks,
    )


def _emit_complete_status(
    *,
    root: Path,
    counts: dict[str, dict[str, int]],
    splits: Sequence[str],
    classes: Sequence[str],
    minimum: int,
    bucket_prefix: str,
    summary_prefix: str,
    quiet: bool,
) -> int:
    required = {split: int(minimum) for split in splits}
    shortfalls = split_class_shortfalls(counts, splits=splits, classes=classes, required_by_split=required)
    if not quiet:
        for item in shortfalls:
            print(f"{bucket_prefix}={bucket_path(root, str(item['split']), str(item['class']), classes=classes)}")
    if shortfalls:
        if not quiet:
            print(f"{summary_prefix}=invalid root={root}")
        return 1
    if not quiet:
        print(f"{summary_prefix}=ok root={root}")
    return 0


def _emit_minimum_status(
    *,
    root: Path,
    counts: dict[str, dict[str, int]],
    splits: Sequence[str],
    classes: Sequence[str],
    required_by_split: dict[str, int],
    bucket_prefix: str,
    summary_prefix: str,
    quiet: bool,
) -> int:
    shortfalls = split_class_shortfalls(counts, splits=splits, classes=classes, required_by_split=required_by_split)
    if not quiet:
        for item in shortfalls:
            bucket = bucket_path(root, str(item["split"]), str(item["class"]), classes=classes)
            print(f"{bucket_prefix}={bucket} have={item['have']} need={item['need']}")
    if shortfalls:
        if not quiet:
            print(
                f"{summary_prefix}=invalid root={root} "
                f"train_min={required_by_split.get('train', 0)} "
                f"val_min={required_by_split.get('val', 0)} "
                f"test_min={required_by_split.get('test', 0)}"
            )
        return 1
    if not quiet:
        print(
            f"{summary_prefix}=ok root={root} "
            f"train_min={required_by_split.get('train', 0)} "
            f"val_min={required_by_split.get('val', 0)} "
            f"test_min={required_by_split.get('test', 0)}"
        )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Shared dataset layout/counting helpers")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_counts = sub.add_parser("counts", help="Print split/class counts as JSON")
    p_counts.add_argument("--root", required=True)
    p_counts.add_argument("--kind", choices=["image", "video"], default="image")
    p_counts.add_argument("--allow-train-root-alias", action="store_true")
    p_counts.add_argument("--skip-symlinks", action="store_true")

    p_image_complete = sub.add_parser("check-image-complete", help="Require at least one image in every split/class bucket")
    p_image_complete.add_argument("--root", required=True)
    p_image_complete.add_argument("--quiet", action="store_true")

    p_video_complete = sub.add_parser("check-video-complete", help="Require at least one video in every train/val bucket")
    p_video_complete.add_argument("--root", required=True)
    p_video_complete.add_argument("--quiet", action="store_true")

    p_image_minimums = sub.add_parser("check-image-minimums", help="Require minimum image counts for every split/class bucket")
    p_image_minimums.add_argument("--root", required=True)
    p_image_minimums.add_argument("--train-min", type=int, default=0)
    p_image_minimums.add_argument("--val-min", type=int, default=0)
    p_image_minimums.add_argument("--test-min", type=int, default=0)
    p_image_minimums.add_argument("--quiet", action="store_true")

    args = ap.parse_args()
    root = Path(args.root)

    if args.cmd == "counts":
        include_symlinks = not bool(args.skip_symlinks)
        if args.kind == "video":
            counts = video_counts(
                root,
                allow_train_root_alias=bool(args.allow_train_root_alias),
                include_symlinks=include_symlinks,
            )
        else:
            counts = image_counts(
                root,
                allow_train_root_alias=bool(args.allow_train_root_alias),
                include_symlinks=include_symlinks,
            )
        print(json.dumps(counts, indent=2))
        return 0

    if args.cmd == "check-image-complete":
        return _emit_complete_status(
            root=root,
            counts=image_counts(root),
            splits=IMAGE_SPLITS,
            classes=CLASSES,
            minimum=1,
            bucket_prefix="missing_image_bucket",
            summary_prefix="image_training_data",
            quiet=bool(args.quiet),
        )

    if args.cmd == "check-video-complete":
        return _emit_complete_status(
            root=root,
            counts=video_counts(root),
            splits=VIDEO_SPLITS,
            classes=CLASSES,
            minimum=1,
            bucket_prefix="missing_video_bucket",
            summary_prefix="video_training_data",
            quiet=bool(args.quiet),
        )

    return _emit_minimum_status(
        root=root,
        counts=image_counts(root),
        splits=IMAGE_SPLITS,
        classes=CLASSES,
        required_by_split={
            "train": int(args.train_min),
            "val": int(args.val_min),
            "test": int(args.test_min),
        },
        bucket_prefix="insufficient_image_bucket",
        summary_prefix="image_collection_counts",
        quiet=bool(args.quiet),
    )


if __name__ == "__main__":
    raise SystemExit(main())
