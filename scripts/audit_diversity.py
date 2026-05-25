from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Dict

from ai_image_detector.utils import read_json_dict_lenient


SOURCE_RE = re.compile(r"source=([^_]+(?:_[^_]+)*)__")
HARDNEG_RE = re.compile(r"hardneg=([a-z0-9]+)__")


def _count_sources(split_dir: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if not split_dir.exists():
        return counts
    for p in split_dir.glob("*.jpg"):
        m = SOURCE_RE.search(p.name)
        if not m:
            continue
        src = m.group(1)
        counts[src] = counts.get(src, 0) + 1
    return counts


def _count_hardneg(train_dir: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if not train_dir.exists():
        return counts
    for p in train_dir.glob("hardneg=*.jpg"):
        m = HARDNEG_RE.search(p.name)
        if not m:
            continue
        mode = m.group(1)
        counts[mode] = counts.get(mode, 0) + 1
    return counts


def _max_source_share(counts: Dict[str, int]) -> float:
    total = int(sum(counts.values()))
    if total <= 0:
        return 0.0
    return float(max(counts.values())) / float(total)


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit dataset diversity from filenames/report")
    ap.add_argument("--data", default="./data_best")
    ap.add_argument("--min-unique-sources", type=int, default=20)
    ap.add_argument("--min-hardneg-modes", type=int, default=4)
    ap.add_argument("--min-sources-per-split-class", type=int, default=0)
    ap.add_argument("--max-class-imbalance", type=float, default=0.08, help="max |ai-real|/max(ai,real) per split")
    ap.add_argument("--max-source-share-per-split", type=float, default=0.0, help="max fraction contributed by one source within a split")
    ap.add_argument("--max-source-share-per-split-class", type=float, default=0.0, help="max fraction contributed by one source within a split/class bucket")
    args = ap.parse_args()

    root = Path(args.data)
    report_path = root / "dataset_build_report.json"
    report = read_json_dict_lenient(report_path)

    failures: list[str] = []
    info: dict[str, object] = {}

    split_sources: Dict[str, Dict[str, int]] = {}
    split_class_sources: Dict[str, Dict[str, int]] = {}
    for split in ("train", "val", "test"):
        src_counts: Dict[str, int] = {}
        for cls in ("ai", "real"):
            c = _count_sources(root / split / cls)
            split_class_sources[f"{split}/{cls}"] = c
            for k, v in c.items():
                src_counts[k] = src_counts.get(k, 0) + v
        split_sources[split] = src_counts
        uniq = len(src_counts)
        info[f"{split}_unique_sources"] = uniq
        split_share = _max_source_share(src_counts)
        info[f"{split}_max_source_share"] = round(float(split_share), 5)
        if uniq < args.min_unique_sources:
            failures.append(f"{split}: unique_sources={uniq} < min={args.min_unique_sources}")
        if args.max_source_share_per_split > 0 and split_share > args.max_source_share_per_split:
            failures.append(
                f"{split}: max_source_share={split_share:.4f} > max={args.max_source_share_per_split:.4f}"
            )
        if args.min_sources_per_split_class > 0:
            for cls in ("ai", "real"):
                bucket_sources = len(split_class_sources[f"{split}/{cls}"])
                info[f"{split}_{cls}_unique_sources"] = bucket_sources
                bucket_share = _max_source_share(split_class_sources[f"{split}/{cls}"])
                info[f"{split}_{cls}_max_source_share"] = round(float(bucket_share), 5)
                if bucket_sources < args.min_sources_per_split_class:
                    failures.append(
                        f"{split}/{cls}: unique_sources={bucket_sources} < min={args.min_sources_per_split_class}"
                    )
                if args.max_source_share_per_split_class > 0 and bucket_share > args.max_source_share_per_split_class:
                    failures.append(
                        f"{split}/{cls}: max_source_share={bucket_share:.4f} > max={args.max_source_share_per_split_class:.4f}"
                    )

    hard_modes = set(_count_hardneg(root / "train" / "ai")) | set(_count_hardneg(root / "train" / "real"))
    info["hardneg_modes"] = sorted(hard_modes)
    if len(hard_modes) < args.min_hardneg_modes:
        failures.append(f"hardneg_modes={len(hard_modes)} < min={args.min_hardneg_modes}")

    final_counts = report.get("final_counts", {})
    for split in ("train", "val", "test"):
        split_counts = final_counts.get(split, {})
        ai_n = int(split_counts.get("ai", 0))
        real_n = int(split_counts.get("real", 0))
        denom = max(ai_n, real_n, 1)
        imbalance = abs(ai_n - real_n) / denom
        info[f"{split}_imbalance"] = round(float(imbalance), 5)
        if imbalance > args.max_class_imbalance:
            failures.append(
                f"{split}: imbalance={imbalance:.4f} > max={args.max_class_imbalance:.4f} (ai={ai_n}, real={real_n})"
            )

    out = {
        "ok": len(failures) == 0,
        "checks": info,
        "failures": failures,
    }
    print(json.dumps(out, indent=2))
    if failures:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
