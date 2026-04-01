from __future__ import annotations

import argparse
from collections import defaultdict
import math
import os
from pathlib import Path
import random
import re
import time
from typing import Callable, DefaultDict, Dict, List, Optional, Tuple

from PIL import Image, ImageFilter

from ai_image_detector.io_limits import configure_pil_limits, open_image_rgb
from build_best_dataset_policy import (
    next_split_for_class,
    next_split_for_source_class,
    should_skip_source_from_manifest,
    source_manifest_policy,
    utc_now_iso,
)
from build_best_dataset_sources import DEFAULT_ALLOWED_LICENSE_TAGS, build_source_list
from build_best_dataset_support import build_summary, make_source_report, run_source_acceptance_loop, write_summary_files
from dataset_builder_common import HF_CACHE_DIR_DEFAULT, configure_hf_cache_env, targets_met
from hf_data import (
    LoadedDatasetSource,
    append_source_manifest_entry,
    iter_source_examples,
    load_hf_dataset_source,
    load_latest_source_manifest,
    normalize_hf_token,
    normalize_image_dataset_split,
)
from image_materialize import (
    ImageDeduper,
    ImageQualityPolicy,
    open_example_image,
    passes_quality_filters,
    save_img,
)


def normalize_label(v) -> Optional[str]:
    if isinstance(v, bool):
        return "ai" if v else "real"
    if isinstance(v, int):
        if int(v) == 1:
            return "ai"
        if int(v) == 0:
            return "real"
        return None
    s = str(v).strip().lower()
    if s.isdigit():
        if int(s) == 1:
            return "ai"
        if int(s) == 0:
            return "real"
        return None
    if any(k in s for k in ["ai", "fake", "generated", "synthetic", "deepfake"]):
        return "ai"
    if any(k in s for k in ["real", "human", "natural", "authentic"]):
        return "real"
    return None


def find_fields(ds_split) -> Tuple[str, str]:
    cols = ds_split.column_names
    image_field = "image" if "image" in cols else next((c for c in cols if "image" in c.lower() or c.lower() == "img"), None)
    label_field = "label" if "label" in cols else next((c for c in cols if c.lower() in {"class", "target", "labels"}), None)
    if image_field is None or label_field is None:
        raise RuntimeError(f"Unable to infer fields from columns: {cols}")
    return image_field, label_field


def build_label_resolver(ds_split, label_field: str) -> Callable[[object], Optional[str]]:
    features = getattr(ds_split, "features", None) or {}
    feature = features.get(label_field) if hasattr(features, "get") else None
    names = getattr(feature, "names", None)
    class_map: dict[int, str] = {}
    if isinstance(names, (list, tuple)):
        for idx, name in enumerate(names):
            cls = normalize_label(name)
            if cls is not None:
                class_map[int(idx)] = cls

    def resolve(value: object) -> Optional[str]:
        if class_map:
            if isinstance(value, bool):
                return normalize_label(value)
            if isinstance(value, int) and int(value) in class_map:
                return class_map[int(value)]
            if isinstance(value, str):
                stripped = value.strip()
                if stripped.isdigit() and int(stripped) in class_map:
                    return class_map[int(stripped)]
        return normalize_label(value)

    return resolve


def augment_hard_negative(img: Image.Image, mode: str) -> Image.Image:
    if mode == "jpeg35":
        import io

        bio = io.BytesIO()
        img.save(bio, format="JPEG", quality=35)
        bio.seek(0)
        return Image.open(bio).convert("RGB")
    if mode == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=1.2))
    if mode == "resize60":
        w, h = img.size
        nw, nh = max(16, int(w * 0.6)), max(16, int(h * 0.6))
        return img.resize((nw, nh), Image.BILINEAR).resize((w, h), Image.BILINEAR)
    if mode == "sharpen":
        return img.filter(ImageFilter.UnsharpMask(radius=1.4, percent=130, threshold=3))
    if mode == "screenshot":
        canvas = Image.new("RGB", (img.width + 40, img.height + 80), (18, 18, 22))
        canvas.paste(img, (20, 20))
        return canvas.resize(img.size, Image.BILINEAR)
    return img


def source_tag(src: str) -> str:
    base = src.split("/")[-1]
    tag = re.sub(r"[^a-zA-Z0-9]+", "_", base).strip("_").lower()
    return (tag or "src")[:30]


def likely_rate_limited(msg: str) -> bool:
    low = msg.lower()
    return any(k in low for k in ["429", "too many requests", "rate limit", "ratelimit", "5 min"])


def likely_transient_hf_error(msg: str) -> bool:
    low = msg.lower()
    return any(
        k in low
        for k in [
            "timed out",
            "timeout",
            "connection reset",
            "connection aborted",
            "temporarily unavailable",
            "service unavailable",
            "bad gateway",
            "gateway timeout",
            "internal server error",
            "remoteprotocolerror",
            "connectionerror",
        ]
    )


SPLITS = ("train", "val", "test")
CLASSES = ("ai", "real")


def done(have: Dict[str, Dict[str, int]], need: Dict[str, Dict[str, int]]) -> bool:
    return targets_met(have, need, SPLITS, CLASSES)


def count_output_files(out: Path, include_hardneg: bool = True) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = {split: {cls: 0 for cls in CLASSES} for split in SPLITS}
    for split in SPLITS:
        for cls in CLASSES:
            split_dir = out / split / cls
            if not split_dir.exists():
                continue
            total = 0
            for path in split_dir.glob("*.jpg"):
                if not include_hardneg and path.name.startswith("hardneg="):
                    continue
                total += 1
            counts[split][cls] = total
    return counts


def count_existing(out: Path) -> Dict[str, Dict[str, int]]:
    return count_output_files(out, include_hardneg=False)


def build_existing_dedupe_state(out: Path) -> tuple[set[str], Dict[str, List[str]]]:
    deduper = ImageDeduper.from_output(out, splits=SPLITS, classes=CLASSES)
    return deduper.seen_exact, deduper.seen_dhash_by_cls


def counts_snapshot(counts: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, int]]:
    return {split: dict(bucket) for split, bucket in counts.items()}


def reset_hard_negative_outputs(out: Path) -> None:
    for cls in CLASSES:
        train_dir = out / "train" / cls
        if not train_dir.exists():
            continue
        for path in train_dir.glob("hardneg=*.jpg"):
            path.unlink(missing_ok=True)


def generate_hard_negatives(
    out: Path,
    targets: Dict[str, Dict[str, int]],
    hardneg_fraction: float,
    jpeg_quality: int,
    seed: int,
) -> Dict[str, int]:
    reset_hard_negative_outputs(out)
    configure_pil_limits()
    hard_modes = ["jpeg35", "blur", "resize60", "sharpen", "screenshot"]
    generated: Dict[str, int] = {}
    for cls in CLASSES:
        base_files = [p for p in (out / "train" / cls).glob("*.jpg") if not p.name.startswith("hardneg=")]
        shuffle_rng = random.Random(seed + (1 if cls == "ai" else 2))
        mode_rng = random.Random(seed + (101 if cls == "ai" else 202))
        shuffle_rng.shuffle(base_files)
        hard_target = int(targets["train"][cls] * max(hardneg_fraction, 0.0))
        if hard_target <= 0:
            generated[cls] = 0
            continue
        hn_count = 0
        for path in base_files[: min(hard_target, len(base_files))]:
            try:
                img = open_image_rgb(path)
            except Exception:
                continue
            mode = mode_rng.choice(hard_modes)
            aug = augment_hard_negative(img, mode)
            dst = out / "train" / cls / f"hardneg={mode}__{path.stem}__hn{hn_count:07d}.jpg"
            save_img(aug, dst, quality=max(70, min(95, jpeg_quality - 2)))
            hn_count += 1
        generated[cls] = hn_count
    return generated


def main():
    ap = argparse.ArgumentParser(description="Build large, high-quality AI-vs-real image dataset from Hugging Face sources")
    ap.add_argument("--out", default="data_best")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-per-class", type=int, default=30000)
    ap.add_argument("--val-per-class", type=int, default=7000)
    ap.add_argument("--test-per-class", type=int, default=7000)
    ap.add_argument("--near-hamming", type=int, default=2)
    ap.add_argument("--near-window", type=int, default=2400)
    ap.add_argument("--min-side", type=int, default=192)
    ap.add_argument("--max-aspect-ratio", type=float, default=3.0)
    ap.add_argument("--min-entropy", type=float, default=3.2)
    ap.add_argument("--max-unique-per-source", type=int, default=220000)
    ap.add_argument("--max-per-source-class", type=int, default=120000)
    ap.add_argument("--max-per-source-split-class", type=int, default=0, help="0 = derive from max-per-source-class / split count")
    ap.add_argument("--jpeg-quality", type=int, default=92)
    ap.add_argument("--hardneg-fraction", type=float, default=0.6)
    ap.add_argument("--sources-file", default="")
    ap.add_argument("--extra-source", action="append", default=[])
    ap.add_argument("--no-default-sources", action="store_true", default=False, help="Disable built-in static source list")
    ap.add_argument("--discover-hf", action="store_true", default=False)
    ap.add_argument("--no-discover-hf", dest="discover_hf", action="store_false")
    ap.add_argument("--hf-query", action="append", default=[])
    ap.add_argument("--hf-discovery-limit", type=int, default=180, help="Per-query max datasets to scan in discovery")
    ap.add_argument("--hf-max-sources", type=int, default=420, help="Global cap on discovered dataset ids")
    ap.add_argument("--hf-min-downloads", type=int, default=25)
    ap.add_argument("--hf-min-likes", type=int, default=1)
    ap.add_argument("--hf-min-quality-score", type=float, default=1.35)
    ap.add_argument("--hf-print-top", type=int, default=24)
    ap.add_argument("--hf-discovery-workers", type=int, default=8, help="Parallel worker count for HF discovery queries")
    ap.add_argument("--hf-query-pause-ms", type=int, default=0, help="Pause between HF discovery queries to stay under page limits")
    ap.add_argument("--hf-license-allow", action="append", default=list(DEFAULT_ALLOWED_LICENSE_TAGS), help="Allowed open/free HF dataset license markers")
    ap.add_argument("--hf-require-open-license", action="store_true", default=True, help="Require discovered HF sources to advertise an allowed open/free license")
    ap.add_argument("--no-hf-require-open-license", dest="hf_require_open_license", action="store_false")
    ap.add_argument("--hf-cache-file", default="", help="Optional file path to cache discovered HF source ids")
    ap.add_argument("--hf-cache-only-if-present", action="store_true", default=True, help="If cache file exists, use it and skip live HF discovery calls")
    ap.add_argument("--no-hf-cache-only-if-present", dest="hf_cache_only_if_present", action="store_false")
    ap.add_argument("--streaming", action="store_true", default=True, help="Use HF streaming mode to reduce metadata overhead")
    ap.add_argument("--no-streaming", dest="streaming", action="store_false")
    ap.add_argument("--cache-dir", default=HF_CACHE_DIR_DEFAULT, help="HF datasets cache directory (improves resume and avoids repeated downloads)")
    ap.add_argument("--stream-buffer-size", type=int, default=12000, help="Shuffle buffer for streaming datasets")
    ap.add_argument("--quiet-progress", action="store_true", default=True, help="Suppress noisy datasets map/filter progress bars")
    ap.add_argument("--verbose-progress", dest="quiet_progress", action="store_false")
    ap.add_argument("--max-samples-per-source", type=int, default=60000, help="Max examples to inspect per source before moving on")
    ap.add_argument("--acceptance-warmup-samples", type=int, default=400)
    ap.add_argument("--min-acceptance-rate", type=float, default=0.01)
    ap.add_argument("--repo-base-pause-ms", type=int, default=900, help="Base pause between HF repositories")
    ap.add_argument("--repo-jitter-ms", type=int, default=900, help="Extra random pause between HF repositories")
    ap.add_argument("--repo-cooldown-ms", type=int, default=45000, help="Cooldown after rate-limit or repeated source failures")
    ap.add_argument("--transient-error-cooldown-ms", type=int, default=3000, help="Short cooldown after repeated transient HF failures")
    ap.add_argument("--max-consecutive-failures", type=int, default=2, help="Cooldown trigger for consecutive source failures")
    ap.add_argument("--token-env", default="HF_TOKEN")
    ap.add_argument("--discover-only", action="store_true", default=False, help="Only run HF discovery/cache update and exit")
    ap.add_argument("--require-full-targets", action="store_true", default=False, help="Exit non-zero if final dataset is below requested class/split targets")
    ap.add_argument("--min-hf-sources-with-accepted", type=int, default=0, help="Require at least this many HF sources to contribute accepted samples")
    ap.add_argument("--min-hf-sources-per-class", type=int, default=0, help="Require at least this many HF sources with accepted samples for each class")
    ap.add_argument("--min-hf-sources-per-split-class", type=int, default=0, help="Require at least this many HF sources to contribute to each split/class bucket")
    args = ap.parse_args()
    start_time = time.time()

    random.seed(args.seed)
    rng = random.Random(args.seed + 17)

    token = normalize_hf_token(os.environ.get(args.token_env))
    if token:
        print(f"using_token_env={args.token_env}")
    else:
        print(f"warning_no_token env={args.token_env} (public datasets still work, but with lower limits)")
    cache_dir = configure_hf_cache_env(args.cache_dir)
    if cache_dir is not None:
        print(f"hf_cache_dir={cache_dir}")
    print(f"hf_quality_filters min_downloads={args.hf_min_downloads} min_likes={args.hf_min_likes} min_score={args.hf_min_quality_score} min_acceptance_rate={args.min_acceptance_rate}")
    print(
        "hf_license_policy require_open_license={} allowed={}".format(
            int(bool(args.hf_require_open_license)),
            ",".join(sorted({str(tag).strip().lower() for tag in (args.hf_license_allow or list(DEFAULT_ALLOWED_LICENSE_TAGS)) if str(tag).strip()})),
        )
    )

    if args.discover_only:
        hf_sources = build_source_list(args)
        print(f"discover_only=1 discovered_hf_sources={len(hf_sources)}")
        return

    out = Path(args.out)
    for split in ["train", "val", "test"]:
        for cls in ["ai", "real"]:
            (out / split / cls).mkdir(parents=True, exist_ok=True)

    targets = {
        "train": {"ai": args.train_per_class, "real": args.train_per_class},
        "val": {"ai": args.val_per_class, "real": args.val_per_class},
        "test": {"ai": args.test_per_class, "real": args.test_per_class},
    }
    counts: Dict[str, Dict[str, int]] = count_existing(out)
    max_per_source_split_class = int(args.max_per_source_split_class)
    if max_per_source_split_class <= 0:
        max_per_source_split_class = max(1, int(math.ceil(float(args.max_per_source_class) / float(len(SPLITS)))))
    print(
        "existing_counts "
        + " ".join([f"{s}/{c}={counts[s][c]}" for s in ["train", "val", "test"] for c in ["ai", "real"]])
    )

    # Global dedupe to prevent leakage across splits.
    deduper = ImageDeduper.from_output(out, splits=SPLITS, classes=CLASSES)
    quality_policy = ImageQualityPolicy(
        min_side=args.min_side,
        max_aspect_ratio=args.max_aspect_ratio,
        min_entropy=args.min_entropy,
    )
    manifest_policy = source_manifest_policy(args)
    source_manifest_path = out / "dataset_source_manifest.jsonl"
    latest_manifest = load_latest_source_manifest(source_manifest_path)

    global_rejects: DefaultDict[str, int] = defaultdict(int)
    source_reports: List[Dict[str, object]] = []

    def append_manifest_entry(entry: Dict[str, object]) -> None:
        append_source_manifest_entry(
            source_manifest_path,
            {
                "cache_dir": str(cache_dir) if cache_dir is not None else "",
                "manifest_version": 2,
                "policy": manifest_policy,
                **entry,
            },
        )

    def try_accept_and_save(
        img: Image.Image,
        cls: str,
        src: str,
        source_counts: Dict[str, int],
        source_split_counts: Dict[str, Dict[str, int]],
    ) -> bool:
        if done(counts, targets):
            return False
        if source_counts[cls] >= args.max_per_source_class:
            global_rejects["source_class_cap"] += 1
            return False

        ok, reason = passes_quality_filters(img, quality_policy)
        if not ok:
            global_rejects[reason] += 1
            return False

        duplicate_reason = deduper.duplicate_reason(
            img,
            cls=cls,
            near_hamming=args.near_hamming,
            near_window=args.near_window,
        )
        if duplicate_reason is not None:
            global_rejects[duplicate_reason] += 1
            return False

        split = next_split_for_source_class(
            counts,
            targets,
            source_split_counts,
            cls,
            rng,
            max_per_source_split_class=max_per_source_split_class,
        )
        if split is None:
            split = next_split_for_class(counts, targets, cls, rng)
        if split is None:
            global_rejects["no_split_needed"] += 1
            return False

        deduper.remember(img, cls=cls)

        n = counts[split][cls]
        src_tag = source_tag(src)
        dst = out / split / cls / f"source={src_tag}__{split}_{cls}_{n:07d}.jpg"
        save_img(img, dst, quality=args.jpeg_quality)
        counts[split][cls] += 1
        source_counts[cls] += 1
        source_split_counts[split][cls] += 1
        return True

    hf_sources = build_source_list(args)
    if not hf_sources:
        raise SystemExit("no_hf_sources_resolved: enable --discover-hf or provide HF sources cache/file")
    print(f"hf_source_candidates={len(hf_sources)}")

    consecutive_source_failures = 0
    for src_idx, src in enumerate(hf_sources, start=1):
        if done(counts, targets):
            break
        if should_skip_source_from_manifest(latest_manifest.get(src), manifest_policy):
            print(f"skip_source={src} reason=manifest_exhausted")
            append_manifest_entry(
                {
                    "source": src,
                    "source_index": int(src_idx),
                    "type": "hf",
                    "status": "skipped_manifest",
                    "reason": "manifest_exhausted",
                    "skip_future_runs": True,
                    "started_utc": utc_now_iso(),
                    "finished_utc": utc_now_iso(),
                }
            )
            continue
        repo_pause = args.repo_base_pause_ms + random.randint(0, max(args.repo_jitter_ms, 0))
        if repo_pause > 0:
            time.sleep(repo_pause / 1000.0)
        source_counts = {"ai": 0, "real": 0}
        source_split_counts = {split: {cls: 0 for cls in CLASSES} for split in SPLITS}
        source_started_utc = utc_now_iso()
        source_started_monotonic = time.time()
        counts_before = counts_snapshot(counts)
        try:
            loaded_source = load_hf_dataset_source(
                src,
                token=token,
                streaming=args.streaming,
                cache_dir=(args.cache_dir or None),
            )
        except Exception as e:
            msg = str(e)
            print(f"skip_source={src} reason={msg}")
            append_manifest_entry(
                {
                    "source": src,
                    "source_index": int(src_idx),
                    "type": "hf",
                    "status": "load_failed",
                    "reason": msg,
                    "skip_future_runs": False,
                    "started_utc": source_started_utc,
                    "finished_utc": utc_now_iso(),
                    "elapsed_sec": round(float(time.time() - source_started_monotonic), 3),
                    "counts_before": counts_before,
                    "counts_after": counts_snapshot(counts),
                }
            )
            if likely_rate_limited(msg):
                cooldown = int(args.repo_cooldown_ms)
                print(f"cooldown_ms={cooldown} reason=rate_limited")
                time.sleep(cooldown / 1000.0)
                consecutive_source_failures = 0
            elif likely_transient_hf_error(msg):
                consecutive_source_failures += 1
                if consecutive_source_failures >= args.max_consecutive_failures:
                    cooldown = int(args.transient_error_cooldown_ms)
                    print(f"cooldown_ms={cooldown} reason=transient_failures")
                    if cooldown > 0:
                        time.sleep(cooldown / 1000.0)
                    consecutive_source_failures = 0
            else:
                consecutive_source_failures = 0
            continue
        consecutive_source_failures = 0

        split = loaded_source.split
        try:
            image_field, label_field = find_fields(split)
        except Exception as e:
            print(f"skip_source={src} reason={e}")
            append_manifest_entry(
                {
                    "source": src,
                    "source_index": int(src_idx),
                    "type": "hf",
                    "status": "field_inference_failed",
                    "reason": str(e),
                    "skip_future_runs": False,
                    "started_utc": source_started_utc,
                    "finished_utc": utc_now_iso(),
                    "elapsed_sec": round(float(time.time() - source_started_monotonic), 3),
                    "split_name": loaded_source.split_name,
                    "counts_before": counts_before,
                    "counts_after": counts_snapshot(counts),
                }
            )
            continue
        resolve_label = build_label_resolver(split, label_field)
        try:
            normalized_split = normalize_image_dataset_split(
                split,
                label_field=label_field,
                resolve_label=resolve_label,
                show_progress=not args.quiet_progress,
            )
        except Exception as e:
            print(f"skip_source={src} reason=normalize_failed:{e}")
            append_manifest_entry(
                {
                    "source": src,
                    "source_index": int(src_idx),
                    "type": "hf",
                    "status": "normalize_failed",
                    "reason": str(e),
                    "skip_future_runs": False,
                    "started_utc": source_started_utc,
                    "finished_utc": utc_now_iso(),
                    "elapsed_sec": round(float(time.time() - source_started_monotonic), 3),
                    "split_name": loaded_source.split_name,
                    "counts_before": counts_before,
                    "counts_after": counts_snapshot(counts),
                }
            )
            continue
        normalized_source = LoadedDatasetSource(
            source_id=loaded_source.source_id,
            split_name=loaded_source.split_name,
            split=normalized_split,
            streaming=loaded_source.streaming,
        )

        def _extract_payload(ex: dict) -> tuple[object, object, str | None]:
            decoded = open_example_image(ex, image_field)
            return (
                ex.get("_normalized_label"),
                decoded,
                None if decoded is not None else "decode_fail",
            )

        loop_result = run_source_acceptance_loop(
            iter_source_examples(
                normalized_source,
                seed=args.seed + src_idx * 137,
                shuffle_buffer_size=args.stream_buffer_size,
                max_samples=args.max_samples_per_source,
            ),
            is_done=lambda: done(counts, targets),
            max_unique_per_source=args.max_unique_per_source,
            global_rejects=global_rejects,
            try_accept_and_save=try_accept_and_save,
            extract_payload=_extract_payload,
            source_name=src,
            source_counts=source_counts,
            source_split_counts=source_split_counts,
            acceptance_warmup_samples=args.acceptance_warmup_samples,
            min_acceptance_rate=args.min_acceptance_rate,
        )
        if "low_acceptance_rate" in loop_result["rejections"]:
            print(
                f"early_stop_source={src} reason=low_acceptance_rate "
                f"accepted={loop_result['accepted_total']} processed={loop_result['processed_total']} "
                f"rate={loop_result['acceptance_rate']:.5f}"
            )

        report = make_source_report(
            source=src,
            source_type="hf",
            source_counts=source_counts,
            source_split_counts=source_split_counts,
            loop_result=loop_result,
        )
        source_reports.append(report)
        append_manifest_entry(
            {
                **report,
                "source_index": int(src_idx),
                "split_name": loaded_source.split_name,
                "image_field": image_field,
                "label_field": label_field,
                "started_utc": source_started_utc,
                "finished_utc": utc_now_iso(),
                "elapsed_sec": round(float(time.time() - source_started_monotonic), 3),
                "counts_before": counts_before,
                "counts_after": counts_snapshot(counts),
                "skip_future_runs_reason": "low_acceptance_or_exhausted" if loop_result["skip_future_runs"] else "",
            },
        )
        print(
            f"loaded_source={src} accepted_ai={source_counts['ai']} accepted_real={source_counts['real']} "
            f"processed={report['processed_total']} rejected={sum(report['rejections'].values())} acceptance_rate={report['acceptance_rate']:.5f}"
        )

    raw_counts = count_output_files(out, include_hardneg=False)
    for split in ["train", "val", "test"]:
        for cls in ["ai", "real"]:
            n = raw_counts[split][cls]
            print(f"{split}/{cls}={n}")
            if n < targets[split][cls]:
                print(f"warning_shortfall split={split} cls={cls} have={n} need={targets[split][cls]}")

    summary, hf_sources_per_split_class, shortfalls = build_summary(
        targets=targets,
        raw_counts=raw_counts,
        global_rejections=dict(global_rejects),
        source_reports=source_reports,
        hf_sources=hf_sources,
        cache_dir=str(cache_dir) if cache_dir is not None else "",
        args=args,
        source_manifest_path=source_manifest_path,
        manifest_policy=manifest_policy,
    )
    hf_sources_with_accepted = int(summary["hf_sources_with_accepted"])
    hf_sources_ai = int(summary["hf_sources_ai"])
    hf_sources_real = int(summary["hf_sources_real"])

    if args.min_hf_sources_with_accepted > 0 and hf_sources_with_accepted < int(args.min_hf_sources_with_accepted):
        raise SystemExit(
            f"hf_source_diversity_too_low accepted_sources={hf_sources_with_accepted} required={args.min_hf_sources_with_accepted}"
        )
    if args.min_hf_sources_per_class > 0:
        if hf_sources_ai < int(args.min_hf_sources_per_class) or hf_sources_real < int(args.min_hf_sources_per_class):
            raise SystemExit(
                "hf_source_class_diversity_too_low "
                f"ai_sources={hf_sources_ai} real_sources={hf_sources_real} required={args.min_hf_sources_per_class}"
            )
    if args.min_hf_sources_per_split_class > 0:
        missing_buckets = []
        for split in SPLITS:
            for cls in CLASSES:
                have_sources = int(hf_sources_per_split_class[split][cls])
                if have_sources < int(args.min_hf_sources_per_split_class):
                    missing_buckets.append(f"{split}/{cls}:{have_sources}<{args.min_hf_sources_per_split_class}")
        if missing_buckets:
            raise SystemExit("hf_source_split_diversity_too_low " + ",".join(missing_buckets))

    full_targets_ok = bool(summary["full_targets_ok"])

    elapsed_sec = float(time.time() - start_time)
    accepted_ai_total = int(sum(summary["final_counts"][s]["ai"] for s in ["train", "val", "test"]))
    accepted_real_total = int(sum(summary["final_counts"][s]["real"] for s in ["train", "val", "test"]))

    hardneg_counts = generate_hard_negatives(
        out=out,
        targets=targets,
        hardneg_fraction=args.hardneg_fraction,
        jpeg_quality=args.jpeg_quality,
        seed=args.seed,
    )
    for cls in CLASSES:
        print(f"hard_negatives_{cls}={hardneg_counts.get(cls, 0)}")
    summary["hardneg_counts"] = {cls: int(hardneg_counts.get(cls, 0)) for cls in CLASSES}
    summary["output_counts_with_hardneg"] = count_output_files(out, include_hardneg=True)
    summary["builder_policy"] = {
        "max_per_source_class": int(args.max_per_source_class),
        "max_per_source_split_class": int(max_per_source_split_class),
        "min_hf_sources_with_accepted": int(args.min_hf_sources_with_accepted),
        "min_hf_sources_per_class": int(args.min_hf_sources_per_class),
        "min_hf_sources_per_split_class": int(args.min_hf_sources_per_split_class),
    }

    run_summary = {
        "elapsed_sec": round(elapsed_sec, 2),
        "hf_sources_used": int(hf_sources_with_accepted),
        "accepted_ai_total": accepted_ai_total,
        "accepted_real_total": accepted_real_total,
        "gate_hf_diversity": "pass",
        "gate_hf_per_class": "pass",
        "gate_full_targets": "pass" if full_targets_ok else "fail",
        "report_path": str((out / "dataset_build_report.json").resolve()),
    }

    write_summary_files(out, summary, run_summary)
    print(
        "run_summary "
        f"elapsed_sec={run_summary['elapsed_sec']} "
        f"hf_sources_used={run_summary['hf_sources_used']} "
        f"accepted_ai_total={run_summary['accepted_ai_total']} "
        f"accepted_real_total={run_summary['accepted_real_total']} "
        f"gate_hf_diversity={run_summary['gate_hf_diversity']} "
        f"gate_hf_per_class={run_summary['gate_hf_per_class']} "
        f"gate_full_targets={run_summary['gate_full_targets']}"
    )
    print(f"report={out / 'dataset_build_report.json'}")
    print(f"run_summary_file={out / 'dataset_run_summary.json'}")
    if args.require_full_targets and not full_targets_ok:
        raise SystemExit("dataset_incomplete: " + ",".join(shortfalls))
    print("dataset_ready", out)


if __name__ == "__main__":
    main()
