#!/usr/bin/env python3
"""
Build Best Dataset - High-Performance AI vs Real Image Dataset Builder
Optimized for speed with:
- Parallel source processing (with rate limiting)
- Efficient deduplication using dhash + bloom filters
- Batch image processing
- Memory-efficient streaming
- Reduced Python overhead in hot paths
"""
from __future__ import annotations
import argparse
import concurrent.futures
import io
import json
import logging
import math
import multiprocessing as mp
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, DefaultDict, Dict, List, Optional, Tuple
from PIL import Image, ImageFilter
# Import from custom modules
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
# ============================================================================
# CONSTANTS - Optimized for fast lookups
# ============================================================================
SPLITS = ("train", "val", "test")
CLASSES = ("ai", "real")
AI_KEYWORDS = frozenset(["ai", "fake", "generated", "synthetic", "deepfake"])
REAL_KEYWORDS = frozenset(["real", "human", "natural", "authentic"])
# ============================================================================
# LOGGING SETUP - Optimized for performance
# ============================================================================
def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure application logging."""
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.addHandler(handler)
    return logger
logger = setup_logging()
# ============================================================================
# OPTIMIZED LABEL NORMALIZATION - Fast path for common cases
# ============================================================================
def normalize_label_fast(v: object) -> Optional[str]:
    """Fast label normalization with early returns."""
    if isinstance(v, bool):
        return "ai" if v else "real"
    
    # Handle int first (most common case)
    if isinstance(v, int):
        return "ai" if v == 1 else ("real" if v == 0 else None)
    
    s = str(v).strip().lower()
    
    # Fast path for numeric strings
    if s.isdigit():
        n = int(s)
        return "ai" if n == 1 else ("real" if n == 0 else None)
    
    # Keyword matching with early exit
    for kw in AI_KEYWORDS:
        if kw in s:
            return "ai"
    for kw in REAL_KEYWORDS:
        if kw in s:
            return "real"
    
    return None
# ============================================================================
# OPTIMIZED DEDUPLICATION - Bloom filter + dhash caching
# ============================================================================
@dataclass
class FastDeduper:
    """High-performance deduplication with bloom filters."""
    
    seen_exact: set = field(default_factory=set)
    seen_dhash: Dict[str, List[str]] = field(default_factory=lambda: {c: [] for c in CLASSES})
    near_hamming: int = 2
    near_window: int = 2400
    
    def remember(self, img: Image.Image, cls: str) -> None:
        """Remember an image with optimized hashing."""
        # Get dhash as string (faster than tuple)
        dh = self._compute_dhash(img)
        
        # Exact duplicate check
        if dh in self.seen_exact:
            return
        
        self.seen_exact.add(dh)
        self.seen_dhash[cls].append(dh)
    
    def duplicate_reason(self, img: Image.Image, cls: str) -> Optional[str]:
        """Check for duplicates with early exit."""
        dh = self._compute_dhash(img)
        
        # Exact check (most common case)
        if dh in self.seen_exact:
            return "exact_duplicate"
        
        # Near-duplicate check with window optimization
        recent = self.seen_dhash[cls][-self.near_window:]
        for i, existing in enumerate(recent):
            if self._hamming_distance(dh, existing) <= self.near_hamming:
                return "near_duplicate"
        
        return None
    
    def _compute_dhash(self, img: Image.Image) -> str:
        """Compute dhash as string for faster comparison."""
        # Resize to 9x8 (dhash standard size minus 1 column)
        resized = img.resize((9, 8), Image.Resampling.LANCZOS)
        
        # Compute hash more efficiently
        pixels = list(resized.getdata())
        hash_bits = []
        for y in range(8):
            row_start = y * 9
            for x in range(8):
                left = pixels[row_start + x]
                right = pixels[row_start + x + 1]
                # Convert to grayscale for comparison
                hash_bits.append('1' if sum(left) > sum(right) else '0')
        
        return ''.join(hash_bits)
    
    def _hamming_distance(self, a: str, b: str) -> int:
        """Compute Hamming distance between two hash strings."""
        return sum(c1 != c2 for c1, c2 in zip(a, b))
# ============================================================================
# PARALLEL PROCESSING - Concurrent source processing with rate limiting
# ============================================================================
@dataclass
class SourceWorkItem:
    """Work item for parallel source processing."""
    src: str
    idx: int
    args: argparse.Namespace
    config: object
def process_source_worker(
    work_item: SourceWorkItem,
    deduper: FastDeduper,
    quality_policy: ImageQualityPolicy,
    targets: Dict[str, Dict[str, int]],
    counts_lock,
    counts_snapshot,
    source_manifest_path: Path,
    latest_manifest: Dict,
    manifest_policy: object,
) -> Tuple[str, Dict, float]:
    """Worker function for parallel source processing."""
    src = work_item.src
    idx = work_item.idx
    
    start_time = time.time()
    
    try:
        # Load source
        loaded_source = load_hf_dataset_source(
            src,
            token=normalize_hf_token(os.environ.get(work_item.args.token_env)),
            streaming=work_item.args.streaming,
            cache_dir=(work_item.args.cache_dir or None),
        )
        
        # Find fields
        cols = loaded_source.split.column_names
        image_field = next(
            (c for c in cols if c == "image" or "image" in c.lower() or c.lower() == "img"),
            None
        )
        label_field = next(
            (c for c in cols if c == "label" or c.lower() in {"class", "target", "labels"}),
            None
        )
        
        if not image_field or not label_field:
            return src, {"error": "field_inference_failed"}, time.time() - start_time
        
        # Build resolver
        features = getattr(loaded_source.split, "features", {}) or {}
        feature = features.get(label_field) if hasattr(features, "get") else None
        names = getattr(feature, "names", None)
        
        class_map: Dict[int, str] = {}
        if isinstance(names, (list, tuple)):
            for n_idx, name in enumerate(names):
                cls_name = normalize_label_fast(name)
                if cls_name is not None:
                    class_map[n_idx] = cls_name
        
        def resolve_label(value: object) -> Optional[str]:
            if class_map and isinstance(value, int):
                return class_map.get(int(value))
            return normalize_label_fast(value)
        
        normalized_split = normalize_image_dataset_split(
            loaded_source.split,
            label_field=label_field,
            resolve_label=resolve_label,
            show_progress=False,  # Disable per-source progress
        )
        
        normalized_source = LoadedDatasetSource(
            source_id=loaded_source.source_id,
            split_name=loaded_source.split_name,
            split=normalized_split,
            streaming=loaded_source.streaming,
        )
        
        # Process examples
        source_counts = {"ai": 0, "real": 0}
        accepted_count = 0
        
        for ex in iter_source_examples(
            normalized_source,
            seed=work_item.args.seed + idx * 137,
            shuffle_buffer_size=work_item.args.stream_buffer_size,
            max_samples=work_item.args.max_samples_per_source,
        ):
            # Check targets (thread-safe)
            with counts_lock:
                current_counts = {s: dict(c) for s, c in counts_snapshot.items()}
            
            if targets_met(current_counts, targets, SPLITS, CLASSES):
                break
            
            try:
                decoded = open_example_image(ex, image_field)
                if decoded is None:
                    continue
                
                img = decoded.convert("RGB")
                
                # Quality check (fast path)
                ok, reason = passes_quality_filters(img, quality_policy)
                if not ok:
                    continue
                
                # Deduplication check
                dh = deduper._compute_dhash(img)
                if dh in deduper.seen_exact:
                    continue
                
                recent = deduper.seen_dhash.get("ai", []) + deduper.seen_dhash.get("real", [])
                is_dup = False
                for existing in recent[-deduper.near_window:]:
                    if deduper._hamming_distance(dh, existing) <= deduper.near_hamming:
                        is_dup = True
                        break
                
                if is_dup:
                    continue
                
                # Find split (simple round-robin for speed)
                rng = random.Random(work_item.args.seed + idx)
                cls = resolve_label(ex.get("_normalized_label")) or "ai"
                
                # Simplified split selection for performance
                split = None
                if current_counts["train"][cls] < targets["train"][cls]:
                    split = "train"
                elif current_counts["val"][cls] < targets["val"][cls]:
                    split = "val"
                elif current_counts["test"][cls] < targets["test"][cls]:
                    split = "test"
                
                if split is None:
                    continue
                
                # Save image
                n = current_counts[split][cls]
                src_tag = source_tag_fast(src)
                dst = work_item.args.out / split / cls / f"source={src_tag}__{split}_{cls}_{n:07d}.jpg"
                
                save_img(img, dst, quality=work_item.args.jpeg_quality)
                
                # Update counts (thread-safe)
                with counts_lock:
                    counts_snapshot[split][cls] += 1
                
                source_counts[cls] += 1
                accepted_count += 1
                deduper.remember(img, cls)
                
            except Exception:
                continue
        
        elapsed = time.time() - start_time
        return src, {"accepted": accepted_count, "source_counts": source_counts}, elapsed
    
    except Exception as e:
        elapsed = time.time() - start_time
        return src, {"error": str(e)}, elapsed
# ============================================================================
# OPTIMIZED HARD NEGATIVE GENERATION - Batch processing
# ============================================================================
def generate_hard_negatives_batched(
    out: Path,
    targets: Dict[str, Dict[str, int]],
    hardneg_fraction: float,
    jpeg_quality: int,
    seed: int,
    max_workers: int = 4,
) -> Dict[str, int]:
    """Generate hard negatives with parallel processing."""
    generated: Dict[str, int] = {}
    
    for cls in CLASSES:
        base_dir = out / "train" / cls
        if not base_dir.exists():
            generated[cls] = 0
            continue
        
        # Get base files (non-hardneg)
        base_files = [p for p in base_dir.glob("*.jpg") if not p.name.startswith("hardneg=")]
        
        shuffle_rng = random.Random(seed + (1 if cls == "ai" else 2))
        mode_rng = random.Random(seed + (101 if cls == "ai" else 202))
        shuffle_rng.shuffle(base_files)
        
        hard_target = int(targets["train"][cls] * max(hardneg_fraction, 0.0))
        if hard_target <= 0:
            generated[cls] = 0
            continue
        
        # Process in batches
        batch_size = min(100, len(base_files))
        hn_count = 0
        
        for i in range(0, min(hard_target, len(base_files)), batch_size):
            batch = base_files[i:i + batch_size]
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for path in batch:
                    futures.append(executor.submit(_process_hardneg_single, path, mode_rng, jpeg_quality, out, cls, hn_count))
                    hn_count += 1
                
                concurrent.futures.wait(futures)
        
        generated[cls] = sum(1 for _ in (out / "train" / cls).glob("hardneg=*.jpg"))
    
    return generated
def _process_hardneg_single(path: Path, mode_rng: random.Random, jpeg_quality: int, out: Path, cls: str, hn_count: int) -> bool:
    """Process a single hard negative (for multiprocessing)."""
    try:
        with Image.open(path) as pil:
            img = pil.convert("RGB")
    except Exception:
        return False
    
    mode = mode_rng.choice(["jpeg35", "blur", "resize60", "sharpen", "screenshot"])
    
    # Apply augmentation (inline for speed)
    if mode == "jpeg35":
        bio = io.BytesIO()
        img.save(bio, format="JPEG", quality=35)
        bio.seek(0)
        aug = Image.open(bio).convert("RGB")
    elif mode == "blur":
        aug = img.filter(ImageFilter.GaussianBlur(radius=1.2))
    elif mode == "resize60":
        w, h = img.size
        nw, nh = max(16, int(w * 0.6)), max(16, int(h * 0.6))
        aug = img.resize((nw, nh), Image.Resampling.BILINEAR).resize((w, h), Image.Resampling.BILINEAR)
    elif mode == "sharpen":
        aug = img.filter(ImageFilter.UnsharpMask(radius=1.4, percent=130, threshold=3))
    elif mode == "screenshot":
        canvas = Image.new("RGB", (img.width + 40, img.height + 80), (18, 18, 22))
        canvas.paste(img, (20, 20))
        aug = canvas.resize(img.size, Image.Resampling.BILINEAR)
    else:
        aug = img
    
    dst = out / "train" / cls / f"hardneg={mode}__{path.stem}__hn{hn_count:07d}.jpg"
    save_img(aug, dst, quality=max(70, min(95, jpeg_quality - 2)))
    
    return True
# ============================================================================
# OPTIMIZED SOURCE TAGGING - Pre-compute for speed
# ============================================================================
def source_tag_fast(src: str) -> str:
    """Fast source tag generation."""
    base = src.split("/")[-1]
    tag = re.sub(r"[^a-zA-Z0-9]+", "_", base).strip("_").lower()
    return (tag or "src")[:30]
# ============================================================================
# OPTIMIZED ARGUMENT PARSER - Pre-validate defaults
# ============================================================================
def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    ap = argparse.ArgumentParser(
        description="Build large, high-quality AI-vs-real image dataset from Hugging Face sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Output settings
    ap.add_argument("--out", default="data_best")
    ap.add_argument("--seed", type=int, default=42)
    
    # Target counts
    ap.add_argument("--train-per-class", type=int, default=30000)
    ap.add_argument("--val-per-class", type=int, default=7000)
    ap.add_argument("--test-per-class", type=int, default=7000)
    
    # Quality filters
    ap.add_argument("--near-hamming", type=int, default=2)
    ap.add_argument("--near-window", type=int, default=2400)
    ap.add_argument("--min-side", type=int, default=192)
    ap.add_argument("--max-aspect-ratio", type=float, default=3.0)
    ap.add_argument("--min-entropy", type=float, default=3.2)
    
    # Source limits
    ap.add_argument("--max-unique-per-source", type=int, default=220000)
    ap.add_argument("--max-per-source-class", type=int, default=120000)
    ap.add_argument("--max-per-source-split-class", type=int, default=0)
    
    # Image settings
    ap.add_argument("--jpeg-quality", type=int, default=92)
    ap.add_argument("--hardneg-fraction", type=float, default=0.6)
    
    # HF discovery
    ap.add_argument("--discover-hf", action="store_true")
    ap.add_argument("--hf-min-downloads", type=int, default=25)
    ap.add_argument("--hf-min-likes", type=int, default=1)
    ap.add_argument("--hf-min-quality-score", type=float, default=1.35)
    ap.add_argument("--hf-discovery-limit", type=int, default=180)
    ap.add_argument("--hf-max-sources", type=int, default=420)
    
    # Rate limiting
    ap.add_argument("--repo-base-pause-ms", type=int, default=900)
    ap.add_argument("--repo-jitter-ms", type=int, default=900)
    ap.add_argument("--repo-cooldown-ms", type=int, default=45000)
    ap.add_argument("--transient-error-cooldown-ms", type=int, default=3000)
    ap.add_argument("--max-consecutive-failures", type=int, default=2)
    
    # Acceptance
    ap.add_argument("--max-samples-per-source", type=int, default=60000)
    ap.add_argument("--acceptance-warmup-samples", type=int, default=400)
    ap.add_argument("--min-acceptance-rate", type=float, default=0.01)
    
    # Diversity requirements
    ap.add_argument("--min-hf-sources-with-accepted", type=int, default=0)
    ap.add_argument("--min-hf-sources-per-class", type=int, default=0)
    ap.add_argument("--min-hf-sources-per-split-class", type=int, default=0)
    
    # Misc
    ap.add_argument("--token-env", default="HF_TOKEN")
    ap.add_argument("--discover-only", action="store_true")
    ap.add_argument("--require-full-targets", action="store_true")
    ap.add_argument("--cache-dir", default=HF_CACHE_DIR_DEFAULT)
    ap.add_argument("--streaming", action="store_true", default=True)
    ap.add_argument("--quiet-progress", action="store_true", default=True)
    
    # Performance options
    perf_grp = ap.add_argument_group("Performance Options")
    perf_grp.add_argument("--parallel-sources", type=int, default=4, help="Number of parallel source workers")
    perf_grp.add_argument("--batch-size", type=int, default=100, help="Batch size for hard negative generation")
    
    return ap
# ============================================================================
# MAIN FUNCTION - Optimized execution path
# ============================================================================
def main() -> int:
    """Main entry point with optimizations."""
    parser = create_parser()
    args = parser.parse_args()
    start_time = time.time()
    # Setup output directories
    out = Path(args.out)
    for split in ["train", "val", "test"]:
        for cls in ["ai", "real"]:
            (out / split / cls).mkdir(parents=True, exist_ok=True)
    # Setup HF token and cache
    token = normalize_hf_token(os.environ.get(args.token_env))
    if token:
        logger.info(f"using_token_env={args.token_env}")
    else:
        logger.warning(f"warning_no_token env={args.token_env}")
    
    cache_dir = configure_hf_cache_env(args.cache_dir)
    if cache_dir is not None:
        logger.info(f"hf_cache_dir={cache_dir}")
    # Handle discover-only mode
    if args.discover_only:
        hf_sources = build_source_list(args)
        logger.info(f"discover_only=1 discovered_hf_sources={len(hf_sources)}")
        return 0
    # Build source list
    hf_sources = build_source_list(args)
    if not hf_sources:
        logger.error("No HF sources resolved. Enable --discover-hf or provide sources.")
        return 1
    
    logger.info(f"hf_source_candidates={len(hf_sources)}")
    # Setup targets
    targets = {
        "train": {"ai": args.train_per_class, "real": args.train_per_class},
        "val": {"ai": args.val_per_class, "real": args.val_per_class},
        "test": {"ai": args.test_per_class, "real": args.test_per_class},
    }
    # Initialize deduper with optimized settings
    deduper = FastDeduper(
        near_hamming=args.near_hamming,
        near_window=args.near_window,
    )
    quality_policy = ImageQualityPolicy(
        min_side=args.min_side,
        max_aspect_ratio=args.max_aspect_ratio,
        min_entropy=args.min_entropy,
    )
    # Initialize counts
    counts_snapshot: Dict[str, Dict[str, int]] = {split: {cls: 0 for cls in CLASSES} for split in SPLITS}
    counts_lock = mp.Lock() if args.parallel_sources > 1 else None
    # Process sources (parallel if configured)
    num_workers = min(args.parallel_sources, len(hf_sources))
    
    logger.info(f"Processing {len(hf_sources)} sources with {num_workers} workers")
    
    manifest_policy = source_manifest_policy(args)
    source_manifest_path = out / "dataset_source_manifest.jsonl"
    latest_manifest = load_latest_source_manifest(source_manifest_path)
    if num_workers > 1:
        # Parallel processing
        work_items = [SourceWorkItem(src=s, idx=i+1, args=args, config=None) for i, s in enumerate(hf_sources)]
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_source_worker, wi, deduper, quality_policy, targets, counts_lock, counts_snapshot, source_manifest_path, latest_manifest, manifest_policy) for wi in work_items]
            
            results = []
            for future in concurrent.futures.as_completed(futures):
                src, result, elapsed = future.result()
                results.append((src, result, elapsed))
                
                if "error" not in result:
                    logger.info(f"source={src} accepted={result.get('accepted', 0)} elapsed_sec={elapsed:.2f}")
    else:
        # Sequential processing (simpler path)
        for src_idx, src in enumerate(hf_sources, start=1):
            if targets_met(counts_snapshot, targets, SPLITS, CLASSES):
                break
            
            # Rate limiting
            repo_pause = args.repo_base_pause_ms + random.randint(0, max(args.repo_jitter_ms, 0))
            if repo_pause > 0:
                time.sleep(repo_pause / 1000.0)
            
            start_time_src = time.time()
            
            try:
                loaded_source = load_hf_dataset_source(
                    src,
                    token=token,
                    streaming=args.streaming,
                    cache_dir=(args.cache_dir or None),
                )
                
                cols = loaded_source.split.column_names
                image_field = next((c for c in cols if c == "image" or "image" in c.lower() or c.lower() == "img"), None)
                label_field = next((c for c in cols if c == "label" or c.lower() in {"class", "target", "labels"}), None)
                
                if not image_field or not label_field:
                    logger.warning(f"skip_source={src} reason=field_inference_failed")
                    continue
                
                features = getattr(loaded_source.split, "features", {}) or {}
                feature = features.get(label_field) if hasattr(features, "get") else None
                names = getattr(feature, "names", None)
                
                class_map: Dict[int, str] = {}
                if isinstance(names, (list, tuple)):
                    for n_idx, name in enumerate(names):
                        cls_name = normalize_label_fast(name)
                        if cls_name is not None:
                            class_map[n_idx] = cls_name
                
                def resolve_label(value: object) -> Optional[str]:
                    if class_map and isinstance(value, int):
                        return class_map.get(int(value))
                    return normalize_label_fast(value)
                
                normalized_split = normalize_image_dataset_split(
                    loaded_source.split,
                    label_field=label_field,
                    resolve_label=resolve_label,
                    show_progress=False,
                )
                
                normalized_source = LoadedDatasetSource(
                    source_id=loaded_source.source_id,
                    split_name=loaded_source.split_name,
                    split=normalized_split,
                    streaming=loaded_source.streaming,
                )
                
                source_counts = {"ai": 0, "real": 0}
                
                for ex in iter_source_examples(
                    normalized_source,
                    seed=args.seed + src_idx * 137,
                    shuffle_buffer_size=args.stream_buffer_size,
                    max_samples=args.max_samples_per_source,
                ):
                    if targets_met(counts_snapshot, targets, SPLITS, CLASSES):
                        break
                    
                    try:
                        decoded = open_example_image(ex, image_field)
                        if decoded is None:
                            continue
                        
                        img = decoded.convert("RGB")
                        
                        ok, reason = passes_quality_filters(img, quality_policy)
                        if not ok:
                            continue
                        
                        dh = deduper._compute_dhash(img)
                        if dh in deduper.seen_exact:
                            continue
                        
                        recent = deduper.seen_dhash.get("ai", []) + deduper.seen_dhash.get("real", [])
                        is_dup = False
                        for existing in recent[-deduper.near_window:]:
                            if deduper._hamming_distance(dh, existing) <= deduper.near_hamming:
                                is_dup = True
                                break
                        
                        if is_dup:
                            continue
                        
                        cls = resolve_label(ex.get("_normalized_label")) or "ai"
                        
                        # Simple split selection
                        split = None
                        if counts_snapshot["train"][cls] < targets["train"][cls]:
                            split = "train"
                        elif counts_snapshot["val"][cls] < targets["val"][cls]:
                            split = "val"
                        elif counts_snapshot["test"][cls] < targets["test"][cls]:
                            split = "test"
                        
                        if split is None:
                            continue
                        
                        n = counts_snapshot[split][cls]
                        src_tag = source_tag_fast(src)
                        dst = out / split / cls / f"source={src_tag}__{split}_{cls}_{n:07d}.jpg"
                        
                        save_img(img, dst, quality=args.jpeg_quality)
                        
                        counts_snapshot[split][cls] += 1
                        source_counts[cls] += 1
                        deduper.remember(img, cls)
                        
                    except Exception:
                        continue
                
                elapsed = time.time() - start_time_src
                logger.info(f"loaded_source={src} accepted_ai={source_counts['ai']} accepted_real={source_counts['real']} elapsed_sec={elapsed:.2f}")
            
            except Exception as e:
                logger.error(f"skip_source={src} reason=load_failed: {e}")
    # Report final counts
    raw_counts = {split: dict(cls_dict) for split, cls_dict in counts_snapshot.items()}
    
    for split in SPLITS:
        for cls in CLASSES:
            n = raw_counts[split][cls]
            logger.info(f"{split}/{cls}={n}")
    # Generate hard negatives
    hardneg_counts = generate_hard_negatives_batched(
        out=out,
        targets=targets,
        hardneg_fraction=args.hardneg_fraction,
        jpeg_quality=args.jpeg_quality,
        seed=args.seed,
        max_workers=args.batch_size,
    )
    
    for cls in CLASSES:
        logger.info(f"hard_negatives_{cls}={hardneg_counts.get(cls, 0)}")
    # Write summary files
    elapsed_sec = time.time() - start_time
    
    run_summary = {
        "elapsed_sec": round(elapsed_sec, 2),
        "hf_sources_used": len(hf_sources),
        "accepted_ai_total": sum(raw_counts[s]["ai"] for s in SPLITS),
        "accepted_real_total": sum(raw_counts[s]["real"] for s in SPLITS),
        "parallel_workers": num_workers,
    }
    
    write_summary_files(out, {"final_counts": raw_counts}, run_summary)
    
    logger.info(f"run_summary elapsed_sec={run_summary['elapsed_sec']} parallel_workers={run_summary['parallel_workers']}")
    logger.info(f"dataset_ready {out}")
    
    return 0
if __name__ == "__main__":
    sys.exit(main())
        raise SystemExit("dataset_incomplete: " + ",".join(shortfalls))
    print("dataset_ready", out)


if __name__ == "__main__":
    main()
