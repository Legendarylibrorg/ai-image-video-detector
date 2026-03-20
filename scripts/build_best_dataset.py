from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import time
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

from datasets import load_dataset
from PIL import Image, ImageFilter

try:
    from huggingface_hub import HfApi
except Exception:  # pragma: no cover - optional dependency path
    HfApi = None  # type: ignore[assignment]


DEFAULT_SOURCES = [
    "Hemg/AI-Generated-vs-Real-Images-Datasets",
    "dragonintelligence/CIFAKE-image-dataset",
    "batgre/CIFAKE",
    "Ronduck/real-fake-images-deduplicated",
    "JamieWithofs/Deepfake-and-real-images",
    "JamieWithofs/Deepfake-and-real-images-2",
]

DEFAULT_DISCOVERY_QUERIES = [
    "cifake",
    "deepfake image",
    "ai generated image real",
    "synthetic image detection",
    "real vs fake image",
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
LOW_QUALITY_NAME_RE = re.compile(r"(^|[^a-z0-9])(toy|dummy|sample|mini|tiny|test)([^a-z0-9]|$)")


def normalize_label(v) -> Optional[str]:
    if isinstance(v, bool):
        return "ai" if v else "real"
    if isinstance(v, int):
        return "ai" if int(v) == 1 else "real"
    s = str(v).strip().lower()
    if s.isdigit():
        return "ai" if int(s) == 1 else "real"
    if any(k in s for k in ["ai", "fake", "generated", "synthetic", "deepfake"]):
        return "ai"
    if any(k in s for k in ["real", "human", "natural", "authentic"]):
        return "real"
    return None


def hash_img_bytes(img: Image.Image) -> str:
    b = img.convert("RGB").tobytes()
    return hashlib.sha256(b).hexdigest()


def dhash_hex(img: Image.Image) -> str:
    g = img.convert("L").resize((9, 8), Image.BILINEAR)
    px = list(g.getdata())
    bits: List[str] = []
    for y in range(8):
        row = px[y * 9 : (y + 1) * 9]
        for x in range(8):
            bits.append("1" if row[x] > row[x + 1] else "0")
    return f"{int(''.join(bits), 2):016x}"


def hamming_hex(a: str, b: str) -> int:
    return (int(a, 16) ^ int(b, 16)).bit_count()


def image_entropy_bits(img: Image.Image) -> float:
    hist = img.convert("L").histogram()
    total = float(sum(hist))
    if total <= 0:
        return 0.0
    ent = 0.0
    for n in hist:
        if n <= 0:
            continue
        p = float(n) / total
        ent -= p * math.log2(p)
    return ent


def passes_quality_filters(
    img: Image.Image,
    min_side: int,
    max_aspect_ratio: float,
    min_entropy: float,
) -> tuple[bool, str]:
    w, h = img.size
    if min(w, h) < int(min_side):
        return False, "too_small"
    aspect = float(max(w, h)) / float(max(1, min(w, h)))
    if aspect > float(max_aspect_ratio):
        return False, "bad_aspect"
    ent = image_entropy_bits(img)
    if ent < float(min_entropy):
        return False, "low_entropy"
    return True, "ok"


def open_example_image(ex, image_field: str) -> Optional[Image.Image]:
    img = ex.get(image_field)
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    try:
        return Image.fromarray(img).convert("RGB")
    except Exception:
        return None


def open_local_image(path: Path) -> Optional[Image.Image]:
    try:
        with Image.open(path) as img:
            return img.convert("RGB")
    except Exception:
        return None


def find_fields(ds_split) -> Tuple[str, str]:
    cols = ds_split.column_names
    image_field = "image" if "image" in cols else next((c for c in cols if "image" in c.lower() or c.lower() == "img"), None)
    label_field = "label" if "label" in cols else next((c for c in cols if c.lower() in {"class", "target", "labels"}), None)
    if image_field is None or label_field is None:
        raise RuntimeError(f"Unable to infer fields from columns: {cols}")
    return image_field, label_field


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


def save_img(img: Image.Image, path: Path, quality: int = 92):
    path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(path, quality=quality)


def read_sources_file(path: Path) -> List[str]:
    out: List[str] = []
    if not path.exists():
        print(f"warning_sources_file_missing path={path}")
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    return out


def unique_preserve(items: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for it in items:
        if it in seen:
            continue
        seen.add(it)
        out.append(it)
    return out


def source_tag(src: str) -> str:
    if src.startswith("local::"):
        base = src.replace("local::", "local_")
    else:
        base = src.split("/")[-1]
    tag = re.sub(r"[^a-zA-Z0-9]+", "_", base).strip("_").lower()
    return (tag or "src")[:30]


def infer_local_class(path: Path) -> Optional[str]:
    for part in reversed(path.parts):
        p = part.lower()
        if p in {"ai", "real"}:
            return p
    return None


def collect_local_paths(root: Path, seed: int) -> List[Tuple[Path, str]]:
    paths: List[Tuple[Path, str]] = []
    if not root.exists():
        print(f"warning_local_source_missing path={root}")
        return paths
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        cls = infer_local_class(p)
        if cls is None:
            continue
        paths.append((p, cls))
    rng = random.Random(seed)
    rng.shuffle(paths)
    return paths


def discover_hf_sources(
    queries: Sequence[str],
    per_query_limit: int,
    max_sources: int,
    min_downloads: int,
    min_likes: int,
    min_quality_score: float,
    print_top_n: int,
) -> List[str]:
    if HfApi is None:
        print("warning_hf_discovery_unavailable reason=huggingface_hub_missing")
        return []
    api = HfApi()
    found: List[Tuple[str, float, int, int]] = []
    for q in queries:
        try:
            matches = api.list_datasets(search=q, limit=per_query_limit, sort="downloads", direction=-1)
        except Exception as e:
            print(f"warning_hf_discovery_query_failed query={q!r} reason={e}")
            continue
        for ds in matches:
            ds_id = str(getattr(ds, "id", "") or "").strip()
            if not ds_id:
                continue
            low = ds_id.lower()
            tags = [str(t).lower() for t in (getattr(ds, "tags", None) or [])]
            looks_image = any("image" in t for t in tags) or any(k in low for k in ["image", "img", "cifake"])
            looks_detection = any(k in low for k in ["fake", "deepfake", "generated", "synthetic", "real"])
            if not (looks_image and looks_detection):
                continue
            downloads = int(getattr(ds, "downloads", 0) or 0)
            likes = int(getattr(ds, "likes", 0) or 0)
            if downloads < min_downloads or likes < min_likes:
                continue
            score = min(3.0, math.log10(max(1, downloads) + 1.0)) + min(2.0, math.log10(max(1, likes) + 1.0))
            if LOW_QUALITY_NAME_RE.search(ds_id.lower()):
                score -= 0.8
            if score < min_quality_score:
                continue
            found.append((ds_id, score, downloads, likes))
    found_sorted = sorted(found, key=lambda x: x[1], reverse=True)
    for ds_id, score, dl, lk in found_sorted[: max(0, int(print_top_n))]:
        print(f"hf_candidate id={ds_id} score={score:.3f} downloads={dl} likes={lk}")
    return unique_preserve([x[0] for x in found_sorted])[:max_sources]


def likely_rate_limited(msg: str) -> bool:
    low = msg.lower()
    return any(k in low for k in ["429", "too many requests", "rate limit", "ratelimit", "5 min"])


def done(have: Dict[str, Dict[str, int]], need: Dict[str, Dict[str, int]]) -> bool:
    return all(have[s][c] >= need[s][c] for s in ["train", "val", "test"] for c in ["ai", "real"])


def next_split_for_class(
    have: Dict[str, Dict[str, int]],
    need: Dict[str, Dict[str, int]],
    cls: str,
    rng: random.Random,
) -> Optional[str]:
    remaining = {s: max(0, need[s][cls] - have[s][cls]) for s in ["train", "val", "test"]}
    choices = [s for s, rem in remaining.items() if rem > 0]
    if not choices:
        return None
    tot = float(sum(remaining[s] for s in choices))
    pick = rng.random() * tot
    acc = 0.0
    for s in choices:
        acc += float(remaining[s])
        if pick <= acc:
            return s
    return choices[-1]


def count_existing(out: Path) -> Dict[str, Dict[str, int]]:
    have = {s: {"ai": 0, "real": 0} for s in ["train", "val", "test"]}
    for split in ["train", "val", "test"]:
        for cls in ["ai", "real"]:
            d = out / split / cls
            have[split][cls] = len(list(d.glob("*.jpg"))) if d.exists() else 0
    return have


def build_source_list(args) -> List[str]:
    sources: List[str] = []
    if not args.no_default_sources and not args.hf_only:
        sources.extend(DEFAULT_SOURCES)
    if args.sources_file and not args.hf_only:
        sources.extend(read_sources_file(Path(args.sources_file)))
    if args.extra_source and not args.hf_only:
        sources.extend(args.extra_source)
    if args.discover_hf:
        discovered: List[str] = []
        cache_path = Path(args.hf_cache_file) if args.hf_cache_file else None
        if cache_path and cache_path.exists():
            discovered = read_sources_file(cache_path)
            print(f"loaded_hf_discovery_cache={cache_path} count={len(discovered)}")
            if args.hf_cache_only_if_present:
                print("hf_discovery_mode=cache_only_if_present")
                print(f"discovered_hf_sources={len(discovered)}")
                sources.extend(discovered)
                return unique_preserve(sources)
        if not discovered:
            discovered = discover_hf_sources(
                queries=args.hf_query or DEFAULT_DISCOVERY_QUERIES,
                per_query_limit=args.hf_discovery_limit,
                max_sources=args.hf_max_sources,
                min_downloads=args.hf_min_downloads,
                min_likes=args.hf_min_likes,
                min_quality_score=args.hf_min_quality_score,
                print_top_n=args.hf_print_top,
            )
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text("\n".join(discovered) + ("\n" if discovered else ""), encoding="utf-8")
                print(f"saved_hf_discovery_cache={cache_path} count={len(discovered)}")
        print(f"discovered_hf_sources={len(discovered)}")
        sources.extend(discovered)
    return unique_preserve(sources)


def main():
    ap = argparse.ArgumentParser(description="Build large, high-quality AI-vs-real image dataset from HF + optional local data")
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
    ap.add_argument("--jpeg-quality", type=int, default=92)
    ap.add_argument("--hardneg-fraction", type=float, default=0.6)
    ap.add_argument("--sources-file", default="")
    ap.add_argument("--extra-source", action="append", default=[])
    ap.add_argument("--local-source", action="append", default=[])
    ap.add_argument("--hf-only", action="store_true", default=False, help="Only use Hugging Face sources (disable local/default additions)")
    ap.add_argument("--no-default-sources", action="store_true", default=False, help="Disable built-in static source list")
    ap.add_argument("--discover-hf", action="store_true", default=False)
    ap.add_argument("--no-discover-hf", dest="discover_hf", action="store_false")
    ap.add_argument("--hf-query", action="append", default=[])
    ap.add_argument("--hf-discovery-limit", type=int, default=120, help="Per-query max datasets to scan in discovery")
    ap.add_argument("--hf-max-sources", type=int, default=260, help="Global cap on discovered dataset ids")
    ap.add_argument("--hf-min-downloads", type=int, default=80)
    ap.add_argument("--hf-min-likes", type=int, default=2)
    ap.add_argument("--hf-min-quality-score", type=float, default=1.7)
    ap.add_argument("--hf-print-top", type=int, default=15)
    ap.add_argument("--hf-cache-file", default="", help="Optional file path to cache discovered HF source ids")
    ap.add_argument("--hf-cache-only-if-present", action="store_true", default=True, help="If cache file exists, use it and skip live HF discovery calls")
    ap.add_argument("--no-hf-cache-only-if-present", dest="hf_cache_only_if_present", action="store_false")
    ap.add_argument("--streaming", action="store_true", default=True, help="Use HF streaming mode to reduce metadata overhead")
    ap.add_argument("--no-streaming", dest="streaming", action="store_false")
    ap.add_argument("--cache-dir", default="", help="HF datasets cache directory (improves resume and avoids repeated downloads)")
    ap.add_argument("--stream-buffer-size", type=int, default=12000, help="Shuffle buffer for streaming datasets")
    ap.add_argument("--max-samples-per-source", type=int, default=60000, help="Max examples to inspect per source before moving on")
    ap.add_argument("--acceptance-warmup-samples", type=int, default=400)
    ap.add_argument("--min-acceptance-rate", type=float, default=0.01)
    ap.add_argument("--repo-base-pause-ms", type=int, default=900, help="Base pause between HF repositories")
    ap.add_argument("--repo-jitter-ms", type=int, default=900, help="Extra random pause between HF repositories")
    ap.add_argument("--repo-cooldown-ms", type=int, default=45000, help="Cooldown after rate-limit or repeated source failures")
    ap.add_argument("--max-consecutive-failures", type=int, default=2, help="Cooldown trigger for consecutive source failures")
    ap.add_argument("--token-env", default="HF_TOKEN")
    ap.add_argument("--discover-only", action="store_true", default=False, help="Only run HF discovery/cache update and exit")
    ap.add_argument("--require-full-targets", action="store_true", default=False, help="Exit non-zero if final dataset is below requested class/split targets")
    args = ap.parse_args()

    random.seed(args.seed)
    rng = random.Random(args.seed + 17)

    token = os.environ.get(args.token_env)
    if token:
        print(f"using_token_env={args.token_env}")
    else:
        print(f"warning_no_token env={args.token_env} (public datasets still work, but with lower limits)")
    print(f"hf_quality_filters min_downloads={args.hf_min_downloads} min_likes={args.hf_min_likes} min_score={args.hf_min_quality_score} min_acceptance_rate={args.min_acceptance_rate}")

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
    print(
        "existing_counts "
        + " ".join([f"{s}/{c}={counts[s][c]}" for s in ["train", "val", "test"] for c in ["ai", "real"]])
    )

    # Global dedupe to prevent leakage across splits.
    seen_exact = set()
    seen_dhash_by_cls: Dict[str, List[str]] = defaultdict(list)

    global_rejects: DefaultDict[str, int] = defaultdict(int)
    source_reports: List[Dict[str, object]] = []

    def try_accept_and_save(img: Image.Image, cls: str, src: str, source_counts: Dict[str, int]) -> bool:
        if done(counts, targets):
            return False
        if source_counts[cls] >= args.max_per_source_class:
            global_rejects["source_class_cap"] += 1
            return False

        ok, reason = passes_quality_filters(
            img=img,
            min_side=args.min_side,
            max_aspect_ratio=args.max_aspect_ratio,
            min_entropy=args.min_entropy,
        )
        if not ok:
            global_rejects[reason] += 1
            return False

        h = hash_img_bytes(img)
        if h in seen_exact:
            global_rejects["dup_exact"] += 1
            return False

        d = dhash_hex(img)
        near_dup = False
        prevs = seen_dhash_by_cls[cls]
        if args.near_window > 0:
            start = max(0, len(prevs) - args.near_window)
            for prev in prevs[start:]:
                if hamming_hex(d, prev) <= args.near_hamming:
                    near_dup = True
                    break
        if near_dup:
            global_rejects["dup_near"] += 1
            return False

        split = next_split_for_class(counts, targets, cls, rng)
        if split is None:
            global_rejects["no_split_needed"] += 1
            return False

        seen_exact.add(h)
        seen_dhash_by_cls[cls].append(d)

        n = counts[split][cls]
        src_tag = source_tag(src)
        dst = out / split / cls / f"source={src_tag}__{split}_{cls}_{n:07d}.jpg"
        save_img(img, dst, quality=args.jpeg_quality)
        counts[split][cls] += 1
        source_counts[cls] += 1
        return True

    hf_sources = build_source_list(args)
    if args.hf_only and args.local_source:
        print("warning_hf_only_ignores_local_sources=1")
    if not hf_sources:
        raise SystemExit("no_hf_sources_resolved: enable --discover-hf or provide HF sources cache/file")
    print(f"hf_source_candidates={len(hf_sources)}")

    consecutive_source_failures = 0
    for src_idx, src in enumerate(hf_sources, start=1):
        if done(counts, targets):
            break
        repo_pause = args.repo_base_pause_ms + random.randint(0, max(args.repo_jitter_ms, 0))
        if repo_pause > 0:
            time.sleep(repo_pause / 1000.0)
        source_counts = {"ai": 0, "real": 0}
        source_rejects: DefaultDict[str, int] = defaultdict(int)
        try:
            try:
                ds = load_dataset(src, token=token, streaming=args.streaming, cache_dir=(args.cache_dir or None))
            except TypeError:
                ds = load_dataset(src, streaming=args.streaming, cache_dir=(args.cache_dir or None))
        except Exception as e:
            print(f"skip_source={src} reason={e}")
            consecutive_source_failures += 1
            if likely_rate_limited(str(e)) or consecutive_source_failures >= args.max_consecutive_failures:
                cooldown = args.repo_cooldown_ms
                print(f"cooldown_ms={cooldown} reason=source_failure")
                time.sleep(cooldown / 1000.0)
                consecutive_source_failures = 0
            continue
        consecutive_source_failures = 0

        split_name = "train" if "train" in ds else ("validation" if "validation" in ds else list(ds.keys())[0])
        split = ds[split_name]
        try:
            image_field, label_field = find_fields(split)
        except Exception as e:
            print(f"skip_source={src} reason={e}")
            continue

        accepted_total = 0
        processed_total = 0
        if args.streaming:
            try:
                shuffled = split.shuffle(seed=args.seed + src_idx * 137, buffer_size=max(500, int(args.stream_buffer_size)))
            except Exception:
                shuffled = split
            stream_iter = shuffled.take(max(1, int(args.max_samples_per_source)))
        else:
            idxs = list(range(len(split)))
            random.Random(args.seed + src_idx * 137).shuffle(idxs)
            stream_iter = (split[i] for i in idxs[: max(1, int(args.max_samples_per_source))])

        for ex in stream_iter:
            if done(counts, targets):
                break
            processed_total += 1
            if processed_total >= int(args.acceptance_warmup_samples):
                acceptance_rate = accepted_total / float(max(1, processed_total))
                if acceptance_rate < float(args.min_acceptance_rate):
                    source_rejects["low_acceptance_rate"] += 1
                    print(f"early_stop_source={src} reason=low_acceptance_rate accepted={accepted_total} processed={processed_total} rate={acceptance_rate:.5f}")
                    break
            if accepted_total >= args.max_unique_per_source:
                source_rejects["source_total_cap"] += 1
                break
            cls = normalize_label(ex[label_field])
            if cls not in {"ai", "real"}:
                source_rejects["unknown_label"] += 1
                continue
            img = open_example_image(ex, image_field)
            if img is None:
                source_rejects["decode_fail"] += 1
                continue
            before = dict(global_rejects)
            if try_accept_and_save(img, cls, src, source_counts):
                accepted_total += 1
            else:
                after = global_rejects
                changed = [k for k, v in after.items() if v != before.get(k, 0)]
                if changed:
                    source_rejects[changed[0]] += 1
                else:
                    source_rejects["rejected_other"] += 1

        report = {
            "source": src,
            "type": "hf",
            "accepted_ai": source_counts["ai"],
            "accepted_real": source_counts["real"],
            "rejections": dict(source_rejects),
        }
        source_reports.append(report)
        print(
            f"loaded_source={src} accepted_ai={source_counts['ai']} accepted_real={source_counts['real']} "
            f"processed={processed_total} rejected={sum(source_rejects.values())} acceptance_rate={(accepted_total / float(max(1, processed_total))):.5f}"
        )

    for local_root in ([] if args.hf_only else args.local_source):
        if done(counts, targets):
            break
        root = Path(local_root)
        src_name = f"local::{root.resolve()}"
        source_counts = {"ai": 0, "real": 0}
        source_rejects: DefaultDict[str, int] = defaultdict(int)
        local_paths = collect_local_paths(root, seed=args.seed + 333)
        accepted_total = 0
        for p, cls in local_paths:
            if done(counts, targets):
                break
            if accepted_total >= args.max_unique_per_source:
                source_rejects["source_total_cap"] += 1
                break
            img = open_local_image(p)
            if img is None:
                source_rejects["decode_fail"] += 1
                continue
            before = dict(global_rejects)
            if try_accept_and_save(img, cls, src_name, source_counts):
                accepted_total += 1
            else:
                after = global_rejects
                changed = [k for k, v in after.items() if v != before.get(k, 0)]
                if changed:
                    source_rejects[changed[0]] += 1
                else:
                    source_rejects["rejected_other"] += 1

        report = {
            "source": str(root),
            "type": "local",
            "accepted_ai": source_counts["ai"],
            "accepted_real": source_counts["real"],
            "rejections": dict(source_rejects),
        }
        source_reports.append(report)
        print(
            f"loaded_local_source={root} accepted_ai={source_counts['ai']} accepted_real={source_counts['real']} "
            f"rejected={sum(source_rejects.values())}"
        )

    # Add hard negatives to train split only.
    hard_modes = ["jpeg35", "blur", "resize60", "sharpen", "screenshot"]
    for cls in ["ai", "real"]:
        base_files = [p for p in (out / "train" / cls).glob("*.jpg") if "hardneg=" not in p.name]
        random.Random(args.seed + (1 if cls == "ai" else 2)).shuffle(base_files)
        hard_target = max(1, int(targets["train"][cls] * max(args.hardneg_fraction, 0.0)))
        hn_count = 0
        for p in base_files[: min(hard_target, len(base_files))]:
            try:
                with Image.open(p) as pil:
                    img = pil.convert("RGB")
            except Exception:
                continue
            mode = random.choice(hard_modes)
            aug = augment_hard_negative(img, mode)
            dst = out / "train" / cls / f"hardneg={mode}__{p.stem}__hn{hn_count:07d}.jpg"
            save_img(aug, dst, quality=max(70, min(95, args.jpeg_quality - 2)))
            hn_count += 1
        print(f"hard_negatives_{cls}={hn_count}")

    for split in ["train", "val", "test"]:
        for cls in ["ai", "real"]:
            n = len(list((out / split / cls).glob("*.jpg")))
            print(f"{split}/{cls}={n}")
            if n < targets[split][cls]:
                print(f"warning_shortfall split={split} cls={cls} have={n} need={targets[split][cls]}")

    summary = {
        "targets": targets,
        "final_counts": {
            split: {cls: len(list((out / split / cls).glob("*.jpg"))) for cls in ["ai", "real"]}
            for split in ["train", "val", "test"]
        },
        "global_rejections": dict(global_rejects),
        "source_reports": source_reports,
    }
    (out / "dataset_build_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("dataset_ready", out)
    print(f"report={out / 'dataset_build_report.json'}")
    if args.require_full_targets:
        shortfalls = []
        for split in ["train", "val", "test"]:
            for cls in ["ai", "real"]:
                have_n = summary["final_counts"][split][cls]
                need_n = targets[split][cls]
                if have_n < need_n:
                    shortfalls.append(f"{split}/{cls}:{have_n}<{need_n}")
        if shortfalls:
            raise SystemExit("dataset_incomplete: " + ",".join(shortfalls))


if __name__ == "__main__":
    main()
