from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Iterable, Sequence

from hf_data import normalize_hf_token, read_noncomment_lines, unique_preserve, write_noncomment_lines

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

LOW_QUALITY_NAME_RE = re.compile(r"(^|[^a-z0-9])(toy|dummy|sample|mini|tiny|test)([^a-z0-9]|$)")
HF_DATASET_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


def cache_policy_path(cache_path: Path) -> Path:
    return cache_path.with_name(cache_path.name + ".policy.json")


def discovery_policy(args) -> dict[str, object]:
    return {
        "queries": list(args.hf_query or DEFAULT_DISCOVERY_QUERIES),
        "hf_discovery_limit": int(args.hf_discovery_limit),
        "hf_max_sources": int(args.hf_max_sources),
        "hf_min_downloads": int(args.hf_min_downloads),
        "hf_min_likes": int(args.hf_min_likes),
        "hf_min_quality_score": float(args.hf_min_quality_score),
        "hf_print_top": int(args.hf_print_top),
        "hf_query_pause_ms": int(args.hf_query_pause_ms),
    }


def load_cache_policy(cache_path: Path) -> dict[str, object] | None:
    policy_path = cache_policy_path(cache_path)
    if not policy_path.exists():
        return None
    try:
        return json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def save_cache_policy(cache_path: Path, policy: dict[str, object]) -> None:
    policy_path = cache_policy_path(cache_path)
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_text(json.dumps(policy, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def read_sources_file(path: Path) -> list[str]:
    if not path.exists():
        print(f"warning_sources_file_missing path={path}")
        return []
    return read_noncomment_lines(path)


def is_probable_hf_dataset_id(src: str) -> bool:
    return bool(HF_DATASET_ID_RE.match(src.strip()))


def discover_hf_sources(
    queries: Sequence[str],
    per_query_limit: int,
    max_sources: int,
    min_downloads: int,
    min_likes: int,
    min_quality_score: float,
    print_top_n: int,
    query_pause_ms: int = 0,
    token: str | None = None,
) -> list[str]:
    if HfApi is None:
        print("warning_hf_discovery_unavailable reason=huggingface_hub_missing")
        return []
    token = normalize_hf_token(token)
    api = HfApi(token=token)
    found: list[tuple[str, float, int, int]] = []
    for idx, q in enumerate(queries, start=1):
        if idx > 1 and int(query_pause_ms) > 0:
            time.sleep(int(query_pause_ms) / 1000.0)
        try:
            matches = api.list_datasets(search=q, limit=per_query_limit, sort="downloads")
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
        except Exception as e:
            print(f"warning_hf_discovery_query_failed query={q!r} reason={e}")
            continue
    found_sorted = sorted(found, key=lambda x: x[1], reverse=True)
    for ds_id, score, dl, lk in found_sorted[: max(0, int(print_top_n))]:
        print(f"hf_candidate id={ds_id} score={score:.3f} downloads={dl} likes={lk}")
    return unique_preserve([x[0] for x in found_sorted])[:max_sources]


def build_source_list(args) -> list[str]:
    def finalize_sources(raw_sources: Iterable[str]) -> list[str]:
        resolved = unique_preserve(raw_sources)
        if not args.hf_only:
            return resolved
        before = len(resolved)
        resolved = [s for s in resolved if not str(s).startswith("local::")]
        filtered = before - len(resolved)
        if filtered > 0:
            print(f"hf_only_filtered_non_hf_sources={filtered}")
        before_valid = len(resolved)
        resolved = [s for s in resolved if is_probable_hf_dataset_id(str(s))]
        invalid = before_valid - len(resolved)
        if invalid > 0:
            print(f"hf_only_filtered_invalid_dataset_ids={invalid}")
        return resolved

    sources: list[str] = []
    if not args.no_default_sources:
        sources.extend(DEFAULT_SOURCES)
    if args.sources_file:
        sources.extend(read_sources_file(Path(args.sources_file)))
    if args.extra_source:
        sources.extend(args.extra_source)
    if args.discover_hf:
        discovered: list[str] = []
        current_policy = discovery_policy(args)
        cache_path = Path(args.hf_cache_file) if args.hf_cache_file else None
        if cache_path and cache_path.exists():
            discovered = read_sources_file(cache_path)
            print(f"loaded_hf_discovery_cache={cache_path} count={len(discovered)}")
            cached_policy = load_cache_policy(cache_path)
            if cached_policy != current_policy:
                print("hf_discovery_cache_policy_mismatch=1 fallback=live_discovery")
                discovered = []
            if discovered and args.hf_cache_only_if_present:
                print("hf_discovery_mode=cache_only_if_present")
                print(f"discovered_hf_sources={len(discovered)}")
                sources.extend(discovered)
                return finalize_sources(sources)
            if args.hf_cache_only_if_present and not discovered:
                print("hf_discovery_cache_empty=1 fallback=live_discovery")
        if not discovered:
            discovered = discover_hf_sources(
                queries=args.hf_query or DEFAULT_DISCOVERY_QUERIES,
                per_query_limit=args.hf_discovery_limit,
                max_sources=args.hf_max_sources,
                min_downloads=args.hf_min_downloads,
                min_likes=args.hf_min_likes,
                min_quality_score=args.hf_min_quality_score,
                print_top_n=args.hf_print_top,
                query_pause_ms=args.hf_query_pause_ms,
                token=normalize_hf_token(os.environ.get(args.token_env)),
            )
            if cache_path:
                write_noncomment_lines(cache_path, discovered)
                save_cache_policy(cache_path, current_policy)
                print(f"saved_hf_discovery_cache={cache_path} count={len(discovered)}")
        print(f"discovered_hf_sources={len(discovered)}")
        sources.extend(discovered)
    return finalize_sources(sources)
