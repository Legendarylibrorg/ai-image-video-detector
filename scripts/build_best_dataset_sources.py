from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

from hf_data import (
    normalize_hf_token,
    read_noncomment_lines,
    resolve_hf_token_value,
    unique_preserve,
    validate_hf_dataset_source_id,
    validate_hf_discovery_query,
    write_noncomment_lines,
)

try:
    from huggingface_hub import HfApi
except ImportError:  # pragma: no cover - optional dependency path
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
    "camera photo dataset",
    "smartphone photo dataset",
    "dslr photo dataset",
    "webcam image dataset",
    "cctv frame image dataset",
    "portrait selfie real fake",
    "group photo real fake",
    "indoor room photo dataset",
    "outdoor landscape photo dataset",
    "product photo dataset",
    "food photo dataset",
    "animal photo dataset",
    "night photo dataset",
    "macro close up photo dataset",
    "panorama photo dataset",
    "high resolution photo dataset",
    "low resolution image dataset",
    "screenshot dataset image",
    "chat ui screenshot",
    "browser screenshot image",
    "mobile app screenshot image",
    "document scan image dataset",
    "receipt scanned document image",
    "invoice form document scan",
    "id card document image",
    "poster infographic image",
    "logo icon brand image",
    "architecture photo dataset",
    "street photo dataset",
    "travel photo dataset",
    "fashion photo dataset",
    "sports action photo dataset",
    "vehicle road photo dataset",
    "drone aerial photo dataset",
    "satellite image dataset",
    "microscopy image dataset",
    "medical photo dataset",
    "old photo scan dataset",
    "film scan photo dataset",
    "raw photo dataset",
    "low light photo dataset",
    "comic panel image dataset",
    "meme screenshot dataset",
    "infographic dataset image",
    "desktop screenshot dataset",
    "tablet screenshot image",
    "anime illustration real fake",
    "digital art illustration dataset",
    "3d render real fake",
    "social media image dataset",
    "watermarked social media image",
    "recompressed image dataset",
    "heavily edited real photo",
    "jpeg photo dataset",
    "png image dataset",
    "webp image dataset",
    "bmp image dataset",
    "tiff image dataset",
    "raw camera photo dataset",
    "surveillance frame image dataset",
    "driving dashcam image dataset",
    "ecommerce product image dataset",
    "museum artwork image dataset",
    "historical archive photo dataset",
    "newspaper photo archive dataset",
    "scientific figure image dataset",
    "diagram image dataset",
    "medical scan image dataset",
    "microscopy cell image dataset",
    "satellite remote sensing image dataset",
    "aerial photography dataset",
    "selfie photo dataset",
    "crowd event photo dataset",
    "wildlife camera trap image dataset",
    "fashion catalog image dataset",
    "food delivery photo dataset",
    "manga artwork dataset",
    "cgi render image dataset",
    "game screenshot dataset",
    "mobile camera photo dataset",
    "portrait photography dataset",
    "street photography dataset",
]

LOW_QUALITY_NAME_RE = re.compile(r"(^|[^a-z0-9])(toy|dummy|sample|mini|tiny|test)([^a-z0-9]|$)")
USEFUL_KEYWORD_GROUPS: dict[str, tuple[str, ...]] = {
    "detector_labels": ("real", "fake", "deepfake", "generated", "synthetic", "cifake"),
    "photo": ("photo", "camera", "smartphone", "dslr", "webcam", "cctv", "selfie", "portrait"),
    "screen": ("screenshot", "screen", "browser", "chat ui", "dashboard", "interface", "ui"),
    "document": ("document", "receipt", "invoice", "id card", "scan", "scanned", "form"),
    "illustration": ("anime", "illustration", "digital art", "artwork", "manga"),
    "render": ("3d", "render", "cgi", "game"),
    "web": ("social media", "meme", "watermarked", "recompressed", "edited"),
    "format": ("jpeg", "jpg", "png", "webp", "bmp", "tiff"),
    "resolution": ("low resolution", "high resolution", "panorama", "macro", "widescreen", "thumbnail"),
    "scene": ("landscape", "indoor", "outdoor", "night", "product", "food", "animal"),
}

DEFAULT_ALLOWED_LICENSE_TAGS = (
    "apache-2.0",
    "mit",
    "bsd-2-clause",
    "bsd-3-clause",
    "cc0-1.0",
    "cc-by-4.0",
    "cc-by-3.0",
    "cc-by-sa-4.0",
    "cc-by-sa-3.0",
    "pddl",
    "odc-by",
    "odbl",
    "cdla-permissive-2.0",
    "etalab-2.0",
)

DEFAULT_DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "photo": ("photo", "camera", "dslr", "smartphone", "portrait", "selfie", "landscape", "wildlife"),
    "screen": ("screenshot", "screen", "browser", "chat ui", "desktop", "mobile app", "ui", "dashboard"),
    "document": ("document", "invoice", "receipt", "scan", "scanned", "id card", "form", "ocr"),
    "illustration": ("illustration", "anime", "manga", "artwork", "drawing", "digital art"),
    "render": ("render", "cgi", "3d", "synthetic", "game screenshot"),
    "web": ("social media", "meme", "watermarked", "recompressed", "edited", "web"),
}


@dataclass(frozen=True)
class AuditedSource:
    source_id: str
    score: float
    downloads: int
    likes: int
    license_markers: tuple[str, ...]
    matched_groups: tuple[str, ...]
    domain_tags: tuple[str, ...]
    image_field: str
    label_field: str
    label_map: dict[str, str]
    split_names: tuple[str, ...]
    total_rows: int | None
    approved: bool
    rejection_reasons: tuple[str, ...]
    config_name: str


def _flatten_license_values(value: object) -> Iterable[str]:
    if isinstance(value, str):
        yield value
        return
    if isinstance(value, dict):
        for key, nested in value.items():
            key_low = str(key).lower()
            if key_low in {"license", "licenses", "license_name"}:
                yield from _flatten_license_values(nested)
        return
    if isinstance(value, (list, tuple, set)):
        for nested in value:
            yield from _flatten_license_values(nested)


def normalize_license_marker(raw: object) -> str:
    value = str(raw or "").strip().lower()
    if value.startswith("license:"):
        value = value.split(":", 1)[1].strip()
    return value


def extract_license_markers(ds: object) -> set[str]:
    markers: set[str] = set()
    for tag in getattr(ds, "tags", None) or []:
        normalized = normalize_license_marker(tag)
        if normalized:
            markers.add(normalized)
    for field_name in ("cardData", "card_data", "carddata"):
        card_data = getattr(ds, field_name, None)
        if card_data is None:
            continue
        for value in _flatten_license_values(card_data):
            normalized = normalize_license_marker(value)
            if normalized:
                markers.add(normalized)
    for field_name in ("license", "licenses", "license_name"):
        value = getattr(ds, field_name, None)
        if value is None:
            continue
        for item in _flatten_license_values(value):
            normalized = normalize_license_marker(item)
            if normalized:
                markers.add(normalized)
    return markers


def matched_useful_keyword_groups(text: str) -> set[str]:
    lowered = text.lower()
    return {
        name
        for name, keywords in USEFUL_KEYWORD_GROUPS.items()
        if any(keyword in lowered for keyword in keywords)
    }


def infer_domain_tags(text: str) -> set[str]:
    lowered = text.lower()
    return {
        name
        for name, keywords in DEFAULT_DOMAIN_KEYWORDS.items()
        if any(keyword in lowered for keyword in keywords)
    }


def infer_image_and_label_fields(columns: Sequence[str]) -> tuple[str, str]:
    image_field = "image" if "image" in columns else next((c for c in columns if "image" in c.lower() or c.lower() == "img"), "")
    label_field = "label" if "label" in columns else next((c for c in columns if c.lower() in {"class", "target", "labels"}), "")
    return image_field, label_field


def normalize_label_text(value: object) -> str:
    text = str(value or "").strip().lower()
    if any(token in text for token in ("ai", "fake", "generated", "synthetic", "deepfake")):
        return "ai"
    if any(token in text for token in ("real", "human", "natural", "authentic", "photo")):
        return "real"
    return ""


def infer_label_map(feature: object) -> dict[str, str]:
    names = getattr(feature, "names", None)
    if names is None and isinstance(feature, dict):
        names = feature.get("names")
    if not isinstance(names, (list, tuple)):
        return {}
    out: dict[str, str] = {}
    for idx, name in enumerate(names):
        normalized = normalize_label_text(name)
        if normalized:
            out[str(idx)] = normalized
    return out


def _dataset_info_value(dataset_info: object, key: str, default: object = None) -> object:
    if isinstance(dataset_info, dict):
        return dataset_info.get(key, default)
    return getattr(dataset_info, key, default)


def _extract_dataset_info_payload(info: object) -> object:
    direct_candidates = [
        getattr(info, "dataset_info", None),
        getattr(info, "datasetInfo", None),
        getattr(info, "features", None),
    ]
    card_data = getattr(info, "cardData", None) or getattr(info, "card_data", None)
    if isinstance(card_data, dict):
        direct_candidates.extend([
            card_data.get("dataset_info"),
            card_data.get("datasetInfo"),
        ])
    for candidate in direct_candidates:
        if candidate not in (None, "", [], {}):
            return candidate
    return {}


def _iter_feature_entries(features: object) -> list[tuple[str, object]]:
    entries: list[tuple[str, object]] = []
    if features is None:
        return entries

    if isinstance(features, dict):
        if "name" in features and any(key in features for key in ("type", "dtype", "_type", "feature")):
            name = str(features.get("name", "")).strip()
            if name:
                return [(name, features.get("type", features.get("feature", features)))]
        for name, feature in features.items():
            clean_name = str(name).strip()
            if not clean_name:
                continue
            entries.append((clean_name, feature))
        return entries

    if hasattr(features, "items"):
        try:
            for name, feature in features.items():
                clean_name = str(name).strip()
                if not clean_name:
                    continue
                entries.append((clean_name, feature))
            if entries:
                return entries
        except (AttributeError, RuntimeError, TypeError, ValueError):
            # Some feature containers expose `.items()` but may fail at runtime.
            # Ignore this probing failure and continue with other parsing paths.
            pass

    if isinstance(features, (list, tuple)):
        for feature in features:
            if isinstance(feature, dict):
                name = str(feature.get("name", "")).strip()
                if name:
                    entries.append((name, feature.get("type", feature.get("feature", feature))))
                    continue
                if len(feature) == 1:
                    only_name, only_feature = next(iter(feature.items()))
                    clean_name = str(only_name).strip()
                    if clean_name:
                        entries.append((clean_name, only_feature))
                continue
            name = str(getattr(feature, "name", "")).strip()
            if name:
                entries.append((name, getattr(feature, "type", feature)))
        return entries

    return entries


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
        "hf_discovery_workers": int(getattr(args, "hf_discovery_workers", 1)),
        "hf_query_pause_ms": int(args.hf_query_pause_ms),
        "hf_require_open_license": bool(getattr(args, "hf_require_open_license", True)),
        "hf_license_allow": list(getattr(args, "hf_license_allow", []) or list(DEFAULT_ALLOWED_LICENSE_TAGS)),
    }


def load_cache_policy(cache_path: Path) -> dict[str, object] | None:
    policy_path = cache_policy_path(cache_path)
    if not policy_path.exists():
        return None
    src = Path(__file__).resolve().parent.parent / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    try:
        from ai_image_detector.io_limits import read_json_file_limited

        data = read_json_file_limited(policy_path)
        return data if data else None
    except (OSError, ValueError, UnicodeDecodeError):
        return None


def save_cache_policy(cache_path: Path, policy: dict[str, object]) -> None:
    policy_path = cache_policy_path(cache_path)
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_text(json.dumps(policy, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def read_sources_file(path: Path) -> list[str]:
    if not path.exists():
        print(f"warning_sources_file_missing path={path}")
        return []
    lines = read_noncomment_lines(path)
    out: list[str] = []
    for idx, line in enumerate(lines, start=1):
        try:
            out.append(validate_hf_dataset_source_id(line))
        except ValueError as exc:
            raise ValueError(f"sources_file_invalid_line path={path} line={idx} value={line!r}") from exc
    return out


def write_audit_manifest(path: Path, entries: Sequence[AuditedSource]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            {
                "source_id": entry.source_id,
                "score": round(float(entry.score), 6),
                "downloads": int(entry.downloads),
                "likes": int(entry.likes),
                "license_markers": list(entry.license_markers),
                "matched_groups": list(entry.matched_groups),
                "domain_tags": list(entry.domain_tags),
                "image_field": entry.image_field,
                "label_field": entry.label_field,
                "label_map": dict(entry.label_map),
                "split_names": list(entry.split_names),
                "total_rows": entry.total_rows,
                "config_name": entry.config_name,
                "approved": bool(entry.approved),
                "rejection_reasons": list(entry.rejection_reasons),
            },
            sort_keys=True,
        )
        for entry in entries
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _dataset_split_rows(split: object) -> int | None:
    num_rows = getattr(split, "num_rows", None)
    if num_rows is None:
        return None
    try:
        return int(num_rows)
    except (TypeError, ValueError):
        return None


def _fetch_dataset_info(api, source_id: str):
    info_fn = getattr(api, "dataset_info", None)
    if callable(info_fn):
        return info_fn(source_id)
    return None


def audit_hf_sources(
    source_ids: Sequence[str],
    *,
    token: str | None = None,
    min_rows: int = 100,
    require_image_field: bool = True,
    require_label_field: bool = True,
) -> list[AuditedSource]:
    if HfApi is None:
        print("warning_hf_audit_unavailable reason=huggingface_hub_missing")
        return []
    token_value = normalize_hf_token(token)
    api = HfApi(token=token_value)
    audited: list[AuditedSource] = []
    for source_id in unique_preserve(source_ids):
        rejection_reasons: list[str] = []
        score = 0.0
        downloads = 0
        likes = 0
        license_markers: set[str] = set()
        matched_groups: set[str] = set()
        domain_tags: set[str] = set()
        image_field = ""
        label_field = ""
        label_map: dict[str, str] = {}
        split_names: list[str] = []
        total_rows: int | None = None
        config_name = "default"
        try:
            info = _fetch_dataset_info(api, source_id)
            if info is None:
                rejection_reasons.append("dataset_info_unavailable")
            else:
                downloads = int(getattr(info, "downloads", 0) or 0)
                likes = int(getattr(info, "likes", 0) or 0)
                license_markers = extract_license_markers(info)
                text_fields = [
                    source_id,
                    str(getattr(info, "description", "") or ""),
                    " ".join(str(tag) for tag in (getattr(info, "tags", None) or [])),
                ]
                useful_text = " ".join(text_fields).lower()
                matched_groups = matched_useful_keyword_groups(useful_text)
                domain_tags = infer_domain_tags(useful_text)
                score = min(3.0, math.log10(max(1, downloads) + 1.0)) + min(2.0, math.log10(max(1, likes) + 1.0))
                score += min(1.5, 0.22 * float(len(matched_groups)))
                score += min(1.0, 0.35 * float(len(domain_tags)))
                siblings = getattr(info, "siblings", None) or []
                split_names = sorted(
                    {
                        str(getattr(split, "split", "")).strip()
                        for split in (getattr(info, "splits", None) or [])
                        if str(getattr(split, "split", "")).strip()
                    }
                )
                if not split_names:
                    split_names = sorted(
                        {
                            path.parts[0]
                            for sibling in siblings
                            for path in [Path(str(getattr(sibling, "rfilename", "") or ""))]
                            if len(path.parts) >= 2 and path.parts[0] in {"train", "test", "validation", "val"}
                        }
                    )
                if len(split_names) >= 2:
                    score += 0.35
                dataset_info = _extract_dataset_info_payload(info)
                config_name = str(
                    getattr(info, "config", "")
                    or _dataset_info_value(dataset_info, "config_name", "")
                    or "default"
                )
                features = _dataset_info_value(dataset_info, "features", dataset_info)
                feature_entries = _iter_feature_entries(features)
                columns = [name for name, _ in feature_entries]
                image_field, label_field = infer_image_and_label_fields(columns)
                if require_image_field and not image_field:
                    rejection_reasons.append("missing_image_field")
                if require_label_field and not label_field:
                    rejection_reasons.append("missing_label_field")
                if label_field:
                    for feature_name, feature_type in feature_entries:
                        if str(feature_name).strip() == label_field:
                            label_map = infer_label_map(feature_type)
                            break
                total_rows = sum(
                    rows
                    for rows in (
                        _dataset_split_rows(split)
                        for split in (getattr(info, "splits", None) or [])
                    )
                    if rows is not None
                ) or None
                if total_rows is not None and total_rows < int(min_rows):
                    rejection_reasons.append("too_few_rows")
                if not domain_tags:
                    rejection_reasons.append("missing_domain_tags")
        except Exception as exc:
            rejection_reasons.append(f"audit_failed:{exc}")
        audited.append(
            AuditedSource(
                source_id=source_id,
                score=score,
                downloads=downloads,
                likes=likes,
                license_markers=tuple(sorted(license_markers)),
                matched_groups=tuple(sorted(matched_groups)),
                domain_tags=tuple(sorted(domain_tags)),
                image_field=image_field,
                label_field=label_field,
                label_map=label_map,
                split_names=tuple(split_names),
                total_rows=total_rows,
                approved=not rejection_reasons,
                rejection_reasons=tuple(rejection_reasons),
                config_name=config_name,
            )
        )
    return audited


def is_probable_hf_dataset_id(src: str) -> bool:
    try:
        validate_hf_dataset_source_id(src)
        return True
    except ValueError:
        return False


def discover_hf_sources(
    queries: Sequence[str],
    per_query_limit: int,
    max_sources: int,
    min_downloads: int,
    min_likes: int,
    min_quality_score: float,
    print_top_n: int,
    query_workers: int = 1,
    query_pause_ms: int = 0,
    token: str | None = None,
    require_open_license: bool = True,
    allowed_license_tags: Sequence[str] = DEFAULT_ALLOWED_LICENSE_TAGS,
) -> list[str]:
    if HfApi is None:
        print("warning_hf_discovery_unavailable reason=huggingface_hub_missing")
        return []
    validated_queries: list[str] = []
    for raw in queries:
        s = str(raw).strip()
        if not s:
            continue
        try:
            validated_queries.append(validate_hf_discovery_query(s))
        except ValueError as exc:
            raise ValueError(f"invalid_hf_discovery_query value={raw!r}") from exc
    if not validated_queries:
        print("warning_hf_discovery_no_queries_after_validation")
        return []
    queries = validated_queries

    found: list[tuple[str, float, int, int]] = []
    allowed_licenses = {normalize_license_marker(tag) for tag in allowed_license_tags if normalize_license_marker(tag)}

    def collect_query_candidates(query: str) -> list[tuple[str, float, int, int]]:
        token_value = normalize_hf_token(token)
        try:
            api = HfApi(token=token_value)
            try:
                matches = api.list_datasets(search=query, limit=per_query_limit, sort="downloads", full=True)
            except TypeError:
                matches = api.list_datasets(search=query, limit=per_query_limit, sort="downloads")
            query_found: list[tuple[str, float, int, int]] = []
            for ds in matches:
                ds_id = str(getattr(ds, "id", "") or "").strip()
                if not ds_id:
                    continue
                low = ds_id.lower()
                tags = [str(t).lower() for t in (getattr(ds, "tags", None) or [])]
                useful_text = " ".join([low, *tags])
                matched_groups = matched_useful_keyword_groups(useful_text)
                looks_image = (
                    any("image" in t for t in tags)
                    or any(k in low for k in ["image", "img", "cifake"])
                    or any(tag in useful_text for tag in ("image-classification", "computer-vision"))
                )
                has_real_image_domain_signal = any(group != "detector_labels" for group in matched_groups)
                if not (looks_image or has_real_image_domain_signal):
                    continue
                downloads = int(getattr(ds, "downloads", 0) or 0)
                likes = int(getattr(ds, "likes", 0) or 0)
                if downloads < min_downloads or likes < min_likes:
                    continue
                license_markers = extract_license_markers(ds)
                if not license_markers or not tags:
                    try:
                        info = api.dataset_info(ds_id)
                    except Exception:
                        info = None
                    if info is not None:
                        if not tags:
                            tags = [str(t).lower() for t in (getattr(info, "tags", None) or [])]
                        if not license_markers:
                            license_markers = extract_license_markers(info)
                if require_open_license and not (license_markers & allowed_licenses):
                    continue
                score = min(3.0, math.log10(max(1, downloads) + 1.0)) + min(2.0, math.log10(max(1, likes) + 1.0))
                useful_text = " ".join([low, *tags, *sorted(license_markers)])
                matched_groups = matched_useful_keyword_groups(useful_text)
                useful_groups = len(matched_groups)
                score += min(1.5, 0.22 * float(useful_groups))
                if "cc0-1.0" in license_markers or "apache-2.0" in license_markers or "mit" in license_markers:
                    score += 0.1
                if any(tag in useful_text for tag in ("image-classification", "computer-vision", "image")):
                    score += 0.15
                if LOW_QUALITY_NAME_RE.search(ds_id.lower()):
                    score -= 0.8
                if score < min_quality_score:
                    continue
                query_found.append((ds_id, score, downloads, likes))
            return query_found
        except Exception as e:
            print(f"warning_hf_discovery_query_failed query={query!r} reason={e}")
            return []

    worker_count = max(1, int(query_workers))
    if worker_count <= 1 or len(queries) <= 1:
        for idx, q in enumerate(queries, start=1):
            if idx > 1 and int(query_pause_ms) > 0:
                time.sleep(int(query_pause_ms) / 1000.0)
            found.extend(collect_query_candidates(q))
    else:
        with ThreadPoolExecutor(max_workers=min(worker_count, len(queries))) as pool:
            futures = {pool.submit(collect_query_candidates, q): q for q in queries}
            for future in as_completed(futures):
                found.extend(future.result())
    found_sorted = sorted(found, key=lambda x: x[1], reverse=True)
    for ds_id, score, dl, lk in found_sorted[: max(0, int(print_top_n))]:
        print(f"hf_candidate id={ds_id} score={score:.3f} downloads={dl} likes={lk}")
    return unique_preserve([x[0] for x in found_sorted])[:max_sources]


def build_source_list(args) -> list[str]:
    def finalize_sources(raw_sources: Iterable[str]) -> list[str]:
        resolved = unique_preserve(raw_sources)
        before = len(resolved)
        resolved = [s for s in resolved if not str(s).startswith("local::")]
        filtered = before - len(resolved)
        if filtered > 0:
            print(f"filtered_non_hf_sources={filtered}")
        before_valid = len(resolved)
        resolved = [s for s in resolved if is_probable_hf_dataset_id(str(s))]
        invalid = before_valid - len(resolved)
        if invalid > 0:
            print(f"filtered_invalid_dataset_ids={invalid}")
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
        audit_entries: list[AuditedSource] = []
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
                query_workers=getattr(args, "hf_discovery_workers", 1),
                query_pause_ms=args.hf_query_pause_ms,
                token=resolve_hf_token_value(args.token_env)[0],
                require_open_license=bool(getattr(args, "hf_require_open_license", True)),
                allowed_license_tags=getattr(args, "hf_license_allow", []) or list(DEFAULT_ALLOWED_LICENSE_TAGS),
            )
            if cache_path:
                write_noncomment_lines(cache_path, discovered)
                save_cache_policy(cache_path, current_policy)
                print(f"saved_hf_discovery_cache={cache_path} count={len(discovered)}")
        audit_path = Path(args.hf_audit_file) if getattr(args, "hf_audit_file", "") else None
        if discovered and getattr(args, "hf_audit_sources", True):
            audit_entries = audit_hf_sources(
                discovered,
                token=resolve_hf_token_value(args.token_env)[0],
                min_rows=getattr(args, "hf_audit_min_rows", 100),
                require_image_field=bool(getattr(args, "hf_audit_require_image_field", True)),
                require_label_field=bool(getattr(args, "hf_audit_require_label_field", True)),
            )
            approved = [entry.source_id for entry in audit_entries if entry.approved]
            rejected = len(audit_entries) - len(approved)
            print(f"audited_hf_sources={len(audit_entries)} approved={len(approved)} rejected={rejected}")
            if audit_path is not None:
                write_audit_manifest(audit_path, audit_entries)
                print(f"saved_hf_audit_manifest={audit_path} count={len(audit_entries)}")
            if audit_entries and getattr(args, "hf_audit_filter_to_approved", True):
                discovered = approved if approved else []
        print(f"discovered_hf_sources={len(discovered)}")
        sources.extend(discovered)
    return finalize_sources(sources)
