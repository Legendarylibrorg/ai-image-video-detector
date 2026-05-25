from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, DefaultDict, Iterable, Iterator, Sequence

from PIL import Image

from build_best_dataset_sources import DEFAULT_ALLOWED_LICENSE_TAGS, build_source_list
from dataset_builder_common import HF_CACHE_DIR_DEFAULT
from hf_data import LoadedDatasetSource, load_hf_dataset_source, resolve_hf_token_value
from image_materialize import ImageDeduper, ImageQualityPolicy, open_example_image, passes_quality_filters, save_img
from script_support import ensure_src_path

ensure_src_path()

from ai_image_detector.collection_paths import validate_collection_io_paths
from ai_image_detector.dataset_layout import IMAGE_EXTS, count_split_class_files


SPLITS = ("train", "val", "test")
LABELISH_PARTS = {
    "annotation",
    "annotations",
    "category",
    "categories",
    "class",
    "classes",
    "label",
    "labels",
    "object",
    "objects",
    "tag",
    "tags",
    "target",
    "targets",
}
STOPWORDS = {
    "a",
    "an",
    "and",
    "for",
    "image",
    "images",
    "of",
    "the",
}


@dataclass(frozen=True)
class TargetSpec:
    target_name: str
    target_description: str
    positive_terms: tuple[str, ...]
    negative_terms: tuple[str, ...]
    exclude_terms: tuple[str, ...]
    ambiguous_terms: tuple[str, ...]
    positive_label_values: tuple[str, ...]
    negative_label_values: tuple[str, ...]
    required_context_terms: tuple[str, ...]
    text_fields: tuple[str, ...]
    treat_other_labeled_as_negative: bool


@dataclass(frozen=True)
class MatchResult:
    label: str | None
    reason: str
    fingerprint: str
    positive_hits: tuple[str, ...]
    negative_hits: tuple[str, ...]
    exclude_hits: tuple[str, ...]
    context_hits: tuple[str, ...]
    labelish_values: tuple[str, ...]


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(value or "").strip().lower()).strip("-")
    return slug or "target"


def _normalize_text(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_field_selector(value: object) -> str:
    return re.sub(r"\s+", "", str(value or "").strip().lower())


def _normalize_field_selectors(values: Iterable[object]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = _normalize_field_selector(value)
        if not item or item in seen:
            continue
        seen.add(item)
        normalized.append(item)
    return tuple(normalized)


def _normalize_terms(values: Iterable[object]) -> tuple[str, ...]:
    normalized = []
    seen: set[str] = set()
    for value in values:
        item = _normalize_text(value)
        if not item or item in seen:
            continue
        seen.add(item)
        normalized.append(item)
    return tuple(normalized)


def _default_positive_terms(target_name: str) -> tuple[str, ...]:
    normalized = _normalize_text(target_name)
    if not normalized:
        return ()
    terms = [normalized]
    parts = [part for part in normalized.split() if part not in STOPWORDS]
    if len(parts) > 1:
        terms.extend(part for part in parts if len(part) >= 3)
    return _normalize_terms(terms)


def build_default_target_spec(
    *,
    target_name: str,
    target_description: str = "",
    positive_terms: Sequence[str] = (),
    negative_terms: Sequence[str] = (),
    exclude_terms: Sequence[str] = (),
    ambiguous_terms: Sequence[str] = (),
    positive_label_values: Sequence[str] = (),
    negative_label_values: Sequence[str] = (),
    required_context_terms: Sequence[str] = (),
    text_fields: Sequence[str] = (),
    treat_other_labeled_as_negative: bool = True,
) -> TargetSpec:
    cleaned_target = str(target_name or "").strip()
    if not cleaned_target:
        raise ValueError("target_name is required")
    cleaned_description = str(target_description or "").strip()
    default_positive = list(_default_positive_terms(cleaned_target))
    return TargetSpec(
        target_name=cleaned_target,
        target_description=cleaned_description,
        positive_terms=_normalize_terms([*default_positive, *positive_terms]),
        negative_terms=_normalize_terms(negative_terms),
        exclude_terms=_normalize_terms(exclude_terms),
        ambiguous_terms=_normalize_terms(ambiguous_terms),
        positive_label_values=_normalize_terms([cleaned_target, *positive_label_values, *positive_terms]),
        negative_label_values=_normalize_terms(negative_label_values),
        required_context_terms=_normalize_terms(required_context_terms),
        text_fields=_normalize_field_selectors(text_fields),
        treat_other_labeled_as_negative=bool(treat_other_labeled_as_negative),
    )


def resolve_target_spec(args) -> TargetSpec:
    file_data: dict[str, Any] = {}
    if args.target_spec_file:
        path = Path(args.target_spec_file)
        file_data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(file_data, dict):
            raise ValueError(f"target spec must be a JSON object: {path}")

    target_name = str(args.target_name or file_data.get("target_name") or "").strip()
    target_description = str(args.target_description or file_data.get("target_description") or "").strip()
    return build_default_target_spec(
        target_name=target_name,
        target_description=target_description,
        positive_terms=[*(file_data.get("positive_terms") or []), *(args.positive_term or [])],
        negative_terms=[*(file_data.get("negative_terms") or []), *(args.negative_term or [])],
        exclude_terms=[*(file_data.get("exclude_terms") or []), *(args.exclude_term or [])],
        ambiguous_terms=[*(file_data.get("ambiguous_terms") or []), *(args.ambiguous_term or [])],
        positive_label_values=[*(file_data.get("positive_label_values") or []), *(args.positive_label_value or [])],
        negative_label_values=[*(file_data.get("negative_label_values") or []), *(args.negative_label_value or [])],
        required_context_terms=[*(file_data.get("required_context_terms") or []), *(args.required_context_term or [])],
        text_fields=[*(file_data.get("text_fields") or []), *(args.text_field or [])],
        treat_other_labeled_as_negative=bool(
            file_data.get("treat_other_labeled_as_negative", args.treat_other_labeled_as_negative)
        ),
    )


def build_llm_target_spec_prompt(spec: TargetSpec) -> str:
    schema = {
        "target_name": spec.target_name,
        "target_description": spec.target_description,
        "positive_terms": list(spec.positive_terms),
        "negative_terms": list(spec.negative_terms),
        "exclude_terms": list(spec.exclude_terms),
        "ambiguous_terms": list(spec.ambiguous_terms),
        "positive_label_values": list(spec.positive_label_values),
        "negative_label_values": list(spec.negative_label_values),
        "required_context_terms": list(spec.required_context_terms),
        "text_fields": list(spec.text_fields),
        "treat_other_labeled_as_negative": bool(spec.treat_other_labeled_as_negative),
    }
    return (
        "Return only JSON for a deterministic Hugging Face image-dataset target spec.\n"
        "Goal: identify examples that should be treated as positive for the target and separate clear negatives.\n"
        "Prefer high-precision terms over broad guesses. Use short normalized phrases.\n"
        "Only include `negative_terms` or `negative_label_values` when they are reliable non-target signals.\n"
        "Use `exclude_terms` for confusing variants that should be skipped instead of treated as negative.\n"
        "Use `required_context_terms` only when the positive phrase is otherwise too ambiguous.\n"
        "Schema:\n"
        f"{json.dumps(schema, indent=2)}\n"
    )


def infer_default_hf_queries(spec: TargetSpec) -> list[str]:
    target = spec.target_name.strip()
    queries = [
        target,
        f"{target} image dataset",
        f"{target} image classification",
        f"{target} object detection",
        f"{target} labeled images",
        f"{target} photos dataset",
    ]
    if spec.target_description:
        queries.append(spec.target_description)
    out: list[str] = []
    seen: set[str] = set()
    for query in queries:
        cleaned = str(query or "").strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
    return out


def infer_image_field(split: object) -> str:
    columns = list(getattr(split, "column_names", None) or [])
    if "image" in columns:
        return "image"
    for column in columns:
        low = str(column).lower()
        if low == "img" or "image" in low:
            return str(column)
    return ""


def iter_examples_deterministic(source: LoadedDatasetSource, *, max_samples: int) -> Iterator[tuple[int, object]]:
    limit = max(1, int(max_samples))
    split = source.split
    if source.streaming:
        for idx, example in enumerate(split.take(limit)):
            yield idx, example
        return
    size = min(limit, len(split))
    for idx in range(size):
        yield idx, split[idx]


def _is_image_like(value: object) -> bool:
    if isinstance(value, Image.Image):
        return True
    shape = getattr(value, "shape", None)
    if isinstance(shape, tuple) and len(shape) in {2, 3}:
        return True
    return False


def _flatten_scalar_values(value: object, *, path: str, out: list[tuple[str, str]], image_field: str) -> None:
    if path == image_field and _is_image_like(value):
        return
    if value is None or isinstance(value, (bytes, bytearray)):
        return
    if _is_image_like(value):
        return
    if isinstance(value, dict):
        for key in sorted(value):
            child_path = f"{path}.{key}" if path else str(key)
            _flatten_scalar_values(value[key], path=child_path, out=out, image_field=image_field)
        return
    if isinstance(value, (list, tuple, set)):
        for idx, item in enumerate(value):
            child_path = f"{path}[{idx}]" if path else f"[{idx}]"
            _flatten_scalar_values(item, path=child_path, out=out, image_field=image_field)
        return
    normalized = _normalize_text(value)
    if normalized:
        out.append((path, normalized))


def collect_search_records(example: dict[str, Any], *, image_field: str, text_fields: Sequence[str]) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    selectors = tuple(_normalize_field_selector(name) for name in text_fields if _normalize_field_selector(name))
    for key in sorted(example):
        path = str(key)
        value = example[key]
        if path == image_field:
            continue
        _flatten_scalar_values(value, path=path, out=records, image_field=image_field)
    if selectors:
        records = [
            (path, value)
            for path, value in records
            if any(
                _normalize_field_selector(path) == selector
                or _normalize_field_selector(path).startswith(selector + ".")
                or _normalize_field_selector(path).startswith(selector + "[")
                for selector in selectors
            )
        ]
    return records


def _contains_term(text: str, term: str) -> bool:
    normalized_text = f" {_normalize_text(text)} "
    normalized_term = f" {_normalize_text(term)} "
    return normalized_term.strip() != "" and normalized_term in normalized_text


def _hits(terms: Sequence[str], texts: Sequence[str]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for term in terms:
        normalized_term = _normalize_text(term)
        if not normalized_term or normalized_term in seen:
            continue
        if any(_contains_term(text, normalized_term) for text in texts):
            out.append(normalized_term)
            seen.add(normalized_term)
    return tuple(out)


def _is_labelish_path(path: str) -> bool:
    low = path.lower()
    parts = re.split(r"[^a-z0-9]+", low)
    return any(part in LABELISH_PARTS for part in parts if part)


def build_match_result(example: dict[str, Any], *, source_id: str, row_index: int, image_field: str, spec: TargetSpec) -> MatchResult:
    records = collect_search_records(example, image_field=image_field, text_fields=spec.text_fields)
    full_text = tuple(value for _, value in records)
    labelish_values = tuple(value for path, value in records if _is_labelish_path(path))
    positive_label_hits = _hits(spec.positive_label_values, labelish_values)
    negative_label_hits = _hits(spec.negative_label_values, labelish_values)
    positive_hits = _hits(spec.positive_terms, full_text)
    negative_hits = _hits(spec.negative_terms, full_text)
    exclude_hits = _hits([*spec.exclude_terms, *spec.ambiguous_terms], full_text)
    context_hits = _hits(spec.required_context_terms, full_text)

    fingerprint_source = json.dumps(
        {
            "source_id": source_id,
            "row_index": int(row_index),
            "records": records,
        },
        ensure_ascii=True,
        sort_keys=True,
    )
    fingerprint = hashlib.sha256(fingerprint_source.encode("utf-8")).hexdigest()

    if exclude_hits:
        return MatchResult(
            label=None,
            reason="excluded_terms",
            fingerprint=fingerprint,
            positive_hits=positive_hits,
            negative_hits=negative_hits,
            exclude_hits=exclude_hits,
            context_hits=context_hits,
            labelish_values=labelish_values,
        )
    if (positive_label_hits or positive_hits) and spec.required_context_terms and not context_hits:
        return MatchResult(
            label=None,
            reason="missing_required_context",
            fingerprint=fingerprint,
            positive_hits=positive_hits,
            negative_hits=negative_hits,
            exclude_hits=exclude_hits,
            context_hits=context_hits,
            labelish_values=labelish_values,
        )
    if positive_label_hits:
        return MatchResult(
            label="positive",
            reason="positive_label_value",
            fingerprint=fingerprint,
            positive_hits=positive_label_hits or positive_hits,
            negative_hits=negative_hits,
            exclude_hits=exclude_hits,
            context_hits=context_hits,
            labelish_values=labelish_values,
        )
    if negative_label_hits:
        return MatchResult(
            label="negative",
            reason="negative_label_value",
            fingerprint=fingerprint,
            positive_hits=positive_hits,
            negative_hits=negative_label_hits or negative_hits,
            exclude_hits=exclude_hits,
            context_hits=context_hits,
            labelish_values=labelish_values,
        )
    if positive_hits and negative_hits:
        return MatchResult(
            label=None,
            reason="conflicting_terms",
            fingerprint=fingerprint,
            positive_hits=positive_hits,
            negative_hits=negative_hits,
            exclude_hits=exclude_hits,
            context_hits=context_hits,
            labelish_values=labelish_values,
        )
    if positive_hits:
        return MatchResult(
            label="positive",
            reason="positive_term",
            fingerprint=fingerprint,
            positive_hits=positive_hits,
            negative_hits=negative_hits,
            exclude_hits=exclude_hits,
            context_hits=context_hits,
            labelish_values=labelish_values,
        )
    if negative_hits:
        return MatchResult(
            label="negative",
            reason="negative_term",
            fingerprint=fingerprint,
            positive_hits=positive_hits,
            negative_hits=negative_hits,
            exclude_hits=exclude_hits,
            context_hits=context_hits,
            labelish_values=labelish_values,
        )
    if labelish_values and spec.treat_other_labeled_as_negative:
        return MatchResult(
            label="negative",
            reason="other_labeled_example",
            fingerprint=fingerprint,
            positive_hits=positive_hits,
            negative_hits=negative_hits,
            exclude_hits=exclude_hits,
            context_hits=context_hits,
            labelish_values=labelish_values,
        )
    return MatchResult(
        label=None,
        reason="unmatched",
        fingerprint=fingerprint,
        positive_hits=positive_hits,
        negative_hits=negative_hits,
        exclude_hits=exclude_hits,
        context_hits=context_hits,
        labelish_values=labelish_values,
    )


def _count_output_files(root: Path, *, positive_dir: str, negative_dir: str) -> dict[str, dict[str, int]]:
    return count_split_class_files(
        root,
        splits=SPLITS,
        classes=(positive_dir, negative_dir),
        exts=IMAGE_EXTS,
    )


def _done(counts: dict[str, dict[str, int]], targets: dict[str, dict[str, int]], classes: Sequence[str]) -> bool:
    return all(counts[split][cls] >= targets[split][cls] for split in SPLITS for cls in classes)


def choose_split(
    *,
    fingerprint: str,
    cls: str,
    counts: dict[str, dict[str, int]],
    targets: dict[str, dict[str, int]],
    source_split_counts: dict[str, dict[str, int]],
    max_per_source_split_class: int,
) -> str | None:
    candidates = [
        split
        for split in SPLITS
        if counts[split][cls] < targets[split][cls] and source_split_counts[split][cls] < max_per_source_split_class
    ]
    if not candidates:
        return None

    total_weight = sum(max(0, int(targets[split][cls])) for split in SPLITS)
    hashed = int(hashlib.sha256(f"{fingerprint}|{cls}".encode("utf-8")).hexdigest(), 16)
    if total_weight > 0:
        pick = hashed % total_weight
        cursor = 0
        for split in SPLITS:
            weight = max(0, int(targets[split][cls]))
            cursor += weight
            if pick < cursor and split in candidates:
                return split

    return min(
        candidates,
        key=lambda split: (
            counts[split][cls] / float(max(1, targets[split][cls])),
            int(hashlib.sha256(f"{fingerprint}|fallback|{split}".encode("utf-8")).hexdigest(), 16),
        ),
    )


def build_target_dataset_from_sources(
    sources: Sequence[LoadedDatasetSource],
    *,
    spec: TargetSpec,
    out: Path,
    positive_dir: str,
    negative_dir: str,
    train_per_class: int,
    val_per_class: int,
    test_per_class: int,
    quality_policy: ImageQualityPolicy,
    near_hamming: int,
    near_window: int,
    max_per_source_class: int,
    max_per_source_split_class: int,
    max_samples_per_source: int,
    jpeg_quality: int,
) -> dict[str, Any]:
    classes = (positive_dir, negative_dir)
    for split in SPLITS:
        for cls in classes:
            (out / split / cls).mkdir(parents=True, exist_ok=True)

    targets = {
        "train": {positive_dir: int(train_per_class), negative_dir: int(train_per_class)},
        "val": {positive_dir: int(val_per_class), negative_dir: int(val_per_class)},
        "test": {positive_dir: int(test_per_class), negative_dir: int(test_per_class)},
    }
    counts = _count_output_files(out, positive_dir=positive_dir, negative_dir=negative_dir)
    deduper = ImageDeduper.from_output(out, splits=SPLITS, classes=classes)

    global_rejects: DefaultDict[str, int] = defaultdict(int)
    source_reports: list[dict[str, Any]] = []

    for source in sources:
        if _done(counts, targets, classes):
            break

        image_field = infer_image_field(source.split)
        if not image_field:
            source_reports.append(
                {
                    "source": source.source_id,
                    "split_name": source.split_name,
                    "status": "missing_image_field",
                }
            )
            continue

        source_counts = {positive_dir: 0, negative_dir: 0}
        source_split_counts = {split: {cls: 0 for cls in classes} for split in SPLITS}
        source_rejects: DefaultDict[str, int] = defaultdict(int)
        match_reasons: DefaultDict[str, int] = defaultdict(int)
        processed_total = 0
        labeled_total = 0
        skipped_total = 0

        for row_index, example in iter_examples_deterministic(source, max_samples=max_samples_per_source):
            if _done(counts, targets, classes):
                break
            processed_total += 1
            if not isinstance(example, dict):
                source_rejects["non_mapping_example"] += 1
                skipped_total += 1
                continue

            match = build_match_result(example, source_id=source.source_id, row_index=row_index, image_field=image_field, spec=spec)
            match_reasons[match.reason] += 1
            if match.label is None:
                source_rejects[match.reason] += 1
                skipped_total += 1
                continue

            cls = positive_dir if match.label == "positive" else negative_dir
            if source_counts[cls] >= int(max_per_source_class):
                source_rejects["source_class_cap"] += 1
                skipped_total += 1
                continue

            split = choose_split(
                fingerprint=match.fingerprint,
                cls=cls,
                counts=counts,
                targets=targets,
                source_split_counts=source_split_counts,
                max_per_source_split_class=int(max_per_source_split_class),
            )
            if split is None:
                source_rejects["no_split_needed"] += 1
                skipped_total += 1
                continue

            image = open_example_image(example, image_field)
            if image is None:
                source_rejects["decode_fail"] += 1
                skipped_total += 1
                continue

            ok, reason = passes_quality_filters(image, quality_policy)
            if not ok:
                source_rejects[reason] += 1
                global_rejects[reason] += 1
                skipped_total += 1
                continue

            duplicate_reason = deduper.duplicate_reason(
                image,
                cls=cls,
                near_hamming=int(near_hamming),
                near_window=int(near_window),
            )
            if duplicate_reason is not None:
                source_rejects[duplicate_reason] += 1
                global_rejects[duplicate_reason] += 1
                skipped_total += 1
                continue

            dst = out / split / cls / (
                f"source={_slugify(source.source_id)}__reason={_slugify(match.reason)}__"
                f"id={match.fingerprint[:12]}__{split}_{cls}_{counts[split][cls]:07d}.jpg"
            )
            save_img(image, dst, quality=jpeg_quality)
            deduper.remember(image, cls=cls)
            counts[split][cls] += 1
            source_counts[cls] += 1
            source_split_counts[split][cls] += 1
            labeled_total += 1

        source_reports.append(
            {
                "source": source.source_id,
                "split_name": source.split_name,
                "status": "completed",
                "processed_total": int(processed_total),
                "accepted_total": int(labeled_total),
                "accepted_positive": int(source_counts[positive_dir]),
                "accepted_negative": int(source_counts[negative_dir]),
                "accepted_by_split": source_split_counts,
                "skipped_total": int(skipped_total),
                "rejections": dict(source_rejects),
                "match_reasons": dict(match_reasons),
                "image_field": image_field,
            }
        )

    final_counts = _count_output_files(out, positive_dir=positive_dir, negative_dir=negative_dir)
    shortfalls = []
    for split in SPLITS:
        for cls in classes:
            have_n = final_counts[split][cls]
            need_n = targets[split][cls]
            if have_n < need_n:
                shortfalls.append(f"{split}/{cls}:{have_n}<{need_n}")

    summary: dict[str, Any] = {
        "target_spec": asdict(spec),
        "target_aliases": {
            "positive_dir": positive_dir,
            "negative_dir": negative_dir,
            "positive_label": spec.target_name,
            "negative_label": f"not_{_slugify(spec.target_name)}",
        },
        "targets": targets,
        "final_counts": final_counts,
        "full_targets_ok": len(shortfalls) == 0,
        "shortfalls": shortfalls,
        "global_rejections": dict(global_rejects),
        "source_reports": source_reports,
    }

    out.mkdir(parents=True, exist_ok=True)
    (out / "target_spec_resolved.json").write_text(json.dumps(asdict(spec), indent=2) + "\n", encoding="utf-8")
    (out / "target_label_aliases.json").write_text(json.dumps(summary["target_aliases"], indent=2) + "\n", encoding="utf-8")
    (out / "target_dataset_build_report.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out / "dataset_run_summary.json").write_text(
        json.dumps(
            {
                "positive_label": spec.target_name,
                "negative_label": f"not_{_slugify(spec.target_name)}",
                "full_targets_ok": bool(summary["full_targets_ok"]),
                "report_path": str((out / "target_dataset_build_report.json").resolve()),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def _loaded_sources_from_ids(
    source_ids: Sequence[str],
    *,
    streaming: bool,
    cache_dir: str,
    token_env: str,
) -> list[LoadedDatasetSource]:
    token, token_source = resolve_hf_token_value(token_env)
    if token:
        if token_source.startswith("env:"):
            print(f"using_token_env={token_source.split(':', 1)[1]}")
        else:
            print(f"using_token_source={token_source}")
    else:
        print(f"warning_no_token env={token_env} (public datasets still work; hf auth login also works)")

    loaded: list[LoadedDatasetSource] = []
    for source_id in source_ids:
        try:
            loaded.append(
                load_hf_dataset_source(
                    source_id,
                    token=token,
                    streaming=bool(streaming),
                    cache_dir=cache_dir or None,
                )
            )
        except Exception as exc:
            print(f"skip_source={source_id} reason=load_failed:{exc}")
    return loaded


def validate_output_class_dirs(positive_dir: str, negative_dir: str, *, allow_nonstandard: bool) -> None:
    if positive_dir == negative_dir:
        raise SystemExit("positive_and_negative_dirs_must_differ")
    if allow_nonstandard:
        return
    if positive_dir != "ai" or negative_dir != "real":
        raise SystemExit(
            "nonstandard_class_dirs_not_supported_by_repo_trainer "
            f"positive_dir={positive_dir} negative_dir={negative_dir} expected=ai/real "
            "pass --allow-nonstandard-class-dirs only if you are exporting for tooling outside this repo"
        )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build a deterministic target-vs-background training dataset from Hugging Face image datasets")
    ap.add_argument("--target-name", default="")
    ap.add_argument("--target-description", default="")
    ap.add_argument("--target-spec-file", default="")
    ap.add_argument("--positive-term", action="append", default=[])
    ap.add_argument("--negative-term", action="append", default=[])
    ap.add_argument("--exclude-term", action="append", default=[])
    ap.add_argument("--ambiguous-term", action="append", default=[])
    ap.add_argument("--positive-label-value", action="append", default=[])
    ap.add_argument("--negative-label-value", action="append", default=[])
    ap.add_argument("--required-context-term", action="append", default=[])
    ap.add_argument("--text-field", action="append", default=[])
    ap.add_argument("--treat-other-labeled-as-negative", action="store_true", default=True)
    ap.add_argument("--no-treat-other-labeled-as-negative", dest="treat_other_labeled_as_negative", action="store_false")
    ap.add_argument("--emit-llm-prompt", action="store_true", default=False)
    ap.add_argument("--print-target-spec", action="store_true", default=False)

    ap.add_argument("--out", default="")
    ap.add_argument("--positive-dir-name", default="ai")
    ap.add_argument("--negative-dir-name", default="real")
    ap.add_argument("--allow-nonstandard-class-dirs", action="store_true", default=False)
    ap.add_argument("--train-per-class", type=int, default=2000)
    ap.add_argument("--val-per-class", type=int, default=400)
    ap.add_argument("--test-per-class", type=int, default=400)
    ap.add_argument("--near-hamming", type=int, default=2)
    ap.add_argument("--near-window", type=int, default=2400)
    ap.add_argument("--min-side", type=int, default=160)
    ap.add_argument("--max-aspect-ratio", type=float, default=4.0)
    ap.add_argument("--min-entropy", type=float, default=3.0)
    ap.add_argument("--max-per-source-class", type=int, default=800)
    ap.add_argument("--max-per-source-split-class", type=int, default=320)
    ap.add_argument("--max-samples-per-source", type=int, default=30000)
    ap.add_argument("--jpeg-quality", type=int, default=92)
    ap.add_argument("--require-full-targets", action="store_true", default=False)

    ap.add_argument("--sources-file", default="")
    ap.add_argument("--extra-source", action="append", default=[])
    ap.add_argument("--discover-hf", action="store_true", default=True)
    ap.add_argument("--no-discover-hf", dest="discover_hf", action="store_false")
    ap.add_argument("--hf-query", action="append", default=[])
    ap.add_argument("--hf-discovery-limit", type=int, default=120)
    ap.add_argument("--hf-max-sources", type=int, default=120)
    ap.add_argument("--hf-min-downloads", type=int, default=5)
    ap.add_argument("--hf-min-likes", type=int, default=0)
    ap.add_argument("--hf-min-quality-score", type=float, default=0.0)
    ap.add_argument("--hf-print-top", type=int, default=24)
    ap.add_argument("--hf-discovery-workers", type=int, default=8)
    ap.add_argument("--hf-query-pause-ms", type=int, default=0)
    ap.add_argument("--hf-license-allow", action="append", default=list(DEFAULT_ALLOWED_LICENSE_TAGS))
    ap.add_argument("--hf-require-open-license", action="store_true", default=True)
    ap.add_argument("--no-hf-require-open-license", dest="hf_require_open_license", action="store_false")
    ap.add_argument("--hf-cache-file", default="")
    ap.add_argument("--hf-cache-only-if-present", action="store_true", default=False)
    ap.add_argument("--hf-audit-sources", action="store_true", default=True)
    ap.add_argument("--no-hf-audit-sources", dest="hf_audit_sources", action="store_false")
    ap.add_argument("--hf-audit-file", default="")
    ap.add_argument("--hf-audit-min-rows", type=int, default=50)
    ap.add_argument("--hf-audit-require-image-field", action="store_true", default=True)
    ap.add_argument("--no-hf-audit-require-image-field", dest="hf_audit_require_image_field", action="store_false")
    ap.add_argument("--hf-audit-require-label-field", action="store_true", default=False)
    ap.add_argument("--hf-audit-filter-to-approved", action="store_true", default=True)
    ap.add_argument("--no-hf-audit-filter-to-approved", dest="hf_audit_filter_to_approved", action="store_false")
    ap.add_argument("--discover-only", action="store_true", default=False)

    ap.add_argument("--streaming", action="store_true", default=True)
    ap.add_argument("--no-streaming", dest="streaming", action="store_false")
    ap.add_argument("--cache-dir", default=HF_CACHE_DIR_DEFAULT)
    ap.add_argument("--token-env", default="HF_TOKEN")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    spec = resolve_target_spec(args)
    slug = _slugify(spec.target_name)
    validate_output_class_dirs(
        args.positive_dir_name,
        args.negative_dir_name,
        allow_nonstandard=bool(args.allow_nonstandard_class_dirs),
    )
    args.no_default_sources = True
    if not args.hf_query:
        args.hf_query = infer_default_hf_queries(spec)
    if not args.out:
        args.out = f"./.local/target_training_data/{slug}"
    if not args.hf_cache_file:
        args.hf_cache_file = f"./.local/reports/target_sources__{slug}.txt"
    if not args.hf_audit_file:
        args.hf_audit_file = f"./.local/reports/target_source_audit__{slug}.jsonl"

    validate_collection_io_paths(
        out=args.out,
        sources_file=args.sources_file or None,
        hf_cache_file=args.hf_cache_file or None,
        hf_audit_file=args.hf_audit_file or None,
        cache_dir=args.cache_dir or None,
    )

    if args.emit_llm_prompt:
        print(build_llm_target_spec_prompt(spec))
        return 0
    if args.print_target_spec:
        print(json.dumps(asdict(spec), indent=2))
        return 0

    source_ids = build_source_list(args)
    if args.discover_only:
        print(json.dumps({"target_name": spec.target_name, "source_ids": source_ids}, indent=2))
        return 0
    if not source_ids:
        raise SystemExit("no_hf_sources_resolved")

    out = Path(args.out)
    loaded_sources = _loaded_sources_from_ids(
        source_ids,
        streaming=bool(args.streaming),
        cache_dir=str(args.cache_dir or ""),
        token_env=str(args.token_env),
    )
    if not loaded_sources:
        raise SystemExit("no_sources_loaded")

    summary = build_target_dataset_from_sources(
        loaded_sources,
        spec=spec,
        out=out,
        positive_dir=args.positive_dir_name,
        negative_dir=args.negative_dir_name,
        train_per_class=args.train_per_class,
        val_per_class=args.val_per_class,
        test_per_class=args.test_per_class,
        quality_policy=ImageQualityPolicy(
            min_side=args.min_side,
            max_aspect_ratio=args.max_aspect_ratio,
            min_entropy=args.min_entropy,
        ),
        near_hamming=args.near_hamming,
        near_window=args.near_window,
        max_per_source_class=args.max_per_source_class,
        max_per_source_split_class=args.max_per_source_split_class,
        max_samples_per_source=args.max_samples_per_source,
        jpeg_quality=args.jpeg_quality,
    )
    print(json.dumps(summary["final_counts"], indent=2))
    print(f"report={out / 'target_dataset_build_report.json'}")
    print(f"dataset_ready={out}")
    if args.require_full_targets and not bool(summary["full_targets_ok"]):
        raise SystemExit("dataset_incomplete: " + ",".join(summary["shortfalls"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
