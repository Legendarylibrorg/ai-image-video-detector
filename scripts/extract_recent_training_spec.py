from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Iterable

from script_support import ensure_src_path

ensure_src_path()

from ai_image_detector.dataset_layout import IMAGE_EXTS

SCHEMA_TEMPLATE = {
    "target_name": "",
    "target_description": "",
    "positive_terms": [],
    "negative_terms": [],
    "exclude_terms": [],
    "ambiguous_terms": [],
    "positive_label_values": [],
    "negative_label_values": [],
    "required_context_terms": [],
    "text_fields": [],
    "treat_other_labeled_as_negative": True,
}
GENERIC_TOKENS = {
    "ai",
    "real",
    "train",
    "val",
    "test",
    "source",
    "model",
    "generator",
    "reason",
    "image",
    "images",
    "jpg",
    "jpeg",
    "png",
    "webp",
    "bmp",
    "tif",
    "tiff",
    "id",
}


def _read_json_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _utc_iso_from_mtime_ns(mtime_ns: int) -> str:
    return datetime.fromtimestamp(mtime_ns / 1_000_000_000, tz=timezone.utc).isoformat()


def _resolve_incremental_root(root: Path) -> Path:
    if (root / "train").is_dir():
        return root / "train"
    return root


def _parse_filename_tags(path: Path) -> dict[str, str]:
    tags: dict[str, str] = {}
    for part in path.stem.split("__"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip().lower()
        value = value.strip().lower()
        if key and value:
            tags[key] = value
    return tags


def _tokenize_filename(path: Path) -> list[str]:
    tokens: list[str] = []
    for piece in re.split(r"[^a-z0-9]+", path.stem.lower()):
        if not piece or piece in GENERIC_TOKENS:
            continue
        if len(piece) >= 12 and re.fullmatch(r"[a-f0-9]+", piece):
            continue
        if piece.isdigit():
            continue
        tokens.append(piece)
    return tokens


def _flatten_scalar_text(value: object, *, path: str, out: list[tuple[str, str]]) -> None:
    if value is None or isinstance(value, (bytes, bytearray)):
        return
    if isinstance(value, dict):
        for key in sorted(value):
            child_path = f"{path}.{key}" if path else str(key)
            _flatten_scalar_text(value[key], path=child_path, out=out)
        return
    if isinstance(value, (list, tuple, set)):
        for idx, item in enumerate(value):
            child_path = f"{path}[{idx}]" if path else f"[{idx}]"
            _flatten_scalar_text(item, path=child_path, out=out)
        return
    text = str(value).strip()
    if not text:
        return
    normalized = re.sub(r"\s+", " ", text)
    out.append((path, normalized))


def _read_sidecar_summary(image_path: Path, *, max_items: int = 12) -> dict[str, Any]:
    sidecar = image_path.with_suffix(".json")
    data = _read_json_dict(sidecar)
    if not data:
        return {}
    flattened: list[tuple[str, str]] = []
    _flatten_scalar_text(data, path="", out=flattened)
    text_fields = []
    for key, value in flattened[: max(0, int(max_items))]:
        text_fields.append({"field": key or "<root>", "value": value[:180]})
    return {
        "sidecar_path": str(sidecar),
        "text_fields": text_fields,
    }


def find_latest_target_spec(target_training_root: Path) -> dict[str, Any] | None:
    if not target_training_root.exists():
        return None
    candidates = sorted(target_training_root.rglob("target_spec_resolved.json"))
    if not candidates:
        return None
    latest = max(candidates, key=lambda path: path.stat().st_mtime_ns)
    parent = latest.parent
    aliases_path = parent / "target_label_aliases.json"
    report_path = parent / "target_dataset_build_report.json"
    return {
        "mode": "resolved_target_spec",
        "dataset_root": str(parent.resolve()),
        "spec_path": str(latest.resolve()),
        "aliases_path": str(aliases_path.resolve()) if aliases_path.exists() else "",
        "report_path": str(report_path.resolve()) if report_path.exists() else "",
        "mtime_utc": _utc_iso_from_mtime_ns(latest.stat().st_mtime_ns),
        "spec": _read_json_dict(latest),
        "aliases": _read_json_dict(aliases_path),
        "report_summary": {
            "full_targets_ok": bool(_read_json_dict(report_path).get("full_targets_ok", False)) if report_path.exists() else False,
            "shortfalls": _read_json_dict(report_path).get("shortfalls", []) if report_path.exists() else [],
        },
    }


def summarize_recent_incremental_data(incremental_root: Path, *, recent_count: int = 16) -> dict[str, Any]:
    root = _resolve_incremental_root(incremental_root)
    classes = ("ai", "real")
    summary: dict[str, Any] = {
        "mode": "recent_incremental_summary",
        "incremental_root": str(root.resolve()),
        "recent_count_per_class": int(recent_count),
        "classes": {},
        "totals": {},
    }

    source_counter: Counter[str] = Counter()
    token_counter: Counter[str] = Counter()
    sidecar_field_counter: Counter[str] = Counter()
    latest_mtime_ns = 0

    for cls in classes:
        bucket = root / cls
        files = []
        if bucket.exists():
            files = sorted(
                (
                    path
                    for path in bucket.iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTS
                ),
                key=lambda path: path.stat().st_mtime_ns,
                reverse=True,
            )
        recent = files[: max(0, int(recent_count))]
        examples = []
        for path in recent:
            stat = path.stat()
            latest_mtime_ns = max(latest_mtime_ns, stat.st_mtime_ns)
            tags = _parse_filename_tags(path)
            source = tags.get("source", "")
            if source:
                source_counter[source] += 1
            for token in _tokenize_filename(path):
                token_counter[token] += 1
            sidecar = _read_sidecar_summary(path)
            for field in sidecar.get("text_fields", []):
                sidecar_field_counter[str(field.get("field", ""))] += 1
            examples.append(
                {
                    "path": str(path.relative_to(root)),
                    "mtime_utc": _utc_iso_from_mtime_ns(stat.st_mtime_ns),
                    "filename_tags": tags,
                    "sidecar": sidecar,
                }
            )
        summary["classes"][cls] = {
            "total_files": len(files),
            "recent_examples": examples,
        }
        summary["totals"][cls] = len(files)

    summary["latest_addition_utc"] = _utc_iso_from_mtime_ns(latest_mtime_ns) if latest_mtime_ns else ""
    summary["top_sources"] = [
        {"source": source, "count": count}
        for source, count in source_counter.most_common(12)
    ]
    summary["top_filename_tokens"] = [
        {"token": token, "count": count}
        for token, count in token_counter.most_common(20)
    ]
    summary["top_sidecar_fields"] = [
        {"field": field, "count": count}
        for field, count in sidecar_field_counter.most_common(20)
        if field
    ]
    return summary


def build_llm_spec_extraction_prompt(summary: dict[str, Any]) -> str:
    compact = {
        "incremental_root": summary.get("incremental_root", ""),
        "latest_addition_utc": summary.get("latest_addition_utc", ""),
        "totals": summary.get("totals", {}),
        "top_sources": summary.get("top_sources", []),
        "top_filename_tokens": summary.get("top_filename_tokens", []),
        "top_sidecar_fields": summary.get("top_sidecar_fields", []),
        "classes": summary.get("classes", {}),
    }
    return (
        "Return only JSON matching this schema for the target-dataset spec:\n"
        f"{json.dumps(SCHEMA_TEMPLATE, indent=2)}\n"
        "Infer the spec from the recent incremental training-data summary below.\n"
        "Use only evidence that is clearly supported by filenames, tags, or sidecar fields.\n"
        "If the target is unclear, keep arrays empty and use a short conservative `target_name` placeholder.\n"
        "Prefer precise positive terms, use `exclude_terms` for confusing variants, and do not invent unsupported negatives.\n"
        "Recent incremental training-data summary:\n"
        f"{json.dumps(compact, indent=2)}\n"
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Extract the most recent training-data target spec or build an LLM-ready spec prompt from recent additions")
    ap.add_argument("--target-training-root", default="./.local/target_training_data")
    ap.add_argument("--incremental-root", default="./data_new")
    ap.add_argument("--recent-count", type=int, default=16)
    ap.add_argument("--emit", choices=["json", "prompt", "both"], default="json")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    target_training_root = Path(args.target_training_root)
    incremental_root = Path(args.incremental_root)

    latest = find_latest_target_spec(target_training_root)
    if latest is not None:
        if args.emit in {"json", "both"}:
            print(json.dumps(latest, indent=2))
        if args.emit in {"prompt", "both"}:
            prompt = (
                "Resolved target spec already exists. Use this JSON directly or revise it if needed:\n"
                + json.dumps(latest.get("spec", {}), indent=2)
                + "\n"
            )
            print(prompt)
        return 0

    summary = summarize_recent_incremental_data(incremental_root, recent_count=args.recent_count)
    prompt = build_llm_spec_extraction_prompt(summary)
    payload = {
        "summary": summary,
        "schema_template": SCHEMA_TEMPLATE,
        "prompt": prompt,
    }
    if args.emit in {"json", "both"}:
        print(json.dumps(payload, indent=2))
    if args.emit in {"prompt", "both"}:
        print(prompt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
