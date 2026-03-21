from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def _walk_images(root: Path):
    for p in root.rglob("*"):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            yield p


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _dhash(path: Path) -> str:
    try:
        from PIL import Image
    except Exception:
        return ""
    img = Image.open(path).convert("L").resize((9, 8))
    px = list(img.tobytes())
    bits = []
    for y in range(8):
        row = px[y * 9 : (y + 1) * 9]
        for x in range(8):
            bits.append("1" if row[x] > row[x + 1] else "0")
    return f"{int(''.join(bits), 2):016x}"


def _hamming_hex(a: str, b: str) -> int:
    if not a or not b:
        return 64
    return (int(a, 16) ^ int(b, 16)).bit_count()


def cmd_manifest(data_root: str, out_csv: str):
    root = Path(data_root)
    rows = []
    for p in _walk_images(root):
        rel = p.relative_to(root)
        parts = rel.parts
        split = parts[0] if len(parts) > 0 else ""
        klass = parts[1] if len(parts) > 1 else ""
        rows.append({
            "path": str(rel),
            "split": split,
            "class": klass,
            "sha256": _sha256(p),
            "dhash": _dhash(p),
            "size": p.stat().st_size,
        })

    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "split", "class", "sha256", "dhash", "size"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"rows={len(rows)} saved={out}")


def cmd_dedupe(data_root: str, dry_run: bool):
    root = Path(data_root)
    seen: dict[str, Path] = {}
    dupes: list[Path] = []

    for p in _walk_images(root):
        h = _sha256(p)
        if h in seen:
            dupes.append(p)
        else:
            seen[h] = p

    print(f"exact_duplicates={len(dupes)}")
    for p in dupes[:20]:
        print(f"dupe={p}")

    if not dry_run:
        for p in dupes:
            p.unlink(missing_ok=True)
        print("deleted exact duplicates")


def cmd_near_dupes(data_root: str, max_images: int, max_hamming: int):
    root = Path(data_root)
    items = []
    for i, p in enumerate(_walk_images(root)):
        if i >= max_images:
            break
        items.append((p, _dhash(p)))

    pairs = []
    for i in range(len(items)):
        p1, h1 = items[i]
        for j in range(i + 1, len(items)):
            p2, h2 = items[j]
            d = _hamming_hex(h1, h2)
            if d <= max_hamming:
                pairs.append((str(p1), str(p2), d))

    print(json.dumps({"near_dupe_pairs": pairs[:200], "count": len(pairs)}, indent=2))


def cmd_balance_report(data_root: str):
    root = Path(data_root)
    counts: dict[str, int] = {}
    for p in _walk_images(root):
        rel = p.relative_to(root)
        key = "/".join(rel.parts[:2]) if len(rel.parts) >= 2 else "unknown"
        counts[key] = counts.get(key, 0) + 1

    min_count = min(counts.values()) if counts else 0
    report = {
        "counts": counts,
        "suggested_balanced_per_bucket": min_count,
    }
    print(json.dumps(report, indent=2))


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _count_image_files(root: Path) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {
        split: {cls: 0 for cls in ("ai", "real")}
        for split in ("train", "val", "test")
    }
    for split in ("train", "val", "test"):
        for cls in ("ai", "real"):
            bucket = root / split / cls
            if not bucket.exists():
                continue
            counts[split][cls] = sum(1 for p in bucket.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"})
    return counts


def _count_video_files(root: Path) -> dict[str, dict[str, int]]:
    exts = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
    counts: dict[str, dict[str, int]] = {
        split: {cls: 0 for cls in ("ai", "real")}
        for split in ("train", "val")
    }
    for split in ("train", "val"):
        for cls in ("ai", "real"):
            bucket = root / split / cls
            if not bucket.exists():
                continue
            counts[split][cls] = sum(1 for p in bucket.iterdir() if p.is_file() and p.suffix.lower() in exts)
    return counts


def _load_manifest_entries(path: Path) -> list[dict]:
    entries: list[dict] = []
    if not path.exists():
        return entries
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            entries.append(item)
    return entries


def cmd_collection_status(
    data_root: str,
    incremental_root: str = "",
    prepared_root: str = "",
    video_root: str = "",
    recent_sources: int = 8,
) -> None:
    root = Path(data_root)
    report_path = root / "dataset_build_report.json"
    run_summary_path = root / "dataset_run_summary.json"
    state_path = root / "dataset_state.json"
    manifest_path = root / "dataset_source_manifest.jsonl"

    build_report = _read_json(report_path)
    run_summary = _read_json(run_summary_path)
    state = _read_json(state_path)
    manifest_entries = _load_manifest_entries(manifest_path)

    latest_by_source: dict[str, dict] = {}
    for entry in manifest_entries:
        source = str(entry.get("source", "")).strip()
        if source:
            latest_by_source[source] = entry

    latest_entries = list(latest_by_source.values())
    latest_entries.sort(key=lambda item: str(item.get("finished_utc", "")), reverse=True)
    skipped_sources = [entry for entry in latest_entries if bool(entry.get("skip_future_runs", False))]
    completed_sources = [entry for entry in latest_entries if str(entry.get("status", "")) == "completed"]
    failed_sources = [
        entry for entry in latest_entries
        if str(entry.get("status", "")) not in {"completed", "skipped_manifest", ""}
    ]

    source_candidates = state.get("source_candidates", [])
    if not isinstance(source_candidates, list):
        source_candidates = []
    source_candidates_count = len(source_candidates)
    full_targets_ok = bool(state.get("full_targets_ok", build_report.get("full_targets_ok", False)))
    resume_needed = not full_targets_ok

    report = {
        "data_root": str(root.resolve()),
        "paths": {
            "build_report": str(report_path.resolve()),
            "run_summary": str(run_summary_path.resolve()),
            "dataset_state": str(state_path.resolve()),
            "source_manifest": str(manifest_path.resolve()),
        },
        "current_counts": _count_image_files(root),
        "build_report_present": report_path.exists(),
        "run_summary_present": run_summary_path.exists(),
        "dataset_state_present": state_path.exists(),
        "source_manifest_present": manifest_path.exists(),
        "full_targets_ok": full_targets_ok,
        "manifest": {
            "entries": len(manifest_entries),
            "unique_sources": len(latest_entries),
            "completed_sources": len(completed_sources),
            "failed_sources": len(failed_sources),
            "skipped_future_runs": len(skipped_sources),
            "accepted_total": int(sum(int(entry.get("accepted_total", 0)) for entry in completed_sources)),
            "processed_total": int(sum(int(entry.get("processed_total", 0)) for entry in completed_sources)),
            "last_finished_utc": latest_entries[0].get("finished_utc") if latest_entries else "",
            "recent_sources": latest_entries[: max(0, int(recent_sources))],
        },
        "resume": {
            "resume_supported": manifest_path.exists() or report_path.exists(),
            "resume_needed": resume_needed,
            "source_candidates": source_candidates_count,
            "remaining_candidates_estimate": max(0, source_candidates_count - len(skipped_sources)) if source_candidates_count > 0 else None,
            "recommended_command": "./local.sh collect" if resume_needed else "./local.sh train",
        },
        "run_summary": run_summary,
    }

    if incremental_root:
        report["incremental_root"] = {
            "path": str(Path(incremental_root).resolve()),
            "counts": _count_image_files(Path(incremental_root)),
        }
    if prepared_root:
        report["prepared_training_root"] = {
            "path": str(Path(prepared_root).resolve()),
            "counts": _count_image_files(Path(prepared_root)),
        }
    if video_root:
        report["video_root"] = {
            "path": str(Path(video_root).resolve()),
            "counts": _count_video_files(Path(video_root)),
        }

    print(json.dumps(report, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description="Dataset hygiene tools")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_manifest = sub.add_parser("manifest", help="Build dataset manifest CSV")
    p_manifest.add_argument("--data", required=True)
    p_manifest.add_argument("--out", default="./artifacts/dataset_manifest.csv")

    p_dedupe = sub.add_parser("dedupe", help="Find/remove exact duplicate files")
    p_dedupe.add_argument("--data", required=True)
    p_dedupe.add_argument("--dry-run", action="store_true")

    p_nd = sub.add_parser("near-dupes", help="Find near duplicates via dHash")
    p_nd.add_argument("--data", required=True)
    p_nd.add_argument("--max-images", type=int, default=1500)
    p_nd.add_argument("--max-hamming", type=int, default=6)

    p_bal = sub.add_parser("balance-report", help="Report class balance by split/class buckets")
    p_bal.add_argument("--data", required=True)

    p_status = sub.add_parser("collection-status", help="Report dataset collection/build state and resume info")
    p_status.add_argument("--data", required=True)
    p_status.add_argument("--incremental", default="")
    p_status.add_argument("--prepared", default="")
    p_status.add_argument("--video", default="")
    p_status.add_argument("--recent-sources", type=int, default=8)

    args = ap.parse_args()

    if args.cmd == "manifest":
        cmd_manifest(args.data, args.out)
    elif args.cmd == "dedupe":
        cmd_dedupe(args.data, args.dry_run)
    elif args.cmd == "near-dupes":
        cmd_near_dupes(args.data, args.max_images, args.max_hamming)
    elif args.cmd == "balance-report":
        cmd_balance_report(args.data)
    elif args.cmd == "collection-status":
        cmd_collection_status(
            args.data,
            incremental_root=args.incremental,
            prepared_root=args.prepared,
            video_root=args.video,
            recent_sources=args.recent_sources,
        )


if __name__ == "__main__":
    main()
