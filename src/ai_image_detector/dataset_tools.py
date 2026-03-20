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

    args = ap.parse_args()

    if args.cmd == "manifest":
        cmd_manifest(args.data, args.out)
    elif args.cmd == "dedupe":
        cmd_dedupe(args.data, args.dry_run)
    elif args.cmd == "near-dupes":
        cmd_near_dupes(args.data, args.max_images, args.max_hamming)
    elif args.cmd == "balance-report":
        cmd_balance_report(args.data)


if __name__ == "__main__":
    main()
