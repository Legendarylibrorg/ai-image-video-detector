from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
from pathlib import Path
import random
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset
from PIL import Image, ImageFilter


def normalize_label(v) -> Optional[str]:
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
    bits = []
    for y in range(8):
        row = px[y * 9 : (y + 1) * 9]
        for x in range(8):
            bits.append("1" if row[x] > row[x + 1] else "0")
    return f"{int(''.join(bits), 2):016x}"


def hamming_hex(a: str, b: str) -> int:
    return (int(a, 16) ^ int(b, 16)).bit_count()


def open_example_image(ex, image_field: str) -> Optional[Image.Image]:
    img = ex.get(image_field)
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    try:
        return Image.fromarray(img).convert("RGB")
    except Exception:
        return None


def find_fields(ds_split) -> Tuple[str, str]:
    cols = ds_split.column_names
    image_field = "image" if "image" in cols else next((c for c in cols if "image" in c.lower() or "img" == c.lower()), None)
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


def main():
    ap = argparse.ArgumentParser(description="Build multi-source, deduped, hard-negative dataset")
    ap.add_argument("--out", default="data_best")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-per-class", type=int, default=30000)
    ap.add_argument("--val-per-class", type=int, default=7000)
    ap.add_argument("--test-per-class", type=int, default=7000)
    ap.add_argument("--near-hamming", type=int, default=2)
    args = ap.parse_args()

    random.seed(args.seed)

    # Several public candidates; unavailable ones are skipped automatically.
    sources = [
        "Hemg/AI-Generated-vs-Real-Images-Datasets",
        "dragonintelligence/CIFAKE-image-dataset",
        "batgre/CIFAKE",
        "Ronduck/real-fake-images-deduplicated",
        "JamieWithofs/Deepfake-and-real-images",
        "JamieWithofs/Deepfake-and-real-images-2",
    ]

    out = Path(args.out)
    for split in ["train", "val", "test"]:
        for cls in ["ai", "real"]:
            (out / split / cls).mkdir(parents=True, exist_ok=True)

    targets = {
        "train": {"ai": args.train_per_class, "real": args.train_per_class},
        "val": {"ai": args.val_per_class, "real": args.val_per_class},
        "test": {"ai": args.test_per_class, "real": args.test_per_class},
    }
    counts: Dict[str, Dict[str, int]] = {s: {"ai": 0, "real": 0} for s in ["train", "val", "test"]}

    # Global dedupe to prevent leakage across splits
    seen_exact = set()
    seen_dhash_by_cls: Dict[str, List[str]] = defaultdict(list)

    per_class_queue: Dict[str, List[Tuple[Image.Image, str]]] = defaultdict(list)

    for src in sources:
        try:
            ds = load_dataset(src)
        except Exception as e:
            print(f"skip_source={src} reason={e}")
            continue

        split_name = "train" if "train" in ds else list(ds.keys())[0]
        split = ds[split_name]
        image_field, label_field = find_fields(split)

        idxs = list(range(len(split)))
        random.shuffle(idxs)

        used = 0
        for i in idxs:
            ex = split[i]
            cls = normalize_label(ex[label_field])
            if cls not in {"ai", "real"}:
                continue
            img = open_example_image(ex, image_field)
            if img is None:
                continue

            h = hash_img_bytes(img)
            if h in seen_exact:
                continue

            d = dhash_hex(img)
            near_dup = any(hamming_hex(d, prev) <= args.near_hamming for prev in seen_dhash_by_cls[cls][-1200:])
            if near_dup:
                continue

            seen_exact.add(h)
            seen_dhash_by_cls[cls].append(d)
            per_class_queue[cls].append((img, src))
            used += 1

            if used >= 140000:
                break

        print(f"loaded_source={src} unique_added={used}")

    for cls in ["ai", "real"]:
        random.shuffle(per_class_queue[cls])

    for split in ["train", "val", "test"]:
        for cls in ["ai", "real"]:
            need = targets[split][cls]
            idx = 0
            while counts[split][cls] < need and idx < len(per_class_queue[cls]):
                img, src = per_class_queue[cls][idx]
                idx += 1
                n = counts[split][cls]
                src_tag = src.split("/")[-1][:24].replace("-", "_")
                out_path = out / split / cls / f"source={src_tag}__{split}_{cls}_{n:06d}.jpg"
                save_img(img, out_path)
                counts[split][cls] += 1

            per_class_queue[cls] = per_class_queue[cls][idx:]

    # Add hard negatives to train split only
    hard_modes = ["jpeg35", "blur", "resize60", "sharpen", "screenshot"]
    for cls in ["ai", "real"]:
        base_files = sorted((out / "train" / cls).glob("*.jpg"))[: max(1, targets["train"][cls] // 2)]
        hn_count = 0
        for p in base_files:
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                continue
            mode = random.choice(hard_modes)
            aug = augment_hard_negative(img, mode)
            dst = out / "train" / cls / f"hardneg={mode}__{p.name}"
            save_img(aug, dst, quality=90)
            hn_count += 1
        print(f"hard_negatives_{cls}={hn_count}")

    for split in ["train", "val", "test"]:
        for cls in ["ai", "real"]:
            n = len(list((out / split / cls).glob("*.jpg")))
            print(f"{split}/{cls}={n}")

    print("dataset_ready", out)


if __name__ == "__main__":
    main()
