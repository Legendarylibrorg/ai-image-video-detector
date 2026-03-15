from __future__ import annotations

import argparse
from pathlib import Path
import random

from datasets import load_dataset
from PIL import Image


def to_class(label_value):
    s = str(label_value).strip().lower()
    if s.isdigit():
        return "ai" if int(s) == 1 else "real"
    if any(k in s for k in ["ai", "fake", "generated", "synthetic"]):
        return "ai"
    if any(k in s for k in ["real", "human", "natural", "authentic"]):
        return "real"
    return None


def main():
    ap = argparse.ArgumentParser(description="Build a large ready-to-train ai/real image dataset")
    ap.add_argument("--dataset", default="Hemg/AI-Generated-vs-Real-Images-Datasets")
    ap.add_argument("--out", default="data")
    ap.add_argument("--train-per-class", type=int, default=20000)
    ap.add_argument("--val-per-class", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quality", type=int, default=92)
    args = ap.parse_args()

    random.seed(args.seed)
    out = Path(args.out)
    for split in ["train", "val"]:
        for cls in ["ai", "real"]:
            d = out / split / cls
            d.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset)
    split_name = "train" if "train" in ds else list(ds.keys())[0]
    split = ds[split_name]

    cols = set(split.column_names)
    image_field = "image" if "image" in cols else next((c for c in split.column_names if "image" in c.lower()), None)
    label_field = "label" if "label" in cols else next((c for c in split.column_names if c.lower() in {"class", "target", "labels"}), None)
    if not image_field or not label_field:
        raise RuntimeError(f"Could not infer image/label columns from: {split.column_names}")

    n_total = len(split)
    indices = list(range(n_total))
    random.shuffle(indices)

    need = {
        "train": {"ai": args.train_per_class, "real": args.train_per_class},
        "val": {"ai": args.val_per_class, "real": args.val_per_class},
    }
    have = {
        "train": {"ai": 0, "real": 0},
        "val": {"ai": 0, "real": 0},
    }

    for idx in indices:
        ex = split[idx]
        cls = to_class(ex[label_field])
        if cls not in {"ai", "real"}:
            continue

        target_split = None
        if have["train"][cls] < need["train"][cls]:
            target_split = "train"
        elif have["val"][cls] < need["val"][cls]:
            target_split = "val"
        else:
            continue

        img = ex[image_field]
        if not isinstance(img, Image.Image):
            try:
                img = Image.fromarray(img)
            except Exception:
                continue

        c = have[target_split][cls]
        out_path = out / target_split / cls / f"{target_split}_{cls}_{c:06d}.jpg"
        img.convert("RGB").save(out_path, quality=args.quality)
        have[target_split][cls] += 1

        if (
            have["train"]["ai"] >= need["train"]["ai"]
            and have["train"]["real"] >= need["train"]["real"]
            and have["val"]["ai"] >= need["val"]["ai"]
            and have["val"]["real"] >= need["val"]["real"]
        ):
            break

    print("DONE")
    print(have)


if __name__ == "__main__":
    main()
