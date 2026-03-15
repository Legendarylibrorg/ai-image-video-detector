from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision import datasets, transforms

from .ensemble import EnsembleDetector, load_models
from .metrics import full_metric_report


def _jpeg_roundtrip(img: Image.Image, quality: int) -> Image.Image:
    b = io.BytesIO()
    img.save(b, format="JPEG", quality=quality)
    b.seek(0)
    return Image.open(b).convert("RGB")


def _resize_roundtrip(img: Image.Image, scale: float) -> Image.Image:
    w, h = img.size
    nw, nh = max(16, int(w * scale)), max(16, int(h * scale))
    shrunk = img.resize((nw, nh), Image.BILINEAR)
    return shrunk.resize((w, h), Image.BILINEAR)


def _variants(img: Image.Image) -> dict[str, Image.Image]:
    return {
        "clean": img,
        "jpeg_q60": _jpeg_roundtrip(img, quality=60),
        "jpeg_q35": _jpeg_roundtrip(img, quality=35),
        "blur": img.filter(ImageFilter.GaussianBlur(radius=1.2)),
        "resize_0.6": _resize_roundtrip(img, scale=0.6),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Robustness evaluation across perturbations")
    ap.add_argument("--data", required=True, help="Dataset root containing val/ai and val/real")
    ap.add_argument("--model", nargs="+", required=True)
    ap.add_argument("--max-images", type=int, default=400)
    ap.add_argument("--out", default="./artifacts/robust_eval.json")
    args = ap.parse_args()

    val_dir = Path(args.data) / "val"
    ds = datasets.ImageFolder(val_dir)
    if "ai" not in ds.class_to_idx:
        raise ValueError(f"expected class 'ai' in {ds.class_to_idx}")
    ai_idx = int(ds.class_to_idx["ai"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = load_models(args.model, device)
    model = EnsembleDetector(loaded.models).to(device)
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((loaded.img_size, loaded.img_size)),
        transforms.ToTensor(),
    ])

    buckets: dict[str, list[tuple[float, int]]] = {}

    n = min(len(ds), args.max_images)
    for i in range(n):
        p, y = ds.samples[i]
        y_ai = 1 if int(y) == ai_idx else 0
        img = Image.open(p).convert("RGB")
        for name, vimg in _variants(img).items():
            x = tf(vimg).unsqueeze(0).to(device)
            with torch.no_grad():
                prob = torch.sigmoid(model(x)).item()
            buckets.setdefault(name, []).append((prob, int(y_ai)))

    report = {}
    for name, pairs in buckets.items():
        probs = np.array([p for p, _ in pairs], dtype=np.float64)
        labels = np.array([y for _, y in pairs], dtype=np.float64)
        report[name] = full_metric_report(probs, labels, threshold=loaded.threshold)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()
