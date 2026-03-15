from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import torch
from PIL import Image
from torchvision import datasets, transforms

from ai_image_detector.ensemble import EnsembleDetector, load_models


def main():
    ap = argparse.ArgumentParser(description="Mine hard negatives from train split via ensemble uncertainty")
    ap.add_argument("--data", default="./data_best")
    ap.add_argument("--model", nargs="+", required=True)
    ap.add_argument("--out", default="./data_hard")
    ap.add_argument("--top-k", type=int, default=3000)
    args = ap.parse_args()

    root = Path(args.data)
    train_dir = root / "train"
    ds = datasets.ImageFolder(train_dir)
    if "ai" not in ds.class_to_idx:
        raise ValueError("expected ai class")
    ai_idx = int(ds.class_to_idx["ai"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = load_models(args.model, device)
    model = EnsembleDetector(loaded.models).to(device)
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((loaded.img_size, loaded.img_size)),
        transforms.ToTensor(),
    ])

    scored = []
    with torch.no_grad():
        for path, y in ds.samples:
            img = Image.open(path).convert("RGB")
            x = tf(img).unsqueeze(0).to(device)
            p = torch.sigmoid(model(x) / max(loaded.temperature, 1e-6)).item()
            target = 1 if int(y) == ai_idx else 0
            loss_like = abs(target - p)
            margin = abs(p - loaded.threshold)
            hard = (1.0 - margin) + loss_like
            scored.append((hard, path, target, p))

    scored.sort(key=lambda t: t[0], reverse=True)
    picked = scored[: args.top_k]

    out = Path(args.out)
    for cls in ["ai", "real"]:
        (out / cls).mkdir(parents=True, exist_ok=True)

    for idx, (_, p, target, prob) in enumerate(picked):
        cls = "ai" if target == 1 else "real"
        dst = out / cls / f"hard_{idx:05d}_p{prob:.3f}_{Path(p).name}"
        shutil.copy2(p, dst)

    print(f"mined={len(picked)} saved_to={out}")


if __name__ == "__main__":
    main()
