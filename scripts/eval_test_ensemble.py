from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import datasets, transforms

from ai_image_detector.ensemble import EnsembleDetector, load_models
from ai_image_detector.metrics import full_metric_report


def main():
    ap = argparse.ArgumentParser(description="Evaluate ensemble on test split")
    ap.add_argument("--data", default="./data_best")
    ap.add_argument("--model", nargs="+", required=True)
    ap.add_argument("--out", default="./artifacts_ens/test_metrics.json")
    args = ap.parse_args()

    test_dir = Path(args.data) / "test"
    ds = datasets.ImageFolder(test_dir)
    if "ai" not in ds.class_to_idx:
        raise ValueError(f"Expected class 'ai' in {ds.class_to_idx}")
    ai_idx = int(ds.class_to_idx["ai"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = load_models(args.model, device)
    model = EnsembleDetector(loaded.models).to(device)
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((loaded.img_size, loaded.img_size)),
        transforms.ToTensor(),
    ])

    probs, labels = [], []
    with torch.no_grad():
        for p, y in ds.samples:
            img = Image.open(p).convert("RGB")
            x = tf(img).unsqueeze(0).to(device)
            prob = torch.sigmoid(model(x) / max(loaded.temperature, 1e-6)).item()
            probs.append(prob)
            labels.append(1 if int(y) == ai_idx else 0)

    report = full_metric_report(np.array(probs), np.array(labels), threshold=loaded.threshold)
    report["n_samples"] = len(labels)
    report["model_ids"] = loaded.model_ids

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"saved={out}")


if __name__ == "__main__":
    main()
