from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets, transforms

from ai_image_detector.data import MetadataImageFolder
from ai_image_detector.ensemble import EnsembleDetector, load_models
from ai_image_detector.metrics import full_metric_report


def main():
    ap = argparse.ArgumentParser(description="Evaluate ensemble on test split")
    ap.add_argument("--data", default="./data_best")
    ap.add_argument("--model", nargs="+", required=True)
    ap.add_argument("--ensemble-config", default="", help="Optional JSON with learned ensemble weights/threshold")
    ap.add_argument("--tta", type=int, default=2, help="Test-time augmentation views (1=none, 2=+hflip, 3=+vflip)")
    ap.add_argument("--out", default="./artifacts_ens/test_metrics.json")
    args = ap.parse_args()

    test_dir = Path(args.data) / "test"
    dataset_cls = MetadataImageFolder if loaded.uses_metadata_features else datasets.ImageFolder
    ds = dataset_cls(test_dir)
    if "ai" not in ds.class_to_idx:
        raise ValueError(f"Expected class 'ai' in {ds.class_to_idx}")
    ai_idx = int(ds.class_to_idx["ai"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = load_models(args.model, device, ensemble_config=args.ensemble_config)
    model = EnsembleDetector(loaded.models, weights=loaded.weights, img_sizes=loaded.img_sizes).to(device)
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((loaded.img_size, loaded.img_size)),
        transforms.ToTensor(),
    ])

    probs, labels = [], []
    with torch.no_grad():
        for index in range(len(ds)):
            batch = ds[index]
            metadata_features = None
            if len(batch) == 3:
                img, metadata_features, y = batch
            else:
                img, y = batch
            x = tf(img).unsqueeze(0).to(device)
            if metadata_features is not None:
                metadata_features = metadata_features.unsqueeze(0).to(device=device, dtype=x.dtype)
            views = [x]
            if args.tta >= 2:
                views.append(torch.flip(x, dims=[3]))  # hflip
            if args.tta >= 3:
                views.append(torch.flip(x, dims=[2]))  # vflip
            logits = torch.stack([model(v, metadata_features=metadata_features) for v in views], dim=0).mean(dim=0)
            prob = torch.sigmoid(logits / max(loaded.temperature, 1e-6)).item()
            probs.append(prob)
            labels.append(1 if int(y) == ai_idx else 0)

    report = full_metric_report(np.array(probs), np.array(labels), threshold=loaded.threshold)
    report["n_samples"] = len(labels)
    report["model_ids"] = loaded.model_ids
    report["ensemble_weights"] = [float(w) for w in loaded.weights]
    report["ensemble_config"] = args.ensemble_config or None
    report["tta_views"] = int(max(args.tta, 1))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"saved={out}")


if __name__ == "__main__":
    main()
