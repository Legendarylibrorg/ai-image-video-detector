from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from ai_image_detector.data import MetadataImageFolder, build_loader_kwargs, make_eval_transform, unpack_image_batch
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = load_models(args.model, device, ensemble_config=args.ensemble_config)
    model = EnsembleDetector(loaded.models, weights=loaded.weights, img_sizes=loaded.img_sizes).to(device)
    model.eval()

    test_dir = Path(args.data) / "test"
    dataset_cls = MetadataImageFolder if loaded.uses_metadata_features else datasets.ImageFolder
    ds = dataset_cls(
        test_dir,
        transform=make_eval_transform(loaded.img_size),
    )
    if "ai" not in ds.class_to_idx:
        raise ValueError(f"Expected class 'ai' in {ds.class_to_idx}")
    ai_idx = int(ds.class_to_idx["ai"])
    dl = DataLoader(
        ds,
        batch_size=64,
        shuffle=False,
        **build_loader_kwargs(num_workers=4),
    )

    probs, labels = [], []
    with torch.no_grad():
        for batch in dl:
            x, metadata_features, y = unpack_image_batch(batch)
            x = x.to(device, non_blocking=True)
            if device.type == "cuda":
                x = x.contiguous(memory_format=torch.channels_last)
            if metadata_features is not None:
                metadata_features = metadata_features.to(device=device, dtype=x.dtype, non_blocking=True)
            views = [x]
            if args.tta >= 2:
                views.append(torch.flip(x, dims=[3]))  # hflip
            if args.tta >= 3:
                views.append(torch.flip(x, dims=[2]))  # vflip
            logits = torch.stack([model(v, metadata_features=metadata_features) for v in views], dim=0).mean(dim=0)
            batch_probs = torch.sigmoid(logits / max(loaded.temperature, 1e-6))
            probs.extend(batch_probs.detach().cpu().tolist())
            labels.extend((y == ai_idx).int().tolist())

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
