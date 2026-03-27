from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import datasets, transforms

from ai_image_detector.domain import DOMAIN_NAMES, classify_domain
from ai_image_detector.ensemble import EnsembleDetector, load_models, metadata_features_from_paths
from ai_image_detector.metrics import find_best_threshold, full_metric_report
from ai_image_detector.text_signals import analyze_text_signals


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit per-domain thresholds on validation split")
    ap.add_argument("--data", default="./data_best")
    ap.add_argument("--model", nargs="+", required=True)
    ap.add_argument("--ensemble-config", default="", help="Optional ensemble weights config")
    ap.add_argument("--out", default="./artifacts_ens/domain_config.json")
    ap.add_argument("--objective", choices=["f1", "balanced", "youden"], default="balanced")
    ap.add_argument("--min-samples-per-domain", type=int, default=80)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = load_models(args.model, device, ensemble_config=args.ensemble_config)
    model = EnsembleDetector(loaded.models, weights=loaded.weights, img_sizes=loaded.img_sizes).to(device)
    model.eval()

    ds = datasets.ImageFolder(Path(args.data) / "val")
    ai_idx = int(ds.class_to_idx["ai"])
    tf = transforms.Compose([transforms.Resize((loaded.img_size, loaded.img_size)), transforms.ToTensor()])

    probs: dict[str, list[float]] = {d: [] for d in DOMAIN_NAMES}
    labels: dict[str, list[int]] = {d: [] for d in DOMAIN_NAMES}
    all_probs: list[float] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for path, y in ds.samples:
            img = Image.open(path).convert("RGB")
            x = tf(img).unsqueeze(0).to(device)
            metadata_features = None
            if loaded.uses_metadata_features:
                metadata_features = metadata_features_from_paths([path], device=device, dtype=x.dtype)
            p = torch.sigmoid(model(x, metadata_features=metadata_features) / max(loaded.temperature, 1e-6)).item()
            label = 1 if int(y) == ai_idx else 0
            text_score = float(analyze_text_signals(img)["text_score"])
            domain = classify_domain(img, text_score=text_score)
            probs.setdefault(domain, []).append(float(p))
            labels.setdefault(domain, []).append(label)
            all_probs.append(float(p))
            all_labels.append(label)

    base_threshold, base_score, _ = find_best_threshold(np.array(all_probs), np.array(all_labels), objective=args.objective)
    base_report = full_metric_report(np.array(all_probs), np.array(all_labels), threshold=base_threshold)

    thresholds: dict[str, float] = {}
    per_domain: dict[str, dict[str, float | int]] = {}
    for domain, arr in probs.items():
        y = labels.get(domain, [])
        if len(arr) < args.min_samples_per_domain or len(set(y)) < 2:
            thresholds[domain] = float(base_threshold)
            per_domain[domain] = {"n": len(arr), "threshold": float(base_threshold), "fallback": 1}
            continue
        th, score, metrics = find_best_threshold(np.array(arr), np.array(y), objective=args.objective)
        thresholds[domain] = float(th)
        per_domain[domain] = {
            "n": len(arr),
            "threshold": float(th),
            "objective_score": float(score),
            "f1": float(metrics.get("f1", 0.0)),
            "acc": float(metrics.get("accuracy", 0.0)),
        }

    out = {
        "base_threshold": float(base_threshold),
        "objective": args.objective,
        "base_report": base_report,
        "thresholds": thresholds,
        "per_domain": per_domain,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()
