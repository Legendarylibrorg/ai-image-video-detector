from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision import datasets

from .data import make_eval_transform, make_jailed_rgb_loader
from .io_limits import jpeg_roundtrip_rgb
from .ensemble import EnsembleDetector, load_models, metadata_features_from_paths
from .metrics import full_metric_report
from .utils import write_json_dict
from .runtime import training_device


def _resize_roundtrip(img: Image.Image, scale: float) -> Image.Image:
    w, h = img.size
    nw, nh = max(16, int(w * scale)), max(16, int(h * scale))
    shrunk = img.resize((nw, nh), Image.BILINEAR)
    return shrunk.resize((w, h), Image.BILINEAR)


def _variants(img: Image.Image) -> dict[str, Image.Image]:
    return {
        "clean": img,
        "jpeg_q60": jpeg_roundtrip_rgb(img, 60),
        "jpeg_q35": jpeg_roundtrip_rgb(img, 35),
        "blur": img.filter(ImageFilter.GaussianBlur(radius=1.2)),
        "resize_0.6": _resize_roundtrip(img, scale=0.6),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Robustness evaluation across perturbations")
    ap.add_argument("--data", required=True, help="Dataset root containing val/ai and val/real")
    ap.add_argument("--model", nargs="+", required=True)
    ap.add_argument("--ensemble-config", default="", help="Optional JSON with learned ensemble weights/threshold")
    ap.add_argument("--max-images", type=int, default=400)
    ap.add_argument("--out", default="./artifacts/robust_eval.json")
    args = ap.parse_args()

    val_dir = (Path(args.data) / "val").resolve()
    val_loader_fn = make_jailed_rgb_loader(val_dir)
    ds = datasets.ImageFolder(val_dir, loader=val_loader_fn)
    if "ai" not in ds.class_to_idx:
        raise ValueError(f"expected class 'ai' in {ds.class_to_idx}")
    ai_idx = int(ds.class_to_idx["ai"])

    device = training_device()
    loaded = load_models(args.model, device, ensemble_config=args.ensemble_config)
    model = EnsembleDetector(loaded.models, weights=loaded.weights, img_sizes=loaded.img_sizes).to(device)
    model.eval()

    tf = make_eval_transform(loaded.img_size)

    buckets: dict[str, list[tuple[float, int]]] = {}
    n = min(len(ds), args.max_images)
    batch_size = 32
    sample_paths = ds.samples[:n]
    variant_names = tuple(_variants(Image.new("RGB", (16, 16))).keys())

    with torch.no_grad():
        for start in range(0, len(sample_paths), batch_size):
            batch_samples = sample_paths[start : start + batch_size]
            batch_paths = [path for path, _ in batch_samples]
            batch_labels = [1 if int(y) == ai_idx else 0 for _, y in batch_samples]
            variant_batches: dict[str, list[torch.Tensor]] = {name: [] for name in variant_names}
            for path, _ in batch_samples:
                img = val_loader_fn(path)
                for name, vimg in _variants(img).items():
                    variant_batches[name].append(tf(vimg))

            metadata_features = None
            if loaded.uses_metadata_features:
                metadata_features = metadata_features_from_paths(batch_paths, device=device)

            for name in variant_names:
                x = torch.stack(variant_batches[name], dim=0).to(device, non_blocking=True)
                if x.device.type == "cuda":
                    x = x.contiguous(memory_format=torch.channels_last)
                batch_metadata = None
                if metadata_features is not None:
                    batch_metadata = metadata_features.to(device=device, dtype=x.dtype)
                probs = torch.sigmoid(model(x, metadata_features=batch_metadata) / max(loaded.temperature, 1e-6))
                buckets.setdefault(name, []).extend(
                    zip(probs.detach().cpu().tolist(), batch_labels)
                )

    report = {}
    for name, pairs in buckets.items():
        probs = np.array([p for p, _ in pairs], dtype=np.float64)
        labels = np.array([y for _, y in pairs], dtype=np.float64)
        report[name] = full_metric_report(probs, labels, threshold=loaded.threshold)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_dict(out_path, report)
    print(json.dumps(report, indent=2))
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()
