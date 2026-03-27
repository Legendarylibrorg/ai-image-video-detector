from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from ai_image_detector.data import MetadataImageFolder, build_loader_kwargs, make_eval_transform, unpack_image_batch
from ai_image_detector.ensemble import EnsembleDetector, load_models


def main():
    ap = argparse.ArgumentParser(description="Mine hard negatives from train split via ensemble uncertainty")
    ap.add_argument("--data", default="./data_best")
    ap.add_argument("--model", nargs="+", required=True)
    ap.add_argument("--ensemble-config", default="", help="Optional JSON with learned ensemble weights/threshold")
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
    loaded = load_models(args.model, device, ensemble_config=args.ensemble_config)
    model = EnsembleDetector(loaded.models, weights=loaded.weights, img_sizes=loaded.img_sizes).to(device)
    model.eval()

    dataset_cls = MetadataImageFolder if loaded.uses_metadata_features else datasets.ImageFolder
    ds = dataset_cls(
        train_dir,
        transform=make_eval_transform(loaded.img_size),
    )
    dl = DataLoader(
        ds,
        batch_size=64,
        shuffle=False,
        **build_loader_kwargs(num_workers=4),
    )

    scored = []
    offset = 0
    with torch.no_grad():
        for batch in dl:
            x, metadata_features, y = unpack_image_batch(batch)
            x = x.to(device, non_blocking=True)
            if device.type == "cuda":
                x = x.contiguous(memory_format=torch.channels_last)
            if metadata_features is not None:
                metadata_features = metadata_features.to(device=device, dtype=x.dtype, non_blocking=True)
            probs = torch.sigmoid(model(x, metadata_features=metadata_features) / max(loaded.temperature, 1e-6))
            batch_paths = [path for path, _ in ds.samples[offset : offset + x.shape[0]]]
            offset += x.shape[0]
            batch_probs = probs.detach().cpu().tolist()
            batch_targets = (y == ai_idx).int().tolist()
            for path, target, prob in zip(batch_paths, batch_targets, batch_probs):
                loss_like = abs(int(target) - float(prob))
                margin = abs(float(prob) - loaded.threshold)
                hard = (1.0 - margin) + loss_like
                scored.append((hard, path, int(target), float(prob)))

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
