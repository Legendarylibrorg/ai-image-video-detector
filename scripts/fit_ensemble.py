from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from ai_image_detector.ensemble import load_models, stack_model_logits
from ai_image_detector.metrics import find_best_threshold, full_metric_report, roc_auc


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit weighted ensemble on validation split")
    ap.add_argument("--data", default="./data_best", help="Dataset root with val/{ai,real}")
    ap.add_argument("--model", nargs="+", required=True)
    ap.add_argument("--out", default="./artifacts_ens/ensemble_config.json")
    ap.add_argument("--objective", choices=["f1", "balanced", "youden"], default="balanced")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--l2", type=float, default=0.001)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--max-val-images", type=int, default=0, help="0 means all")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    _seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = load_models(args.model, device)
    for m in loaded.models:
        m.eval()

    val_dir = Path(args.data) / "val"
    ds = datasets.ImageFolder(
        val_dir,
        transform=transforms.Compose(
            [
                transforms.Resize((loaded.img_size, loaded.img_size)),
                transforms.ToTensor(),
            ]
        ),
    )
    if "ai" not in ds.class_to_idx:
        raise ValueError(f"Expected class 'ai' in val split, got {ds.class_to_idx}")
    ai_idx = int(ds.class_to_idx["ai"])

    if args.max_val_images > 0 and len(ds) > args.max_val_images:
        idxs = list(range(len(ds)))
        random.Random(args.seed).shuffle(idxs)
        ds = Subset(ds, idxs[: args.max_val_images])

    dl = DataLoader(
        ds,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=True,
    )

    logits_all: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y_ai = (y == ai_idx).float().to(device, non_blocking=True)
            model_logits = stack_model_logits(loaded.models, loaded.img_sizes, x).transpose(0, 1)
            logits_all.append(model_logits.detach().cpu())
            labels_all.append(y_ai.detach().cpu())

    if not logits_all:
        raise RuntimeError("No validation samples found for ensemble fitting")

    logits = torch.cat(logits_all, dim=0)
    labels = torch.cat(labels_all, dim=0)
    n_samples, n_models = logits.shape

    fit_device = device
    logits_t = logits.to(fit_device)
    labels_t = labels.to(fit_device)

    raw_w = torch.zeros(n_models, device=fit_device, requires_grad=True)
    log_temp = torch.zeros(1, device=fit_device, requires_grad=True)
    opt = Adam([raw_w, log_temp], lr=float(args.lr))

    best_loss = float("inf")
    best_w = torch.softmax(raw_w.detach(), dim=0)
    best_temp = torch.tensor([1.0], device=fit_device)

    for _ in range(max(1, int(args.steps))):
        opt.zero_grad(set_to_none=True)
        w = torch.softmax(raw_w, dim=0)
        temp = torch.exp(log_temp).clamp(min=0.2, max=5.0)
        ensemble_logit = (logits_t * w.view(1, -1)).sum(dim=1)
        probs = torch.sigmoid(ensemble_logit / temp)
        loss = F.binary_cross_entropy(probs.clamp(1e-6, 1.0 - 1e-6), labels_t)
        loss = loss + float(args.l2) * torch.sum(w * w)
        loss.backward()
        opt.step()

        cur = float(loss.detach().item())
        if cur < best_loss:
            best_loss = cur
            best_w = w.detach().clone()
            best_temp = temp.detach().clone()

    best_w_np = best_w.detach().cpu().numpy()
    best_temp_f = float(best_temp.detach().cpu().item())

    ensemble_logits_np = (logits.numpy() * best_w_np.reshape(1, -1)).sum(axis=1)
    ensemble_probs_np = 1.0 / (1.0 + np.exp(-np.clip(ensemble_logits_np / max(best_temp_f, 1e-6), -40.0, 40.0)))
    labels_np = labels.numpy()

    threshold, score, tuned = find_best_threshold(ensemble_probs_np, labels_np, objective=args.objective)
    report = full_metric_report(ensemble_probs_np, labels_np, threshold=threshold)
    report["threshold_objective"] = args.objective
    report["threshold_objective_score"] = float(score)
    report["tuned_metrics"] = tuned
    report["n_val"] = int(n_samples)

    per_model = []
    for i in range(n_models):
        mtemp = max(float(loaded.model_temperatures[i]), 1e-6)
        p = 1.0 / (1.0 + np.exp(-np.clip(logits.numpy()[:, i] / mtemp, -40.0, 40.0)))
        per_model.append(
            {
                "model_id": loaded.model_ids[i],
                "auc": float(roc_auc(p, labels_np)),
            }
        )

    out_cfg = {
        "model_paths": [str(Path(p).resolve()) for p in args.model],
        "model_ids": loaded.model_ids,
        "weights": [float(x) for x in best_w_np.tolist()],
        "temperature": best_temp_f,
        "threshold": float(threshold),
        "fit": {
            "objective": args.objective,
            "loss": float(best_loss),
            "steps": int(args.steps),
            "lr": float(args.lr),
            "l2": float(args.l2),
            "report": report,
            "per_model": per_model,
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(out_cfg, indent=2), encoding="utf-8")
    print(json.dumps(out_cfg, indent=2))
    print(f"saved={out}")


if __name__ == "__main__":
    main()
