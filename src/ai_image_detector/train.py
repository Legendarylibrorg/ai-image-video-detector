from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .data import make_loaders
from .metrics import find_best_threshold, fit_temperature, full_metric_report, sigmoid
from .model import build_model


def _path_tags(path: str) -> dict[str, str]:
    stem = Path(path).stem.lower()
    tags: dict[str, str] = {}
    for part in stem.split("__"):
        if "=" in part:
            k, v = part.split("=", 1)
            if k in {"source", "generator", "model", "set"}:
                tags["source"] = v
            if k in {"camera", "device", "phone", "lens"}:
                tags["camera"] = v
    return tags


def _group_report(
    probs: np.ndarray,
    labels: np.ndarray,
    paths: list[str],
    threshold: float,
) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[float]]] = {"source": {}, "camera": {}}

    for p, y, path in zip(probs.tolist(), labels.tolist(), paths):
        tags = _path_tags(path)
        for key in ("source", "camera"):
            if key not in tags:
                continue
            bucket = grouped[key].setdefault(tags[key], {"probs": [], "labels": []})
            bucket["probs"].append(float(p))
            bucket["labels"].append(float(y))

    out: dict[str, Any] = {"source": {}, "camera": {}}
    for key in ("source", "camera"):
        for name, data in grouped[key].items():
            if len(data["labels"]) < 8:
                continue
            out[key][name] = full_metric_report(data["probs"], data["labels"], threshold)
            out[key][name]["samples"] = len(data["labels"])
    return out


class FocalBCE(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = torch.where(targets > 0.5, probs, 1.0 - probs)
        focal = (1.0 - pt).pow(self.gamma)
        return (focal * bce).mean()


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        msd = model.state_dict()
        for k, v in self.shadow.state_dict().items():
            if k in msd and torch.is_floating_point(v):
                v.copy_(v * self.decay + msd[k] * (1.0 - self.decay))


def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    ai_idx: int,
    val_samples: list[tuple[str, int]],
    loss_fn: nn.Module,
):
    model.eval()

    loss_sum = 0.0
    total = 0
    probs_all: list[float] = []
    labels_all: list[float] = []
    logits_all: list[float] = []
    paths_all: list[str] = []

    offset = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if device.type == "cuda":
                x = x.to(memory_format=torch.channels_last)
            target = (y == ai_idx).float()

            logits = model(x)
            loss = loss_fn(logits, target)
            probs = torch.sigmoid(logits)

            bs = y.shape[0]
            batch_paths = [val_samples[offset + i][0] for i in range(bs)]
            offset += bs

            probs_all.extend(probs.detach().cpu().tolist())
            labels_all.extend(target.detach().cpu().tolist())
            logits_all.extend(logits.detach().cpu().tolist())
            paths_all.extend(batch_paths)

            loss_sum += float(loss.item()) * bs
            total += bs

    avg_loss = loss_sum / max(total, 1)
    return (
        avg_loss,
        np.asarray(logits_all, dtype=np.float64),
        np.asarray(probs_all, dtype=np.float64),
        np.asarray(labels_all, dtype=np.float64),
        paths_all,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="./artifacts")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--loss", choices=["bce", "focal"], default="focal")
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--backbone", choices=["tiny", "effb0", "effb2"], default="tiny")
    ap.add_argument("--no-pretrained-backbone", action="store_true")
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--compile", action="store_true", default=True)
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    ap.add_argument("--threshold-objective", choices=["f1", "balanced", "youden"], default="balanced")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    if args.num_workers <= 0:
        cpu = os.cpu_count() or 8
        args.num_workers = min(12, max(4, cpu // 2))
    train_loader, val_loader, classes, class_to_idx, val_samples = make_loaders(
        args.data,
        args.img_size,
        args.batch_size,
        num_workers=args.num_workers,
    )

    if set(classes) != {"ai", "real"}:
        raise ValueError(f"Expected classes exactly ai/real, got {classes}")
    ai_idx = int(class_to_idx["ai"])

    model = build_model(
        backbone=args.backbone,
        pretrained_backbone=(not args.no_pretrained_backbone),
    ).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    if args.compile:
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as exc:
            print(f"compile_disabled reason={exc}")
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=max(args.epochs, 1), eta_min=args.lr * 0.05)
    loss_fn: nn.Module = nn.BCEWithLogitsLoss() if args.loss == "bce" else FocalBCE(gamma=args.focal_gamma)

    ema = EMA(model, decay=args.ema_decay)
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    grad_accum = max(1, int(args.grad_accum))

    best_auc = -1.0
    model_id = f"advanced-ai-detector-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        step_idx = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if device.type == "cuda":
                x = x.to(memory_format=torch.channels_last)
            target = (y == ai_idx).float()

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x)
                loss = loss_fn(logits, target) / grad_accum
            scaler.scale(loss).backward()
            step_idx += 1

            if step_idx % grad_accum == 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                ema.update(model)

        if step_idx % grad_accum != 0:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            ema.update(model)

        sched.step()

        val_loss, val_logits, _, val_labels, val_paths = evaluate(
            ema.shadow,
            val_loader,
            device,
            ai_idx,
            val_samples,
            loss_fn,
        )

        temperature, temp_nll = fit_temperature(val_logits, val_labels)
        val_probs = sigmoid(val_logits / max(temperature, 1e-6))
        threshold, score, tuned_metrics = find_best_threshold(
            val_probs,
            val_labels,
            objective=args.threshold_objective,
        )

        report = full_metric_report(val_probs, val_labels, threshold)
        report["val_loss"] = float(val_loss)
        report["temperature"] = float(temperature)
        report["temperature_nll"] = float(temp_nll)
        report["threshold_objective"] = args.threshold_objective
        report["threshold_objective_score"] = float(score)
        report["tuned_metrics"] = tuned_metrics
        report["lr"] = float(opt.param_groups[0]["lr"])

        grouped = _group_report(val_probs, val_labels, val_paths, threshold)

        print(
            "epoch={} val_loss={:.5f} auc={:.4f} f1={:.4f} ece={:.4f} brier={:.4f} th={:.3f} temp={:.3f} lr={:.6f}".format(
                epoch,
                report["val_loss"],
                report["auc"],
                report["f1"],
                report["ece"],
                report["brier"],
                threshold,
                temperature,
                report["lr"],
            )
        )

        last = {
            "epoch": epoch,
            "metrics": report,
            "group_metrics": grouped,
        }
        (out / "last_metrics.json").write_text(json.dumps(last, indent=2), encoding="utf-8")

        if report["auc"] > best_auc:
            best_auc = float(report["auc"])
            ckpt = {
                "state_dict": ema.shadow.state_dict(),
                "img_size": args.img_size,
                "threshold": float(threshold),
                "temperature": float(temperature),
                "model_id": model_id,
                "metrics": report,
                "classes": classes,
                "backbone": args.backbone,
            }
            torch.save(ckpt, out / "best.pt")
            (out / "best_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            (out / "best_group_metrics.json").write_text(json.dumps(grouped, indent=2), encoding="utf-8")
            (out / "calibration.json").write_text(
                json.dumps(
                    {
                        "threshold": float(threshold),
                        "temperature": float(temperature),
                        "objective": args.threshold_objective,
                        "metrics": report,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    print(f"saved best model to {out / 'best.pt'} with best_auc={best_auc:.4f}")


if __name__ == "__main__":
    main()
