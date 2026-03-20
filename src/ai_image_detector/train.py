from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from torchvision import datasets, transforms

from .checkpoints import load_checkpoint, resolve_checkpoint_path, save_safetensors_checkpoint
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


class BinaryClassificationLoss(nn.Module):
    def __init__(
        self,
        kind: str = "ce",
        gamma: float = 2.0,
        real_weight: float = 1.0,
        ai_weight: float = 1.0,
    ):
        super().__init__()
        self.kind = kind
        self.gamma = gamma
        self.register_buffer("real_weight", torch.tensor(float(real_weight), dtype=torch.float32))
        self.register_buffer("ai_weight", torch.tensor(float(ai_weight), dtype=torch.float32))
        self.register_buffer(
            "class_weights",
            torch.tensor([float(real_weight), float(ai_weight)], dtype=torch.float32),
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        if self.kind == "ce":
            logits_2c = torch.stack((torch.zeros_like(logits), logits), dim=1)
            return F.cross_entropy(logits_2c, targets.long(), weight=self.class_weights)

        sample_weights = torch.where(targets > 0.5, self.ai_weight, self.real_weight)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        if self.kind == "bce":
            return (bce * sample_weights).sum() / sample_weights.sum().clamp(min=1e-6)

        probs = torch.sigmoid(logits)
        pt = torch.where(targets > 0.5, probs, 1.0 - probs)
        focal = (1.0 - pt).pow(self.gamma)
        weighted = focal * bce * sample_weights
        return weighted.sum() / sample_weights.sum().clamp(min=1e-6)


def _binary_targets(y: torch.Tensor, ai_idx: int) -> torch.Tensor:
    return (y == ai_idx).float()


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


def _eval_metrics_from_probs(probs: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, Any]:
    y_true = labels.astype(np.int64)
    y_pred = (probs >= threshold).astype(np.int64)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    try:
        auc = float(roc_auc_score(y_true, probs))
    except Exception:
        auc = 0.5
    try:
        pr_auc = float(average_precision_score(y_true, probs))
    except Exception:
        pr_auc = 0.5
    try:
        bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    except Exception:
        bal_acc = 0.0
    tn, fp, fn, tp = cm.ravel()
    pred_unique = np.unique(y_pred)
    return {
        "threshold": float(threshold),
        "auc": auc,
        "pr_auc": pr_auc,
        "balanced_accuracy": bal_acc,
        "confusion_matrix": cm.tolist(),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "precision_real": float(prec[0]),
        "recall_real": float(rec[0]),
        "f1_real": float(f1[0]),
        "support_real": int(support[0]),
        "precision_ai": float(prec[1]),
        "recall_ai": float(rec[1]),
        "f1_ai": float(f1[1]),
        "support_ai": int(support[1]),
        "predicts_single_class": bool(len(pred_unique) == 1),
        "predicted_classes": pred_unique.astype(int).tolist(),
    }


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
            y_bin = _binary_targets(y, ai_idx)
            if device.type == "cuda":
                x = x.to(memory_format=torch.channels_last)
            logits = model(x)
            loss = loss_fn(logits, y_bin)
            probs = torch.sigmoid(logits)

            bs = y.shape[0]
            batch_paths = [val_samples[offset + i][0] for i in range(bs)]
            offset += bs

            probs_all.extend(probs.detach().cpu().tolist())
            labels_all.extend(y_bin.detach().cpu().tolist())
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


def _git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True)
        return out.strip()
    except Exception:
        return "unknown"


def _dataset_counts(root: Path) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        out[split] = {}
        for cls in ("ai", "real"):
            d = root / split / cls
            out[split][cls] = len(list(d.glob("*"))) if d.exists() else 0
    return out


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
    ap.add_argument("--loss", choices=["ce", "bce", "focal"], default="ce")
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--backbone", choices=["tiny", "effb0", "effb2"], default="tiny")
    ap.add_argument("--no-pretrained-backbone", action="store_true")
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--compile", action="store_true", default=True)
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    ap.add_argument("--threshold-objective", choices=["f1", "balanced", "youden"], default="balanced")
    ap.add_argument(
        "--decision-threshold",
        type=float,
        default=None,
        help="Optional fixed threshold for class=ai; otherwise tuned on validation using --threshold-objective",
    )
    ap.add_argument("--degenerate-patience", type=int, default=2, help="Fail training after N consecutive degenerate val epochs")
    ap.add_argument("--resume", default="", help="Path to training checkpoint (default: <out>/last.pt if present)")
    ap.add_argument("--save-every", type=int, default=1, help="Save epoch checkpoint every N epochs")
    ap.add_argument("--patience", type=int, default=0, help="Early stopping patience in epochs (0 disables)")
    ap.add_argument("--min-delta", type=float, default=0.0, help="Minimum AUC improvement to reset patience")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--deterministic", action="store_true", help="Enable deterministic behavior (slower)")
    ap.add_argument("--export-release", action="store_true", default=True)
    ap.add_argument("--no-export-release", dest="export_release", action="store_false")
    ap.add_argument("--save-safetensors", action="store_true", default=True)
    ap.add_argument("--no-save-safetensors", dest="save_safetensors", action="store_false")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if not args.deterministic:
            torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    if args.num_workers < 0:
        cpu = os.cpu_count() or 8
        args.num_workers = min(12, max(4, cpu // 2))
    train_loader, val_loader, classes, class_to_idx, val_samples, train_distribution, val_distribution, class_weight_map = make_loaders(
        args.data,
        args.img_size,
        args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"class_distribution train={train_distribution} val={val_distribution}")
    print(f"class_weights_inverse_freq={class_weight_map}")

    run_config = {
        "args": vars(args),
        "git_commit": _git_commit(),
        "dataset_counts": _dataset_counts(data_root),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out / "config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    if set(classes) != {"ai", "real"}:
        raise ValueError(f"Expected classes exactly ai/real, got {classes}")
    ai_idx = int(class_to_idx["ai"])
    real_weight = float(class_weight_map["real"])
    ai_weight = float(class_weight_map["ai"])

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
    loss_fn: nn.Module = BinaryClassificationLoss(
        kind=args.loss,
        gamma=args.focal_gamma,
        real_weight=real_weight,
        ai_weight=ai_weight,
    ).to(device)

    ema = EMA(model, decay=args.ema_decay)
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    grad_accum = max(1, int(args.grad_accum))

    best_auc = -1.0
    no_improve = 0
    degenerate_epochs = 0
    model_id = f"advanced-ai-detector-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    start_epoch = 1

    def _save_train_ckpt(path: Path, epoch: int) -> None:
        torch.save(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "ema_state_dict": ema.shadow.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "scaler": scaler.state_dict(),
                "best_auc": float(best_auc),
                "no_improve": int(no_improve),
                "degenerate_epochs": int(degenerate_epochs),
                "model_id": model_id,
                "args": vars(args),
                "classes": classes,
                "class_to_idx": class_to_idx,
                "backbone": args.backbone,
                "img_size": args.img_size,
            },
            path,
        )
        (out / "latest_checkpoint.txt").write_text(path.name, encoding="utf-8")

    resume_path = Path(args.resume) if args.resume else resolve_checkpoint_path(out / "last.pt")
    if resume_path.exists():
        ckpt = load_checkpoint(resume_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        ema.shadow.load_state_dict(ckpt.get("ema_state_dict", ckpt["state_dict"]))
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            sched.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        best_auc = float(ckpt.get("best_auc", best_auc))
        no_improve = int(ckpt.get("no_improve", no_improve))
        degenerate_epochs = int(ckpt.get("degenerate_epochs", degenerate_epochs))
        model_id = str(ckpt.get("model_id", model_id))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"resumed_from={resume_path} start_epoch={start_epoch} best_auc={best_auc:.4f}")

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            opt.zero_grad(set_to_none=True)
            step_idx = 0
            skipped_batches = 0
            for x, y in train_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                y_bin = _binary_targets(y, ai_idx)
                if device.type == "cuda":
                    x = x.to(memory_format=torch.channels_last)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(x)
                    loss = loss_fn(logits, y_bin) / grad_accum
                if not torch.isfinite(loss):
                    skipped_batches += 1
                    opt.zero_grad(set_to_none=True)
                    print(f"warn epoch={epoch} skipped_batch=non_finite_loss")
                    continue

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
            threshold_source = "fixed"
            threshold_score = None
            if args.decision_threshold is None:
                threshold, threshold_score, _ = find_best_threshold(
                    val_probs,
                    val_labels,
                    objective=args.threshold_objective,
                )
                threshold_source = "tuned"
            else:
                threshold = float(args.decision_threshold)
            report = _eval_metrics_from_probs(val_probs, val_labels, threshold)
            report["val_loss"] = float(val_loss)
            report["temperature"] = float(temperature)
            report["temperature_nll"] = float(temp_nll)
            report["composite_metrics"] = full_metric_report(val_probs, val_labels, threshold)
            report["threshold_source"] = threshold_source
            report["threshold_objective"] = args.threshold_objective
            if threshold_score is not None:
                report["threshold_objective_score"] = float(threshold_score)
            report["lr"] = float(opt.param_groups[0]["lr"])
            report["skipped_batches"] = skipped_batches

            grouped = _group_report(val_probs, val_labels, val_paths, threshold)
            if report["predicts_single_class"]:
                print(f"warning_single_class_predictions classes={report['predicted_classes']}")

            print(
                "epoch={} val_loss={:.5f} auc={:.4f} pr_auc={:.4f} bal_acc={:.4f} th={:.3f} temp={:.3f} precision_ai={:.4f} recall_ai={:.4f} f1_ai={:.4f} precision_real={:.4f} recall_real={:.4f} f1_real={:.4f} lr={:.6f}".format(
                    epoch,
                    report["val_loss"],
                    report["auc"],
                    report["pr_auc"],
                    report["balanced_accuracy"],
                    threshold,
                    temperature,
                    report["precision_ai"],
                    report["recall_ai"],
                    report["f1_ai"],
                    report["precision_real"],
                    report["recall_real"],
                    report["f1_real"],
                    report["lr"],
                )
            )
            degenerate = report["predicts_single_class"] or report["recall_ai"] <= 0.0 or report["recall_real"] <= 0.0
            if degenerate:
                degenerate_epochs += 1
                print(f"degenerate_validation epoch={epoch} streak={degenerate_epochs} limit={args.degenerate_patience}")
                if args.degenerate_patience > 0 and degenerate_epochs >= args.degenerate_patience:
                    raise RuntimeError("degenerate_validation_fail")
            else:
                degenerate_epochs = 0

            last = {
                "epoch": epoch,
                "metrics": report,
                "group_metrics": grouped,
            }
            (out / "last_metrics.json").write_text(json.dumps(last, indent=2), encoding="utf-8")
            with (out / "training_log.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"epoch": epoch, **report}) + "\n")

            _save_train_ckpt(out / "last.pt", epoch)
            if args.save_every > 0 and (epoch % args.save_every == 0):
                _save_train_ckpt(out / f"epoch_{epoch:03d}.pt", epoch)

            if report["auc"] > (best_auc + args.min_delta):
                best_auc = float(report["auc"])
                no_improve = 0
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
                if args.save_safetensors:
                    save_safetensors_checkpoint(out / "best.safetensors", ckpt)
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
            else:
                no_improve += 1
                if args.patience > 0 and no_improve >= args.patience:
                    print(f"early_stopping epoch={epoch} no_improve={no_improve} patience={args.patience}")
                    break
    except KeyboardInterrupt:
        interrupted_epoch = max(start_epoch, min(args.epochs, locals().get("epoch", start_epoch)))
        _save_train_ckpt(out / "interrupted.pt", interrupted_epoch)
        _save_train_ckpt(out / "last.pt", interrupted_epoch)
        print(f"training_interrupted saved={out / 'interrupted.pt'}")
        return

    test_dir = data_root / "test"
    if test_dir.exists() and (out / "best.pt").exists():
        best = torch.load(out / "best.pt", map_location=device)
        eval_model = build_model(
            backbone=best.get("backbone", args.backbone),
            pretrained_backbone=False,
        ).to(device)
        eval_model.load_state_dict(best["state_dict"])
        if device.type == "cuda":
            eval_model = eval_model.to(memory_format=torch.channels_last)
        eval_model.eval()

        test_tf = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
        ])
        test_ds = datasets.ImageFolder(test_dir, transform=test_tf)
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=bool(torch.cuda.is_available()),
        )
        test_ai_idx = int(test_ds.class_to_idx["ai"])
        test_loss_fn = BinaryClassificationLoss(
            kind=args.loss,
            gamma=args.focal_gamma,
            real_weight=real_weight,
            ai_weight=ai_weight,
        ).to(device)
        test_loss_sum = 0.0
        test_total = 0
        test_probs: list[float] = []
        test_labels: list[float] = []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                y_bin = _binary_targets(y, test_ai_idx)
                if device.type == "cuda":
                    x = x.to(memory_format=torch.channels_last)
                logits = eval_model(x)
                loss = test_loss_fn(logits, y_bin)
                probs = torch.sigmoid(logits / max(float(best.get("temperature", 1.0)), 1e-6))
                test_probs.extend(probs.detach().cpu().tolist())
                test_labels.extend(y_bin.detach().cpu().tolist())
                bs = y.shape[0]
                test_loss_sum += float(loss.item()) * bs
                test_total += bs
        test_report = _eval_metrics_from_probs(
            np.asarray(test_probs, dtype=np.float64),
            np.asarray(test_labels, dtype=np.float64),
            threshold=float(best.get("threshold", args.decision_threshold)),
        )
        test_report["test_loss"] = test_loss_sum / max(test_total, 1)
        (out / "test_metrics.json").write_text(json.dumps(test_report, indent=2), encoding="utf-8")
        print(f"saved test metrics to {out / 'test_metrics.json'}")

    if args.export_release and (out / "best.pt").exists():
        rel = out / "releases" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        rel.mkdir(parents=True, exist_ok=True)
        for name in (
            "best.pt",
            "best_metrics.json",
            "best_group_metrics.json",
            "calibration.json",
            "test_metrics.json",
            "config.json",
        ):
            src = out / name
            if src.exists():
                shutil.copy2(src, rel / name)
        (out / "latest_release.txt").write_text(str(rel), encoding="utf-8")
        print(f"saved release bundle to {rel}")

    print(f"saved best model to {out / 'best.pt'} with best_auc={best_auc:.4f}")


if __name__ == "__main__":
    main()
