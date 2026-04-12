"""Training helpers: loss, EMA, mixup, LR schedule, evaluation metrics (used by ``train.main``)."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from .data import unpack_image_batch
from .metrics import full_metric_report


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
            target_probs = torch.stack((1.0 - targets, targets), dim=1)
            log_probs = F.log_softmax(logits_2c, dim=1)
            weighted_targets = target_probs * self.class_weights.unsqueeze(0)
            normalizer = weighted_targets.sum(dim=1).clamp(min=1e-6)
            losses = -(weighted_targets * log_probs).sum(dim=1) / normalizer
            return losses.mean()

        sample_weights = (targets * self.ai_weight) + ((1.0 - targets) * self.real_weight)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        if self.kind == "bce":
            return (bce * sample_weights).sum() / sample_weights.sum().clamp(min=1e-6)

        probs = torch.sigmoid(logits)
        pt = (targets * probs) + ((1.0 - targets) * (1.0 - probs))
        focal = (1.0 - pt).pow(self.gamma)
        weighted = focal * bce * sample_weights
        return weighted.sum() / sample_weights.sum().clamp(min=1e-6)


def _binary_targets(y: torch.Tensor, ai_idx: int) -> torch.Tensor:
    return (y == ai_idx).float()


def _apply_label_smoothing(targets: torch.Tensor, smoothing: float) -> torch.Tensor:
    smoothing = float(max(0.0, min(0.499, smoothing)))
    if smoothing <= 0.0:
        return targets
    return targets * (1.0 - smoothing) + (0.5 * smoothing)


def _build_lr_scheduler(
    opt: torch.optim.Optimizer,
    *,
    epochs: int,
    base_lr: float,
    warmup_epochs: int,
):
    """Cosine decay; optional linear warmup (modern vision recipe)."""
    eta_min = float(base_lr) * 0.05
    e = max(int(epochs), 1)
    w = max(int(warmup_epochs), 0)
    if w == 0:
        return CosineAnnealingLR(opt, T_max=e, eta_min=eta_min)
    if w >= e:
        raise ValueError(f"warmup_epochs ({w}) must be less than epochs ({e})")
    warmup = LinearLR(opt, start_factor=0.01, end_factor=1.0, total_iters=w)
    cosine = CosineAnnealingLR(opt, T_max=e - w, eta_min=eta_min)
    return SequentialLR(opt, schedulers=[warmup, cosine], milestones=[w])


def _mixup_batch(
    x: torch.Tensor,
    targets: torch.Tensor,
    metadata_features: torch.Tensor | None,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    alpha = float(max(0.0, alpha))
    if alpha <= 0.0 or x.shape[0] < 2:
        return x, targets, metadata_features
    lam = float(np.random.beta(alpha, alpha))
    index = torch.randperm(x.shape[0], device=x.device)
    mixed_x = (lam * x) + ((1.0 - lam) * x[index])
    mixed_targets = (lam * targets) + ((1.0 - lam) * targets[index])
    if metadata_features is None:
        mixed_metadata = None
    else:
        mixed_metadata = (lam * metadata_features) + ((1.0 - lam) * metadata_features[index])
    return mixed_x, mixed_targets, mixed_metadata


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


def _promotion_status(report: dict[str, Any]) -> tuple[bool, str]:
    if not bool(report.get("threshold_operable", True)):
        return False, "no_operable_threshold"
    if bool(report.get("predicts_single_class", False)):
        return False, "single_class_predictions"
    if float(report.get("recall_ai", 0.0)) <= 0.0:
        return False, "missing_ai_recall"
    if float(report.get("recall_real", 0.0)) <= 0.0:
        return False, "missing_real_recall"
    objective = str(report.get("threshold_objective", ""))
    objective_score = report.get("threshold_objective_score")
    if objective_score is not None:
        score = float(objective_score)
        if objective == "balanced" and score <= 0.500001:
            return False, "uninformative_balanced_threshold"
        if objective == "youden" and score <= 0.000001:
            return False, "uninformative_youden_threshold"
    return True, "ok"


def _accumulate_binary_classifier_loader(
    model: nn.Module,
    loader: Iterable[Any],
    device: torch.device,
    ai_idx: int,
    loss_fn: nn.Module,
    *,
    paired_paths: list[tuple[str, int]] | None,
    prob_temperature: float | None = None,
    store_logits: bool = True,
) -> tuple[float, list[float] | None, list[float], list[float], list[str]]:
    """Forward ``model`` over ``loader`` with ``loss_fn``; collect loss, labels, probs, optional logits/paths."""
    model.eval()
    loss_sum = 0.0
    total = 0
    logits_all: list[float] = []
    probs_all: list[float] = []
    labels_all: list[float] = []
    paths_all: list[str] = []
    offset = 0
    with torch.no_grad():
        for batch in loader:
            x, metadata_features, y = unpack_image_batch(batch)
            x = x.to(device, non_blocking=True)
            if metadata_features is not None:
                metadata_features = metadata_features.to(device=device, dtype=x.dtype, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_bin = _binary_targets(y, ai_idx)
            if device.type == "cuda":
                x = x.to(memory_format=torch.channels_last)
            logits = model(x, metadata_features=metadata_features)
            loss = loss_fn(logits, y_bin)
            if prob_temperature is None:
                probs = torch.sigmoid(logits)
            else:
                probs = torch.sigmoid(logits / max(float(prob_temperature), 1e-6))

            bs = y.shape[0]
            if paired_paths is not None:
                batch_paths = [paired_paths[offset + i][0] for i in range(bs)]
                offset += bs
            else:
                batch_paths = []

            if store_logits:
                logits_all.extend(logits.detach().cpu().tolist())
            probs_all.extend(probs.detach().cpu().tolist())
            labels_all.extend(y_bin.detach().cpu().tolist())
            paths_all.extend(batch_paths)

            loss_sum += float(loss.item()) * bs
            total += bs

    avg_loss = loss_sum / max(total, 1)
    logits_out: list[float] | None = logits_all if store_logits else None
    return avg_loss, logits_out, probs_all, labels_all, paths_all


def evaluate(
    model: nn.Module,
    loader: Iterable[Any],
    device: torch.device,
    ai_idx: int,
    val_samples: list[tuple[str, int]],
    loss_fn: nn.Module,
):
    avg_loss, logits_list, probs_list, labels_list, paths_list = _accumulate_binary_classifier_loader(
        model,
        loader,
        device,
        ai_idx,
        loss_fn,
        paired_paths=val_samples,
        prob_temperature=None,
        store_logits=True,
    )
    assert logits_list is not None
    return (
        avg_loss,
        np.asarray(logits_list, dtype=np.float64),
        np.asarray(probs_list, dtype=np.float64),
        np.asarray(labels_list, dtype=np.float64),
        paths_list,
    )


def load_matching_checkpoint_state(model: nn.Module, init_ckpt: dict[str, Any]) -> None:
    """Load overlapping keys from ``init_ckpt`` into ``model`` (shape-matched subset)."""
    current = model.state_dict()
    loaded = init_ckpt.get("state_dict", {})
    matched: dict[str, Any] = {}
    for key, value in loaded.items():
        if key in current and current[key].shape == value.shape:
            matched[key] = value
    missing, unexpected = model.load_state_dict(matched, strict=False)
    print(f"initialized_from_checkpoint matched={len(matched)} missing={len(missing)} unexpected={len(unexpected)}")


def _dataset_counts(root: Path) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        out[split] = {}
        for cls in ("ai", "real"):
            d = root / split / cls
            out[split][cls] = len(list(d.glob("*"))) if d.exists() else 0
    return out
