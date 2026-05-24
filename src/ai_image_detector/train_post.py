"""Post-training evaluation on holdout test dir and optional release export."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from .checkpoints import load_checkpoint
from .data import MetadataImageFolder, build_loader_kwargs, make_eval_transform, make_jailed_rgb_loader
from .io_limits import configure_pil_limits
from .model import build_model
from .release_tools import write_timestamped_release
from .train_support import (
    BinaryClassificationLoss,
    _accumulate_binary_classifier_loader,
    _eval_metrics_from_probs,
)
from .utils.jsonio import write_json_atomic


def run_holdout_test_metrics_if_ready(
    *,
    out: Path,
    data_root: Path,
    device: torch.device,
    args: argparse.Namespace,
    real_weight: float,
    ai_weight: float,
) -> None:
    test_dir = data_root / "test"
    best_path = out / "best.safetensors"
    if not test_dir.exists() or not best_path.exists():
        return

    best = load_checkpoint(best_path, map_location=device)
    eval_model = build_model(
        backbone=best.get("backbone", args.backbone),
        pretrained_backbone=False,
        metadata_feature_dim=int(best.get("metadata_feature_dim", 0)),
    ).to(device)
    eval_model.load_state_dict(best["state_dict"])
    if device.type == "cuda":
        eval_model = eval_model.to(memory_format=torch.channels_last)
    eval_model.eval()

    configure_pil_limits()
    img_size = int(best.get("img_size", args.img_size))
    test_tf = make_eval_transform(img_size)
    test_dataset_cls = MetadataImageFolder if bool(best.get("use_metadata_features", False)) else datasets.ImageFolder
    test_loader_fn = make_jailed_rgb_loader(test_dir)
    test_ds = test_dataset_cls(test_dir, transform=test_tf, loader=test_loader_fn)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        **build_loader_kwargs(num_workers=args.num_workers),
    )
    test_ai_idx = int(test_ds.class_to_idx["ai"])
    test_loss_fn = BinaryClassificationLoss(
        kind=args.loss,
        gamma=args.focal_gamma,
        real_weight=real_weight,
        ai_weight=ai_weight,
    ).to(device)
    avg_loss, _, probs_list, labels_list, _ = _accumulate_binary_classifier_loader(
        eval_model,
        test_loader,
        device,
        test_ai_idx,
        test_loss_fn,
        paired_paths=None,
        prob_temperature=float(best.get("temperature", 1.0)),
        store_logits=False,
    )
    test_report = _eval_metrics_from_probs(
        np.asarray(probs_list, dtype=np.float64),
        np.asarray(labels_list, dtype=np.float64),
        threshold=float(best.get("threshold") or args.decision_threshold or 0.5),
    )
    test_report["test_loss"] = avg_loss
    write_json_atomic(out / "test_metrics.json", test_report, indent=2)
    print(f"saved test metrics to {out / 'test_metrics.json'}")


def maybe_export_training_release(out: Path, *, export_release: bool) -> None:
    if not export_release or not (out / "best.safetensors").exists():
        return
    rel = write_timestamped_release(
        out,
        (
            "best_metrics.json",
            "best_group_metrics.json",
            "calibration.json",
            "test_metrics.json",
            "config.json",
            "inference_spec.json",
            "best_checkpoint.txt",
            "best_model_summary.json",
        ),
        preferred_artifact="best.safetensors",
    )
    print(f"saved release bundle to {rel}")
