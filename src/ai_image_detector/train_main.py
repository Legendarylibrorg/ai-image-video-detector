"""Image training orchestration (``train.main`` delegates here)."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .checkpoints import (
    args_dict_for_checkpoint,
    load_checkpoint,
    load_training_checkpoint,
    save_safetensors_checkpoint,
    save_training_checkpoint,
)
from .data import make_loaders, unpack_image_batch
from .metrics import find_best_threshold, fit_temperature, full_metric_report, sigmoid
from .model import build_model, model_runtime_spec
from .runtime import build_adamw, configure_torch_runtime, maybe_compile_model, seed_all, training_device
from .train_post import maybe_export_training_release, run_holdout_test_metrics_if_ready
from .train_run_artifacts import prepare_training_output_dir
from .train_support import (
    BinaryClassificationLoss,
    EMA,
    _apply_label_smoothing,
    _binary_targets,
    _build_lr_scheduler,
    _eval_metrics_from_probs,
    _group_report,
    _mixup_batch,
    _promotion_status,
    evaluate,
    load_matching_checkpoint_state,
)
from .utils.jsonio import write_json_atomic

__all__ = ["build_train_argparser", "run_image_training"]


def build_train_argparser() -> argparse.ArgumentParser:
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
    ap.add_argument(
        "--backbone",
        choices=["tiny", "effb0", "effb2", "convnext_tiny", "convnext_small"],
        default="tiny",
    )
    ap.add_argument(
        "--warmup-epochs",
        type=int,
        default=0,
        help="Linear LR warmup epochs before cosine decay (0 = cosine only, ViT/ConvNeXt-style recipe when >0)",
    )
    ap.add_argument("--no-pretrained-backbone", action="store_true")
    ap.add_argument("--mixup-alpha", type=float, default=0.2, help="Beta(alpha, alpha) mixup strength; 0 disables")
    ap.add_argument("--label-smoothing", type=float, default=0.02, help="Binary label smoothing applied after mixup")
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
    ap.add_argument("--use-metadata-features", action="store_true", help="Add metadata/file cues as a small auxiliary branch")
    ap.add_argument("--init-from", default="", help="Optional checkpoint to partially initialize from before training")
    ap.add_argument(
        "--max-nan-batch-fraction",
        type=float,
        default=0.25,
        help="Abort epoch if fraction of training batches skipped (non-finite loss) exceeds this (0-1)",
    )
    ap.add_argument(
        "--strict-dataset",
        action="store_true",
        help="Require SHA256 on train+val and abort if any file bytes appear in both splits (content leakage)",
    )
    ap.add_argument(
        "--dataset-manifest",
        choices=["off", "standard", "full"],
        default="standard",
        help="Write dataset_manifest.json: standard=val hashed + train paths; full=all hashed; off=skip",
    )
    ap.add_argument(
        "--skip-data-preflight",
        action="store_true",
        help="Skip symlink/mount preflight (not recommended; tests may set AID_SKIP_DATA_PREFLIGHT instead)",
    )
    return ap


def run_image_training(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data)

    seed_all(args.seed)
    device = training_device()
    configure_torch_runtime(device, args.deterministic)
    train_loader, val_loader, classes, class_to_idx, train_samples, val_samples, train_distribution, val_distribution, class_weight_map, metadata_dim = make_loaders(
        args.data,
        args.img_size,
        args.batch_size,
        num_workers=args.num_workers,
        use_metadata_features=bool(args.use_metadata_features),
        skip_data_preflight=bool(args.skip_data_preflight),
    )
    print(f"class_distribution train={train_distribution} val={val_distribution}")
    print(f"class_weights_inverse_freq={class_weight_map}")

    run_config = prepare_training_output_dir(
        out,
        data_root,
        args,
        train_samples=train_samples,
        val_samples=val_samples,
        classes=classes,
        metadata_dim=metadata_dim,
    )

    if set(classes) != {"ai", "real"}:
        raise ValueError(f"Expected classes exactly ai/real, got {classes}")
    ai_idx = int(class_to_idx["ai"])
    real_weight = float(class_weight_map["real"])
    ai_weight = float(class_weight_map["ai"])

    model = build_model(
        backbone=args.backbone,
        pretrained_backbone=(not args.no_pretrained_backbone),
        metadata_feature_dim=metadata_dim,
    ).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    # Keep a plain module for checkpoint I/O; use the compiled wrapper only for forward passes.
    train_model = maybe_compile_model(model, enabled=bool(args.compile))
    opt = build_adamw(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, device=device)
    sched = _build_lr_scheduler(opt, epochs=args.epochs, base_lr=args.lr, warmup_epochs=args.warmup_epochs)
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
    best_checkpoint_saved = False

    if args.init_from:
        init_path = Path(args.init_from)
        if not init_path.exists():
            raise FileNotFoundError(init_path)
        init_ckpt = load_checkpoint(init_path, map_location=device)
        load_matching_checkpoint_state(model, init_ckpt)

    def _save_train_ckpt(path: Path, epoch: int) -> None:
        save_training_checkpoint(
            path,
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
                "args": args_dict_for_checkpoint(args),
                "classes": classes,
                "class_to_idx": class_to_idx,
                "backbone": args.backbone,
                "img_size": args.img_size,
                "metadata_feature_dim": metadata_dim,
                "use_metadata_features": bool(args.use_metadata_features),
            },
        )

    resume_path = Path(args.resume) if args.resume else (out / "last.pt")
    if resume_path.exists():
        ckpt = load_training_checkpoint(resume_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        ema.shadow.load_state_dict(ckpt.get("ema_state_dict", ckpt["state_dict"]))
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            try:
                sched.load_state_dict(ckpt["scheduler"])
            except (KeyError, RuntimeError, ValueError) as exc:
                print(f"scheduler_state_skipped reason={exc}")
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        best_auc = float(ckpt.get("best_auc", best_auc))
        no_improve = int(ckpt.get("no_improve", no_improve))
        degenerate_epochs = int(ckpt.get("degenerate_epochs", degenerate_epochs))
        model_id = str(ckpt.get("model_id", model_id))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_checkpoint_saved = bool(Path(out / "best.safetensors").exists())
        print(f"resumed_from={resume_path} start_epoch={start_epoch} best_auc={best_auc:.4f}")

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            train_model.train()
            opt.zero_grad(set_to_none=True)
            step_idx = 0
            skipped_batches = 0
            for batch in train_loader:
                x, metadata_features, y = unpack_image_batch(batch)
                x = x.to(device, non_blocking=True)
                if metadata_features is not None:
                    metadata_features = metadata_features.to(device=device, dtype=x.dtype, non_blocking=True)
                y = y.to(device, non_blocking=True)
                y_bin = _binary_targets(y, ai_idx)
                if device.type == "cuda":
                    x = x.to(memory_format=torch.channels_last)
                x, y_bin, metadata_features = _mixup_batch(
                    x,
                    y_bin,
                    metadata_features,
                    args.mixup_alpha,
                )
                y_bin = _apply_label_smoothing(y_bin, args.label_smoothing)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = train_model(x, metadata_features=metadata_features)
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

            n_batches = len(train_loader)
            if n_batches > 0 and (skipped_batches / float(n_batches)) > float(args.max_nan_batch_fraction):
                raise RuntimeError(
                    f"training_abort_non_finite_loss epoch={epoch} skipped_batches={skipped_batches} "
                    f"total_batches={n_batches} max_nan_batch_fraction={args.max_nan_batch_fraction}"
                )

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
            threshold_metrics: dict[str, Any] = {"operable": True, "search_status": "fixed"}
            if args.decision_threshold is None:
                threshold, threshold_score, threshold_metrics = find_best_threshold(
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
            report["threshold_operable"] = bool(threshold_metrics.get("operable", True))
            report["threshold_search_status"] = str(threshold_metrics.get("search_status", "unknown"))
            report["lr"] = float(opt.param_groups[0]["lr"])
            report["skipped_batches"] = skipped_batches
            promotion_eligible, promotion_reason = _promotion_status(report)
            report["promotion_eligible"] = bool(promotion_eligible)
            report["promotion_reason"] = promotion_reason

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
            write_json_atomic(out / "last_metrics.json", last, indent=2)
            with (out / "training_log.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"epoch": epoch, **report}) + "\n")

            _save_train_ckpt(out / "last.pt", epoch)
            if args.save_every > 0 and (epoch % args.save_every == 0):
                _save_train_ckpt(out / f"epoch_{epoch:03d}.pt", epoch)

            if report["auc"] > (best_auc + args.min_delta):
                if not promotion_eligible:
                    print(
                        "skip_best_checkpoint epoch={} reason={} auc={:.4f}".format(
                            epoch,
                            promotion_reason,
                            report["auc"],
                        )
                    )
                    no_improve += 1
                    if args.patience > 0 and no_improve >= args.patience:
                        print(f"early_stopping epoch={epoch} no_improve={no_improve} patience={args.patience}")
                        break
                    continue
                best_auc = float(report["auc"])
                no_improve = 0
                ckpt = {
                    "state_dict": ema.shadow.state_dict(),
                    "img_size": args.img_size,
                    "threshold": float(threshold),
                    "temperature": float(temperature),
                    "model_id": model_id,
                    "backbone": args.backbone,
                    "metadata_feature_dim": metadata_dim,
                    "use_metadata_features": bool(args.use_metadata_features),
                    # Keep safetensors metadata small and stable; detailed metrics live in JSON sidecars.
                    "schema": "ai-image-detector-model-v1",
                    "git_commit": str(run_config.get("git_commit", "")),
                    "created_utc": str(run_config.get("created_utc", "")),
                    "calibration": {
                        "threshold_source": str(report.get("threshold_source", "")),
                        "threshold_objective": str(report.get("threshold_objective", "")),
                        "threshold_objective_score": report.get("threshold_objective_score"),
                        "threshold_operable": bool(report.get("threshold_operable", True)),
                        "threshold_search_status": str(report.get("threshold_search_status", "")),
                        "temperature_nll": float(report.get("temperature_nll", 0.0)),
                    },
                    "trainer": {
                        "epochs": int(args.epochs),
                        "base_lr": float(args.lr),
                        "warmup_epochs": int(args.warmup_epochs),
                        "amp": bool(args.amp),
                        "ema_decay": float(args.ema_decay),
                        "mixup_alpha": float(args.mixup_alpha),
                        "label_smoothing": float(args.label_smoothing),
                        "weight_decay": float(args.weight_decay),
                    },
                    "runtime_spec": model_runtime_spec(
                        backbone=args.backbone,
                        img_size=args.img_size,
                        metadata_feature_dim=metadata_dim,
                    ),
                }
                save_safetensors_checkpoint(out / "best.safetensors", ckpt)
                preferred_best = out / "best.safetensors"
                best_checkpoint_saved = True
                (out / "best_checkpoint.txt").write_text(preferred_best.name, encoding="utf-8")
                write_json_atomic(out / "best_metrics.json", report, indent=2)
                write_json_atomic(out / "best_group_metrics.json", grouped, indent=2)
                write_json_atomic(
                    out / "calibration.json",
                    {
                        "threshold": float(threshold),
                        "temperature": float(temperature),
                        "objective": args.threshold_objective,
                        "metrics": report,
                        "runtime_spec": model_runtime_spec(
                            backbone=args.backbone,
                            img_size=args.img_size,
                            metadata_feature_dim=metadata_dim,
                        ),
                    },
                    indent=2,
                )
                write_json_atomic(
                    out / "best_model_summary.json",
                    {
                        "epoch": epoch,
                        "preferred_checkpoint": preferred_best.name,
                        "metrics": report,
                        "calibration": {
                            "threshold": float(threshold),
                            "temperature": float(temperature),
                            "objective": args.threshold_objective,
                            "objective_score": float(threshold_score) if threshold_score is not None else None,
                        },
                        "runtime_spec": model_runtime_spec(
                            backbone=args.backbone,
                            img_size=args.img_size,
                            metadata_feature_dim=metadata_dim,
                        ),
                    },
                    indent=2,
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

    if not best_checkpoint_saved:
        raise RuntimeError("no_promotable_checkpoint")

    run_holdout_test_metrics_if_ready(
        out=out,
        data_root=data_root,
        device=device,
        args=args,
        real_weight=real_weight,
        ai_weight=ai_weight,
    )
    maybe_export_training_release(out, export_release=bool(args.export_release))

    print(f"saved best model to {out / 'best.safetensors'} with best_auc={best_auc:.4f}")
