from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import shutil
import subprocess

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ai_image_detector.checkpoints import load_checkpoint, resolve_checkpoint_path, save_safetensors_checkpoint
from ai_image_detector.ensemble import EnsembleDetector, load_models
from ai_image_detector.model import build_model


def _git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True)
        return out.strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser(description="Distill ensemble teacher into a compact student")
    ap.add_argument("--data", default="./data_best")
    ap.add_argument("--teacher", nargs="+", required=True)
    ap.add_argument("--ensemble-config", default="", help="Optional JSON with learned ensemble weights/threshold")
    ap.add_argument("--out", default="./artifacts_distill")
    ap.add_argument("--student-backbone", choices=["tiny", "effb0"], default="tiny")
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--alpha", type=float, default=0.6, help="teacher loss weight")
    ap.add_argument("--resume", default="", help="Path to training checkpoint (default: <out>/last.pt)")
    ap.add_argument("--save-every", type=int, default=1)
    ap.add_argument("--patience", type=int, default=0)
    ap.add_argument("--min-delta", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--export-release", action="store_true", default=True)
    ap.add_argument("--no-export-release", dest="export_release", action="store_false")
    ap.add_argument("--save-safetensors", action="store_true", default=True)
    ap.add_argument("--no-save-safetensors", dest="save_safetensors", action="store_false")
    args = ap.parse_args()

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

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = load_models(args.teacher, device, ensemble_config=args.ensemble_config)
    teacher = EnsembleDetector(loaded.models, weights=loaded.weights, img_sizes=loaded.img_sizes).to(device)
    teacher.eval()

    student = build_model(backbone=args.student_backbone, pretrained_backbone=True).to(device)

    tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])
    train_ds = datasets.ImageFolder(Path(args.data) / "train", transform=tf)
    val_ds = datasets.ImageFolder(Path(args.data) / "val", transform=tf)
    ai_idx = int(train_ds.class_to_idx["ai"])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    config = {
        "args": vars(args),
        "git_commit": _git_commit(),
        "dataset_counts": {"train": len(train_ds), "val": len(val_ds)},
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    opt = AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    best_acc = -1.0
    no_improve = 0
    start_epoch = 1

    def _save_train_ckpt(path: Path, epoch: int):
        torch.save(
            {
                "epoch": epoch,
                "state_dict": student.state_dict(),
                "optimizer": opt.state_dict(),
                "best_acc": float(best_acc),
                "no_improve": int(no_improve),
                "img_size": args.img_size,
                "backbone": args.student_backbone,
            },
            path,
        )
        (out / "latest_checkpoint.txt").write_text(path.name, encoding="utf-8")

    resume_path = Path(args.resume) if args.resume else (out / "last.pt")
    if resume_path.exists():
        ckpt = load_checkpoint(resume_path, map_location=device)
        student.load_state_dict(ckpt["state_dict"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        best_acc = float(ckpt.get("best_acc", best_acc))
        no_improve = int(ckpt.get("no_improve", no_improve))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"resumed_from={resume_path} start_epoch={start_epoch} best_acc={best_acc:.4f}")

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            student.train()
            skipped_batches = 0
            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)
                target = (y == ai_idx).float()
                with torch.no_grad():
                    t_logit = teacher(x)

                s_logit = student(x)
                hard_loss = F.binary_cross_entropy_with_logits(s_logit, target)
                soft_loss = F.mse_loss(torch.sigmoid(s_logit), torch.sigmoid(t_logit))
                loss = args.alpha * soft_loss + (1.0 - args.alpha) * hard_loss
                if not torch.isfinite(loss):
                    skipped_batches += 1
                    opt.zero_grad(set_to_none=True)
                    print(f"warn epoch={epoch} skipped_batch=non_finite_loss")
                    continue

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

            student.eval()
            corr = 0
            tot = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device)
                    y = y.to(device)
                    target = (y == ai_idx).long()
                    pred = (torch.sigmoid(student(x)) >= 0.5).long()
                    corr += (pred == target).sum().item()
                    tot += target.numel()
            acc = corr / max(tot, 1)
            print(f"epoch={epoch} val_acc={acc:.4f}")
            with (out / "training_log.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"epoch": epoch, "val_acc": acc, "skipped_batches": skipped_batches}) + "\n")

            _save_train_ckpt(out / "last.pt", epoch)
            if args.save_every > 0 and (epoch % args.save_every == 0):
                _save_train_ckpt(out / f"epoch_{epoch:03d}.pt", epoch)

            if acc > (best_acc + args.min_delta):
                best_acc = acc
                no_improve = 0
                ckpt = {
                    "state_dict": student.state_dict(),
                    "img_size": args.img_size,
                    "threshold": 0.5,
                    "temperature": 1.0,
                    "model_id": f"distilled-{args.student_backbone}",
                    "backbone": args.student_backbone,
                    "metrics": {"val_acc": float(acc)},
                }
                torch.save(ckpt, out / "best.pt")
                if args.save_safetensors:
                    save_safetensors_checkpoint(out / "best.safetensors", ckpt)
                preferred_best = resolve_checkpoint_path(out / "best.pt")
                (out / "best_checkpoint.txt").write_text(str(preferred_best), encoding="utf-8")
                (out / "best_model_summary.json").write_text(
                    json.dumps(
                        {
                            "preferred_checkpoint": str(preferred_best),
                            "student_backbone": args.student_backbone,
                            "img_size": args.img_size,
                            "threshold": 0.5,
                            "temperature": 1.0,
                            "metrics": {"val_acc": float(acc)},
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

    best_path = resolve_checkpoint_path(out / "best.pt")
    if args.export_release and best_path.exists():
        rel = out / "releases" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        rel.mkdir(parents=True, exist_ok=True)
        for name in ("config.json", "best_checkpoint.txt", "best_model_summary.json"):
            src = out / name
            if src.exists():
                shutil.copy2(src, rel / name)
        shutil.copy2(best_path, rel / best_path.name)
        (out / "latest_release.txt").write_text(str(rel), encoding="utf-8")
        print(f"saved release bundle to {rel}")

    print(f"saved={best_path} best_acc={best_acc:.4f}")


if __name__ == "__main__":
    main()
