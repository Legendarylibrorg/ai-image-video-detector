from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from .checkpoints import (
    args_dict_for_checkpoint,
    load_checkpoint,
    load_training_checkpoint,
    save_safetensors_checkpoint,
    save_training_checkpoint,
)
from .data import build_loader_kwargs
from .io_limits import MAX_VIDEO_DECODE_FRAMES, prepare_video_path
from .release_tools import write_timestamped_release
from .runtime import build_adamw, configure_torch_runtime, git_commit, seed_all


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def _collect_videos(root: Path) -> List[Tuple[str, int]]:
    classes = {"real": 0, "ai": 1}
    out: List[Tuple[str, int]] = []
    for cls, y in classes.items():
        d = root / cls
        if not d.exists():
            continue
        for p in d.rglob("*"):
            if p.suffix.lower() in VIDEO_EXTS:
                if p.is_symlink():
                    continue
                out.append((str(p), y))
    return out


def _sample_frames(path: str, num_frames: int, size: int, random_offset: bool) -> np.ndarray:
    vpath = prepare_video_path(path)
    vpath_s = str(vpath)
    cap = cv2.VideoCapture(vpath_s)
    if not cap.isOpened():
        raise RuntimeError(f"unable to open video: {vpath_s}")

    total_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_raw < 1:
        total_raw = 1
    total = max(1, min(total_raw, MAX_VIDEO_DECODE_FRAMES))

    if total <= num_frames:
        indices = list(range(total))
        while len(indices) < num_frames:
            indices.append(indices[-1] if indices else 0)
    else:
        stride = total / float(num_frames)
        base = random.random() * stride if random_offset else 0.0
        indices = [int(min(total - 1, base + i * stride)) for i in range(num_frames)]

    idx_set = set(indices)
    current = 0
    target_pos = {i: [] for i in idx_set}
    for pos, idx in enumerate(indices):
        target_pos[idx].append(pos)

    out = [None] * len(indices)
    max_read = min(MAX_VIDEO_DECODE_FRAMES, max(indices) + 1 if indices else 1)
    while current < max_read:
        ok, frame = cap.read()
        if not ok:
            break
        if current in idx_set:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
            arr = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)
            for pos in target_pos[current]:
                out[pos] = arr
        current += 1
        if current > max(indices):
            break

    cap.release()

    fallback = out[0] if out and out[0] is not None else np.zeros((3, size, size), dtype=np.float32)
    for i in range(len(out)):
        if out[i] is None:
            out[i] = fallback

    return np.stack(out, axis=0)


class VideoDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], num_frames: int, size: int, augment: bool):
        self.samples = samples
        self.num_frames = num_frames
        self.size = size
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, y = self.samples[i]
        x = _sample_frames(path, self.num_frames, self.size, random_offset=self.augment)
        if self.augment and random.random() < 0.5:
            x = x[:, :, :, ::-1].copy()
        return torch.from_numpy(x), torch.tensor(float(y), dtype=torch.float32)


class TemporalVideoDetector(nn.Module):
    def __init__(self, hidden: int = 384, pretrained_backbone: bool = False):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        base = models.efficientnet_b0(weights=weights)
        self.frame_encoder = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        feat_dim = 1280

        self.temporal = nn.LSTM(
            input_size=feat_dim,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 256),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(256, 1),
        )

    def encode_frames(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        flat = x.reshape(b * t, c, h, w)
        if flat.device.type == "cuda":
            flat = flat.contiguous(memory_format=torch.channels_last)
        f = self.frame_encoder(flat)
        f = self.pool(f).flatten(1)
        return f.view(b, t, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = self.encode_frames(x)
        y, _ = self.temporal(seq)
        pooled = y.mean(dim=1)
        return self.head(pooled).squeeze(1)


def _evaluate(model, loader, device):
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    loss_sum = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logit = model(x)
            loss = loss_fn(logit, y)
            p = torch.sigmoid(logit)
            pred = (p >= 0.5).float()
            correct += int((pred == y).sum().item())
            total += y.numel()
            loss_sum += float(loss.item()) * y.numel()
    return loss_sum / max(total, 1), correct / max(total, 1)

def train_main() -> None:
    ap = argparse.ArgumentParser(description="Train temporal video deepfake detector")
    ap.add_argument("--data", required=True, help="video_data with train/{ai,real}, val/{ai,real}")
    ap.add_argument("--out", default="./video_artifacts")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--compile", action="store_true", default=True)
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    ap.add_argument("--resume", default="", help="Path to training checkpoint (default: <out>/last_video.pt if present)")
    ap.add_argument("--save-every", type=int, default=1, help="Save epoch checkpoint every N epochs")
    ap.add_argument("--patience", type=int, default=0, help="Early stopping patience in epochs (0 disables)")
    ap.add_argument("--min-delta", type=float, default=0.0, help="Minimum accuracy improvement to reset patience")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--pretrained-backbone", action="store_true", default=False)
    ap.add_argument("--export-release", action="store_true", default=True)
    ap.add_argument("--no-export-release", dest="export_release", action="store_false")
    args = ap.parse_args()

    seed_all(args.seed)
    root = Path(args.data)
    train_samples = _collect_videos(root / "train")
    val_samples = _collect_videos(root / "val")
    if not train_samples or not val_samples:
        raise RuntimeError("Expected videos in train/ai, train/real, val/ai, val/real")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    config = {
        "args": args_dict_for_checkpoint(args),
        "git_commit": git_commit(),
        "dataset_counts": {"train": len(train_samples), "val": len(val_samples)},
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    train_ds = VideoDataset(train_samples, args.frames, args.img_size, augment=True)
    val_ds = VideoDataset(val_samples, args.frames, args.img_size, augment=False)

    dl_kwargs = build_loader_kwargs(num_workers=args.num_workers, prefetch_factor=2)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **dl_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **dl_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_torch_runtime(device, args.deterministic)
    model = TemporalVideoDetector(pretrained_backbone=args.pretrained_backbone).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    if args.compile:
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as exc:
            print(f"compile_disabled reason={exc}")
    opt = build_adamw(model.parameters(), lr=args.lr, weight_decay=1e-4, device=device)
    sched = CosineAnnealingLR(opt, T_max=max(args.epochs, 1), eta_min=args.lr * 0.1)
    loss_fn = nn.BCEWithLogitsLoss()
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    grad_accum = max(1, int(args.grad_accum))

    best_acc = -1.0
    no_improve = 0
    start_epoch = 1

    def _save_train_ckpt(path: Path, epoch: int) -> None:
        save_training_checkpoint(
            path,
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "scaler": scaler.state_dict(),
                "best_acc": float(best_acc),
                "no_improve": int(no_improve),
                "img_size": args.img_size,
                "frames": args.frames,
                "pretrained_backbone": bool(args.pretrained_backbone),
            },
        )

    resume_path = Path(args.resume) if args.resume else (out / "last_video.pt")
    if resume_path.exists():
        ckpt = load_training_checkpoint(resume_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            sched.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        best_acc = float(ckpt.get("best_acc", best_acc))
        no_improve = int(ckpt.get("no_improve", no_improve))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"resumed_from={resume_path} start_epoch={start_epoch} best_acc={best_acc:.4f}")

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            opt.zero_grad(set_to_none=True)
            step_idx = 0
            skipped_batches = 0
            for x, y in train_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logit = model(x)
                    loss = loss_fn(logit, y) / grad_accum
                if not torch.isfinite(loss):
                    skipped_batches += 1
                    opt.zero_grad(set_to_none=True)
                    print(f"warn epoch={epoch} skipped_batch=non_finite_loss")
                    continue
                scaler.scale(loss).backward()
                step_idx += 1
                if step_idx % grad_accum == 0:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)

            if step_idx % grad_accum != 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            sched.step()
            val_loss, val_acc = _evaluate(model, val_loader, device)
            print(f"epoch={epoch} val_loss={val_loss:.5f} val_acc={val_acc:.4f}")
            with (out / "training_log.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"epoch": epoch, "val_loss": val_loss, "val_acc": val_acc, "skipped_batches": skipped_batches, "lr": float(opt.param_groups[0]["lr"])}) + "\n")
            _save_train_ckpt(out / "last_video.pt", epoch)
            if args.save_every > 0 and (epoch % args.save_every == 0):
                _save_train_ckpt(out / f"epoch_video_{epoch:03d}.pt", epoch)

            if val_acc > (best_acc + args.min_delta):
                best_acc = val_acc
                no_improve = 0
                save_safetensors_checkpoint(
                    out / "best_video.safetensors",
                    {
                        "state_dict": model.state_dict(),
                        "img_size": args.img_size,
                        "frames": args.frames,
                        "threshold": 0.5,
                        "model_id": "temporal-video-detector",
                        "pretrained_backbone": bool(args.pretrained_backbone),
                    },
                )
                preferred_best = out / "best_video.safetensors"
                (out / "best_video_checkpoint.txt").write_text(str(preferred_best), encoding="utf-8")
                (out / "best_video_metrics.json").write_text(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "val_loss": float(val_loss),
                            "val_acc": float(val_acc),
                            "threshold": 0.5,
                            "skipped_batches": int(skipped_batches),
                            "lr": float(opt.param_groups[0]["lr"]),
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
        _save_train_ckpt(out / "interrupted_video.pt", interrupted_epoch)
        _save_train_ckpt(out / "last_video.pt", interrupted_epoch)
        print(f"training_interrupted saved={out / 'interrupted_video.pt'}")
        return

    best_release = out / "best_video.safetensors"
    if args.export_release and best_release.exists():
        rel = write_timestamped_release(
            out,
            ("config.json", "best_video_checkpoint.txt", "best_video_metrics.json"),
            preferred_artifact=best_release,
        )
        print(f"saved release bundle to {rel}")

    best_out = out / "best_video.safetensors"
    print(f"saved best temporal model to {best_out} best_acc={best_acc:.4f}")


def infer_main() -> None:
    ap = argparse.ArgumentParser(description="Temporal video detector inference")
    ap.add_argument("--model", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--unknown-margin", type=float, default=0.08)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(args.model, map_location=device)
    model = TemporalVideoDetector(pretrained_backbone=False).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    img_size = int(ckpt.get("img_size", 224))
    frames = int(ckpt.get("frames", 24))
    threshold = float(ckpt.get("threshold", 0.5))

    x = _sample_frames(args.video, frames, img_size, random_offset=False)
    x_t = torch.from_numpy(x).unsqueeze(0).to(device)

    with torch.no_grad():
        p = torch.sigmoid(model(x_t)).item()

    if abs(p - threshold) <= args.unknown_margin:
        label = "Unknown"
    else:
        label = "AI-generated" if p >= threshold else "Real"

    print(f"label={label} prob_ai={p:.6f} threshold={threshold:.3f} frames={frames}")


if __name__ == "__main__":
    infer_main()
