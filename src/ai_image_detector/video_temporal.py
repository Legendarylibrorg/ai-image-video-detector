from __future__ import annotations

import argparse
import os
from pathlib import Path
import random
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import models


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def _collect_videos(root: Path) -> List[Tuple[str, int]]:
    classes = {"real": 0, "ai": 1}
    out: List[Tuple[str, int]] = []
    for cls, y in classes.items():
        d = root / cls
        if not d.exists():
            continue
        for p in d.rglob("*"):
            if p.suffix.lower() in VIDEO_EXTS:
                out.append((str(p), y))
    return out


def _sample_frames(path: str, num_frames: int, size: int, random_offset: bool) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"unable to open video: {path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total = max(total, 1)

    if total <= num_frames:
        indices = list(range(total))
        while len(indices) < num_frames:
            indices.append(indices[-1] if indices else 0)
    else:
        stride = total / float(num_frames)
        base = random.random() * stride if random_offset else 0.0
        indices = [int(min(total - 1, base + i * stride)) for i in range(num_frames)]

    frames = []
    idx_set = set(indices)
    current = 0
    target_pos = {i: [] for i in idx_set}
    for pos, idx in enumerate(indices):
        target_pos[idx].append(pos)

    out = [None] * len(indices)
    while True:
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
    def __init__(self, hidden: int = 384):
        super().__init__()
        base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
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
        # x: [B, T, C, H, W]
        b, t, c, h, w = x.shape
        flat = x.view(b * t, c, h, w)
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
            if device.type == "cuda":
                x = x.to(memory_format=torch.channels_last)
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
    args = ap.parse_args()

    root = Path(args.data)
    train_samples = _collect_videos(root / "train")
    val_samples = _collect_videos(root / "val")
    if not train_samples or not val_samples:
        raise RuntimeError("Expected videos in train/ai, train/real, val/ai, val/real")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_ds = VideoDataset(train_samples, args.frames, args.img_size, augment=True)
    val_ds = VideoDataset(val_samples, args.frames, args.img_size, augment=False)

    if args.num_workers <= 0:
        cpu = os.cpu_count() or 8
        args.num_workers = min(12, max(4, cpu // 2))

    dl_kwargs = {"num_workers": args.num_workers, "pin_memory": True}
    if args.num_workers > 0:
        dl_kwargs["persistent_workers"] = True
        dl_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **dl_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **dl_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    model = TemporalVideoDetector().to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    if args.compile:
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as exc:
            print(f"compile_disabled reason={exc}")
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=max(args.epochs, 1), eta_min=args.lr * 0.1)
    loss_fn = nn.BCEWithLogitsLoss()
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    grad_accum = max(1, int(args.grad_accum))

    best_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        step_idx = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if device.type == "cuda":
                x = x.to(memory_format=torch.channels_last)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logit = model(x)
                loss = loss_fn(logit, y) / grad_accum
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

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "img_size": args.img_size,
                    "frames": args.frames,
                    "threshold": 0.5,
                    "model_id": "temporal-video-detector",
                },
                out / "best_video.pt",
            )

    print(f"saved best temporal model to {out / 'best_video.pt'} best_acc={best_acc:.4f}")


def infer_main() -> None:
    ap = argparse.ArgumentParser(description="Temporal video detector inference")
    ap.add_argument("--model", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--unknown-margin", type=float, default=0.08)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.model, map_location=device)
    model = TemporalVideoDetector().to(device)
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
    # Backward-compatible default: inference when run directly.
    infer_main()
