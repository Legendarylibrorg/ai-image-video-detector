from __future__ import annotations

import io
import random
from pathlib import Path

from PIL import Image, ImageFilter
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms


class RandomJpegCompression:
    def __init__(self, p: float = 0.35, quality_range: tuple[int, int] = (35, 95)):
        self.p = p
        self.quality_range = quality_range

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        q = random.randint(*self.quality_range)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class RandomResizeRoundtrip:
    def __init__(self, p: float = 0.35, scale_range: tuple[float, float] = (0.55, 0.95)):
        self.p = p
        self.scale_range = scale_range

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        w, h = img.size
        s = random.uniform(*self.scale_range)
        nw = max(16, int(w * s))
        nh = max(16, int(h * s))
        return img.resize((nw, nh), Image.BILINEAR).resize((w, h), Image.BILINEAR)


class RandomBlur:
    def __init__(self, p: float = 0.25, radius_range: tuple[float, float] = (0.3, 1.2)):
        self.p = p
        self.radius_range = radius_range

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        radius = random.uniform(*self.radius_range)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))


def make_transforms(img_size: int):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply(
            [transforms.RandomAffine(degrees=5, translate=(0.02, 0.02), scale=(0.95, 1.05))],
            p=0.2,
        ),
        transforms.RandomApply(
            [transforms.RandomPerspective(distortion_scale=0.15, p=1.0)],
            p=0.15,
        ),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
        transforms.RandomGrayscale(p=0.05),
        RandomJpegCompression(p=0.35),
        RandomResizeRoundtrip(p=0.35),
        RandomBlur(p=0.25),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.12), ratio=(0.3, 3.3), value="random"),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    return train_tf, val_tf


def build_loader_kwargs(num_workers: int = 4):
    dl_kwargs = {
        "num_workers": num_workers,
        "pin_memory": bool(torch.cuda.is_available()),
    }
    if num_workers > 0:
        dl_kwargs["persistent_workers"] = True
        dl_kwargs["prefetch_factor"] = 4
    return dl_kwargs


def build_weighted_sampler(targets: list[int], classes: list[str]):
    class_counts = [0 for _ in classes]
    for t in targets:
        class_counts[int(t)] += 1
    if any(c == 0 for c in class_counts):
        raise ValueError(f"Empty class detected in train split: counts={class_counts}")
    class_weights = [1.0 / float(c) for c in class_counts]
    sample_weights = [class_weights[int(t)] for t in targets]
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler, class_counts, class_weights


def make_loaders(data_root: str, img_size: int, batch_size: int, num_workers: int = 4):
    root = Path(data_root)
    train_dir = root / "train"
    val_dir = root / "val"

    train_tf, val_tf = make_transforms(img_size)

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tf)

    train_targets = list(train_ds.targets)
    sampler, class_counts, class_weights = build_weighted_sampler(train_targets, train_ds.classes)
    dl_kwargs = build_loader_kwargs(num_workers=num_workers)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        **dl_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        **dl_kwargs,
    )
    train_distribution = {name: int(class_counts[idx]) for idx, name in enumerate(train_ds.classes)}
    val_distribution = {name: int(sum(1 for t in val_ds.targets if int(t) == idx)) for idx, name in enumerate(val_ds.classes)}
    class_weight_map = {name: float(class_weights[idx]) for idx, name in enumerate(train_ds.classes)}
    return (
        train_loader,
        val_loader,
        train_ds.classes,
        train_ds.class_to_idx,
        val_ds.samples,
        train_distribution,
        val_distribution,
        class_weight_map,
    )
