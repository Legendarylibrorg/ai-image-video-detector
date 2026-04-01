from __future__ import annotations

import io
import random
from pathlib import Path

from PIL import Image, ImageFilter
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

from .io_limits import open_image_rgb
from .metadata import extract_metadata_features, metadata_feature_dim
from .runtime import resolve_num_workers


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
    val_tf = make_eval_transform(img_size)
    return train_tf, val_tf


def make_eval_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])


def build_loader_kwargs(num_workers: int = 4, *, prefetch_factor: int = 4):
    workers = resolve_num_workers(num_workers)
    dl_kwargs = {
        "num_workers": workers,
        "pin_memory": bool(torch.cuda.is_available()),
    }
    if workers > 0:
        dl_kwargs["persistent_workers"] = True
        dl_kwargs["prefetch_factor"] = int(prefetch_factor)
    return dl_kwargs


def unpack_image_batch(
    batch: tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    if len(batch) == 3:
        x, metadata_features, y = batch
        return x, metadata_features, y
    x, y = batch
    return x, None, y


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


class MetadataImageFolder(datasets.ImageFolder):
    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        metadata_features = torch.tensor(extract_metadata_features(path), dtype=torch.float32)
        return sample, metadata_features, target


def make_jailed_rgb_loader(root_dir: Path):
    root_res = root_dir.resolve()

    def loader(path: str) -> Image.Image:
        return open_image_rgb(path, root=root_res)

    return loader


def make_loaders(
    data_root: str,
    img_size: int,
    batch_size: int,
    num_workers: int = 4,
    use_metadata_features: bool = False,
):
    root = Path(data_root)
    train_dir = root / "train"
    val_dir = root / "val"

    train_tf, val_tf = make_transforms(img_size)

    dataset_cls = MetadataImageFolder if use_metadata_features else datasets.ImageFolder
    train_loader_fn = make_jailed_rgb_loader(train_dir)
    val_loader_fn = make_jailed_rgb_loader(val_dir)
    train_ds = dataset_cls(train_dir, transform=train_tf, loader=train_loader_fn)
    val_ds = dataset_cls(val_dir, transform=val_tf, loader=val_loader_fn)

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
        metadata_feature_dim() if use_metadata_features else 0,
    )
