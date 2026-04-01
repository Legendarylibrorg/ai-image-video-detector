from __future__ import annotations

import os
from collections.abc import Iterable

import numpy as np
import torch
from torch.optim import AdamW

from .io_limits import configure_pil_limits
from .utils import git_commit


def seed_all(seed: int) -> None:
    value = int(seed)
    import random

    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def training_device() -> torch.device:
    """Resolve device for training and inference: CUDA if available, else Apple MPS, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_num_workers(num_workers: int = 4) -> int:
    workers = int(num_workers)
    if workers >= 0:
        return workers
    cpu = os.cpu_count() or 8
    return min(12, max(4, cpu // 2))


def configure_torch_runtime(device: torch.device, deterministic: bool) -> None:
    configure_pil_limits()
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
        return
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def build_adamw(
    parameters: Iterable[torch.nn.Parameter],
    *,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> AdamW:
    params = list(parameters)
    kwargs = {"lr": float(lr), "weight_decay": float(weight_decay)}
    if device.type == "cuda":
        try:
            return AdamW(params, fused=True, **kwargs)
        except Exception:
            pass
    return AdamW(params, **kwargs)
