from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .model import AdvancedAIDetector, build_model


@dataclass
class LoadedModels:
    models: list[AdvancedAIDetector]
    img_size: int
    threshold: float
    temperature: float
    model_ids: list[str]


class EnsembleDetector(torch.nn.Module):
    def __init__(self, models: list[torch.nn.Module]):
        super().__init__()
        self.models = torch.nn.ModuleList(models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = [m(x) for m in self.models]
        return torch.stack(logits, dim=0).mean(dim=0)


def _load_single(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    backbone = str(ckpt.get("backbone", "tiny"))
    model = build_model(backbone=backbone, pretrained_backbone=False).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    img_size = int(ckpt.get("img_size", 256))
    threshold = float(ckpt.get("threshold", 0.5))
    temperature = float(ckpt.get("temperature", 1.0))
    model_id = str(ckpt.get("model_id", Path(path).stem))
    return model, img_size, threshold, temperature, model_id


def load_models(model_paths: list[str], device: torch.device) -> LoadedModels:
    if not model_paths:
        raise ValueError("At least one model path is required")

    loaded = [_load_single(p, device) for p in model_paths]
    models = [x[0] for x in loaded]
    img_sizes = [x[1] for x in loaded]
    thresholds = [x[2] for x in loaded]
    temps = [x[3] for x in loaded]
    model_ids = [x[4] for x in loaded]

    if len(set(img_sizes)) != 1:
        raise ValueError(f"All models must use same img_size, got {img_sizes}")

    return LoadedModels(
        models=models,
        img_size=img_sizes[0],
        threshold=float(sum(thresholds) / len(thresholds)),
        temperature=float(sum(temps) / len(temps)),
        model_ids=model_ids,
    )
