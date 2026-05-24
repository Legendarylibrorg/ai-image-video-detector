from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from .checkpoints import load_checkpoint
from .io_limits import read_json_file_limited, validate_ensemble_config
from .metadata import extract_metadata_features
from .model import AdvancedAIDetector, build_model


@dataclass
class LoadedModels:
    models: list[AdvancedAIDetector]
    img_size: int
    img_sizes: list[int]
    threshold: float
    temperature: float
    model_ids: list[str]
    metadata_feature_dims: list[int]
    uses_metadata_features: bool
    model_temperatures: list[float]
    model_thresholds: list[float]
    weights: list[float]


def _resize_for_model(x: torch.Tensor, img_size: int) -> torch.Tensor:
    if x.shape[-2:] == (img_size, img_size):
        return x
    return F.interpolate(
        x,
        size=(img_size, img_size),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )


def _run_model(
    model: torch.nn.Module,
    x: torch.Tensor,
    metadata_features: torch.Tensor | None = None,
) -> torch.Tensor:
    if metadata_features is None:
        return model(x)
    try:
        return model(x, metadata_features=metadata_features)
    except TypeError as exc:
        if "metadata_features" not in str(exc):
            raise
        return model(x)


def metadata_features_from_paths(
    paths: list[str] | tuple[str, ...],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    features = [extract_metadata_features(str(path)) for path in paths]
    return torch.tensor(features, dtype=dtype, device=device)


def stack_model_logits(
    models: list[torch.nn.Module],
    img_sizes: list[int],
    x: torch.Tensor,
    metadata_features: torch.Tensor | None = None,
) -> torch.Tensor:
    if len(models) != len(img_sizes):
        raise ValueError(f"img_sizes count {len(img_sizes)} != models count {len(models)}")

    logits: list[torch.Tensor] = []
    for model, img_size in zip(models, img_sizes):
        model_x = _resize_for_model(x, img_size)
        if model_x.device.type == "cuda":
            model_x = model_x.contiguous(memory_format=torch.channels_last)
        logits.append(_run_model(model, model_x, metadata_features=metadata_features))
    return torch.stack(logits, dim=0)


class EnsembleDetector(torch.nn.Module):
    def __init__(
        self,
        models: list[torch.nn.Module],
        weights: list[float] | None = None,
        img_sizes: list[int] | None = None,
    ):
        super().__init__()
        if not models:
            raise ValueError("EnsembleDetector requires at least one model")
        self.models = torch.nn.ModuleList(models)
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        if img_sizes is None:
            img_sizes = [0] * len(models)
        if len(weights) != len(models):
            raise ValueError(f"weights count {len(weights)} != models count {len(models)}")
        if len(img_sizes) != len(models):
            raise ValueError(f"img_sizes count {len(img_sizes)} != models count {len(models)}")
        w = torch.tensor(weights, dtype=torch.float32)
        w = w / torch.clamp(w.sum(), min=1e-8)
        self.register_buffer("weights", w)
        self.img_sizes = [int(s) for s in img_sizes]

    def forward(self, x: torch.Tensor, metadata_features: torch.Tensor | None = None) -> torch.Tensor:
        effective_sizes = [
            int(size) if int(size) > 0 else int(x.shape[-1])
            for size in self.img_sizes
        ]
        logits = stack_model_logits(list(self.models), effective_sizes, x, metadata_features=metadata_features)
        w = self.weights.view(-1, 1)
        return (logits * w).sum(dim=0)


def _resolve_model_weights(
    model_paths: list[str],
    weights_cfg: dict,
) -> list[float]:
    raw_weights = weights_cfg.get("weights")
    if not isinstance(raw_weights, list) or not raw_weights:
        raise ValueError("ensemble config must include non-empty 'weights' list")

    cfg_paths = weights_cfg.get("model_paths")
    if isinstance(cfg_paths, list) and len(cfg_paths) == len(raw_weights):
        cfg_map = {str(Path(p).resolve()): float(w) for p, w in zip(cfg_paths, raw_weights)}
        resolved = [str(Path(p).resolve()) for p in model_paths]
        if all(p in cfg_map for p in resolved):
            out = [cfg_map[p] for p in resolved]
            s = float(sum(out))
            if s <= 0:
                raise ValueError("ensemble config weights must sum to positive value")
            return [float(x / s) for x in out]

    if len(raw_weights) != len(model_paths):
        raise ValueError(
            f"ensemble config weights length {len(raw_weights)} does not match model count {len(model_paths)}"
        )
    out = [float(w) for w in raw_weights]
    s = float(sum(out))
    if s <= 0:
        raise ValueError("ensemble config weights must sum to positive value")
    return [float(x / s) for x in out]


def _load_single(path: str, device: torch.device):
    ckpt = load_checkpoint(path, map_location=device)
    backbone = str(ckpt.get("backbone", "tiny"))
    metadata_dim = int(ckpt.get("metadata_feature_dim", 0))
    model = build_model(backbone=backbone, pretrained_backbone=False, metadata_feature_dim=metadata_dim).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    img_size = int(ckpt.get("img_size", 256))
    threshold = float(ckpt.get("threshold", 0.5))
    temperature = float(ckpt.get("temperature", 1.0))
    model_id = str(ckpt.get("model_id", Path(path).stem))
    return model, img_size, threshold, temperature, model_id, metadata_dim


def load_models(model_paths: list[str], device: torch.device, ensemble_config: str = "") -> LoadedModels:
    if not model_paths:
        raise ValueError("At least one model path is required")

    from .collection_paths import require_under_collection_workspace

    resolved_paths = [str(require_under_collection_workspace(p)) for p in model_paths]
    loaded = [_load_single(p, device) for p in resolved_paths]
    models = [x[0] for x in loaded]
    img_sizes = [x[1] for x in loaded]
    thresholds = [x[2] for x in loaded]
    temps = [x[3] for x in loaded]
    model_ids = [x[4] for x in loaded]
    metadata_dims = [int(x[5]) for x in loaded]
    weights = [1.0 / len(loaded)] * len(loaded)
    threshold = float(sum(thresholds) / len(thresholds))
    temperature = float(sum(temps) / len(temps))

    if ensemble_config:
        from .collection_paths import resolve_workspace_json_config

        cfg_path = resolve_workspace_json_config(ensemble_config)
        cfg = read_json_file_limited(cfg_path)
        validate_ensemble_config(cfg)
        weights = _resolve_model_weights(resolved_paths, cfg)
        if "threshold" in cfg:
            threshold = float(cfg["threshold"])
        if "temperature" in cfg:
            temperature = float(cfg["temperature"])

    return LoadedModels(
        models=models,
        img_size=max(img_sizes),
        img_sizes=[int(s) for s in img_sizes],
        threshold=threshold,
        temperature=temperature,
        model_ids=model_ids,
        metadata_feature_dims=metadata_dims,
        uses_metadata_features=any(dim > 0 for dim in metadata_dims),
        model_temperatures=[float(t) for t in temps],
        model_thresholds=[float(t) for t in thresholds],
        weights=[float(w) for w in weights],
    )
