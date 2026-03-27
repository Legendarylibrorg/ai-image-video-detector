from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def resolve_checkpoint_path(preferred_pt_path: str | Path) -> Path:
    preferred = Path(preferred_pt_path)
    sft = preferred.with_suffix(".safetensors")
    if sft.exists():
        return sft
    if preferred.exists():
        return preferred
    return preferred


def save_safetensors_checkpoint(path: str | Path, checkpoint: Mapping[str, Any]) -> None:
    from safetensors.torch import save_file

    out = Path(path)
    if "state_dict" not in checkpoint:
        raise ValueError("checkpoint must include 'state_dict'")
    state_dict = checkpoint["state_dict"]
    if not isinstance(state_dict, Mapping):
        raise ValueError("checkpoint['state_dict'] must be a mapping")

    tensors: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if not torch.is_tensor(v):
            raise TypeError(f"state_dict[{k!r}] is not a tensor")
        tensors[str(k)] = v.detach().cpu().contiguous()

    meta_obj = {k: v for k, v in checkpoint.items() if k != "state_dict"}
    meta = {
        "format": "ai-image-detector-checkpoint-v1",
        "checkpoint_meta": json.dumps(meta_obj, default=_json_default),
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out), metadata=meta)


def load_safetensors_checkpoint(path: str | Path, map_location: torch.device | str | None = None) -> dict[str, Any]:
    from safetensors import safe_open
    from safetensors.torch import load_file

    in_path = Path(path)
    tensors = load_file(str(in_path), device="cpu")
    with safe_open(str(in_path), framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}

    meta_text = metadata.get("checkpoint_meta", "{}")
    meta: dict[str, Any] = {}
    try:
        parsed = json.loads(meta_text) if meta_text else {}
        if isinstance(parsed, dict):
            meta = parsed
    except Exception:
        meta = {}

    if map_location is not None:
        target = str(map_location)
        if target != "cpu":
            tensors = {k: v.to(target) for k, v in tensors.items()}

    out = dict(meta)
    out["state_dict"] = tensors
    out["_checkpoint_format"] = "safetensors"
    return out


def load_checkpoint(path: str | Path, map_location: torch.device | str | None = None) -> dict[str, Any]:
    in_path = Path(path)
    if in_path.suffix.lower() == ".safetensors":
        return load_safetensors_checkpoint(in_path, map_location=map_location)
    if os.environ.get("ALLOW_UNSAFE_PICKLE_CHECKPOINTS", "0") != "1":
        raise RuntimeError(
            "pickle_checkpoint_blocked path={} use .safetensors or set ALLOW_UNSAFE_PICKLE_CHECKPOINTS=1".format(in_path)
        )
    return torch.load(in_path, map_location=map_location)
