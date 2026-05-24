from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
from typing import Any, Mapping

from .checkpoint_io import checkpoint_load_staging_enabled, materialize_checkpoint_file
from .io_limits import (
    MAX_SAFETENSORS_FILE_BYTES,
    MAX_SAFETENSORS_METADATA_BYTES,
    check_file_size,
    reject_symlink,
)

_MAX_TRAINING_CHECKPOINT_BYTES = int(
    os.environ.get("AID_MAX_TRAINING_CHECKPOINT_BYTES", str(2 * 1024**3))
)

_EXACT_SENSITIVE_ARG_KEYS = frozenset(
    {
        "api_key",
        "auth_token",
        "bearer_token",
        "credential",
        "hf_token",
        "huggingface_token",
        "openai_api_key",
        "password",
        "secret",
        "token",
        "wandb_api_key",
    }
)


def _is_sensitive_arg_key(key: str) -> bool:
    lk = key.lower()
    if lk in _EXACT_SENSITIVE_ARG_KEYS:
        return True
    if lk.endswith("_password") or lk.endswith("_secret"):
        return True
    if lk.endswith("_token"):
        return True
    if "_api_key" in lk:
        return True
    return False


def args_dict_for_checkpoint(args: Any) -> dict[str, Any]:
    """Snapshot CLI args for JSON/checkpoints: primitives only, no obvious secret keys."""
    raw = vars(args) if hasattr(args, "__dict__") else dict(args)
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if _is_sensitive_arg_key(key):
            continue
        out[key] = _primitive_for_checkpoint(value)
    return out


def _primitive_for_checkpoint(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, (list, tuple)):
        return [_primitive_for_checkpoint(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _primitive_for_checkpoint(v) for k, v in value.items()}
    return str(value)


def _torch():
    import torch

    return torch


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    try:
        torch = _torch()
        if isinstance(value, torch.device):
            return str(value)
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)

def save_safetensors_checkpoint(path: str | Path, checkpoint: Mapping[str, Any]) -> None:
    from safetensors.torch import save_file
    torch = _torch()

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
    tmp = out.with_name(out.name + f".tmp.{os.getpid()}")
    try:
        save_file(tensors, str(tmp), metadata=meta)
        os.replace(tmp, out)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def save_training_checkpoint(
    path: str | Path,
    checkpoint: Mapping[str, Any],
    *,
    latest_name: str = "latest_checkpoint.txt",
) -> None:
    torch = _torch()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + f".tmp.{os.getpid()}")
    try:
        torch.save(dict(checkpoint), tmp)
        os.replace(tmp, out)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    latest = out.parent / latest_name
    tmp_latest = latest.with_name(latest.name + f".tmp.{os.getpid()}")
    try:
        tmp_latest.write_text(out.name, encoding="utf-8")
        os.replace(tmp_latest, latest)
    finally:
        if tmp_latest.exists():
            try:
                tmp_latest.unlink()
            except OSError:
                pass


def load_safetensors_checkpoint(path: str | Path, map_location: Any = None) -> dict[str, Any]:
    in_path = Path(path)
    staged: Path | None = None
    try:
        if checkpoint_load_staging_enabled():
            staged = materialize_checkpoint_file(in_path, max_bytes=MAX_SAFETENSORS_FILE_BYTES)
            load_path = staged
        else:
            reject_symlink(in_path)
            check_file_size(in_path, max_bytes=MAX_SAFETENSORS_FILE_BYTES)
            load_path = in_path

        from safetensors import safe_open
        from safetensors.torch import load_file

        tensors = load_file(str(load_path), device="cpu")
        with safe_open(str(load_path), framework="pt", device="cpu") as f:
            metadata = f.metadata() or {}

        meta_text = metadata.get("checkpoint_meta", "{}") or "{}"
        meta: dict[str, Any] = {}
        meta_bytes = len(meta_text.encode("utf-8"))
        if meta_bytes > MAX_SAFETENSORS_METADATA_BYTES:
            raise ValueError(
                f"checkpoint_metadata_too_large path={in_path} bytes={meta_bytes} "
                f"max={MAX_SAFETENSORS_METADATA_BYTES}"
            )
        if not meta_text or meta_text.strip() in ("", "{}"):
            meta = {}
        else:
            try:
                parsed = json.loads(meta_text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid_checkpoint_metadata path={in_path}") from exc
            if not isinstance(parsed, dict):
                raise ValueError(f"invalid_checkpoint_metadata path={in_path} expected_object=1")
            meta = parsed

        if map_location is not None:
            target = str(map_location)
            if target != "cpu":
                tensors = {k: v.to(target) for k, v in tensors.items()}

        out = dict(meta)
        out["state_dict"] = tensors
        out["_checkpoint_format"] = "safetensors"
        return out
    finally:
        if staged is not None:
            try:
                staged.unlink(missing_ok=True)
            except OSError:
                pass


def load_checkpoint(path: str | Path, map_location: Any = None) -> dict[str, Any]:
    in_path = Path(path)
    if in_path.suffix.lower() == ".safetensors":
        return load_safetensors_checkpoint(in_path, map_location=map_location)
    raise RuntimeError("unsupported_checkpoint_format path={} use .safetensors".format(in_path))


def load_training_checkpoint(path: str | Path, map_location: Any = None) -> dict[str, Any]:
    """Load a `.pt` training checkpoint saved by `save_training_checkpoint`.

    Uses ``weights_only=True`` when supported (PyTorch 2.2+). Arbitrary pickle gadgets are
    not loaded; use checkpoints produced by this codebase or re-export legacy weights.
    """
    torch = _torch()
    p = Path(path)
    staged: Path | None = None
    try:
        if checkpoint_load_staging_enabled():
            staged = materialize_checkpoint_file(p, max_bytes=_MAX_TRAINING_CHECKPOINT_BYTES)
            load_path = staged
        else:
            reject_symlink(p)
            check_file_size(p, max_bytes=_MAX_TRAINING_CHECKPOINT_BYTES)
            load_path = p

        kwargs: dict[str, Any] = {"map_location": map_location}
        if "weights_only" in inspect.signature(torch.load).parameters:
            kwargs["weights_only"] = True

        try:
            return torch.load(load_path, **kwargs)
        except Exception as first_exc:
            raise RuntimeError(
                "training_checkpoint_load_failed path={} "
                "(not a weights_only-safe checkpoint; re-save with current save_training_checkpoint "
                "or load weights via safetensors export)".format(p)
            ) from first_exc
    finally:
        if staged is not None:
            try:
                staged.unlink(missing_ok=True)
            except OSError:
                pass
