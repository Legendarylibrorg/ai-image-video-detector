from __future__ import annotations

"""Resource and path-safety limits for untrusted image/config inputs."""

import json
import math
import os
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

# Images: bound disk read and decompressed pixel count (DoS / zip-bomb mitigation).
MAX_IMAGE_FILE_BYTES = int(os.environ.get("AID_MAX_IMAGE_FILE_BYTES", str(50 * 1024 * 1024)))
MAX_IMAGE_PIXELS = int(os.environ.get("AID_MAX_IMAGE_PIXELS", str(89_478_485)))  # ~9500^2
MAX_EXIF_BYTES = int(os.environ.get("AID_MAX_EXIF_BYTES", str(256 * 1024)))

# Provenance: avoid scanning entire multi-GB blobs.
MAX_PROVENANCE_SCAN_BYTES = int(os.environ.get("AID_MAX_PROVENANCE_SCAN_BYTES", str(512 * 1024)))

# JSON configs (ensemble, domain, tools).
MAX_JSON_CONFIG_BYTES = int(os.environ.get("AID_MAX_JSON_CONFIG_BYTES", str(2 * 1024 * 1024)))

# Line-based text manifests (e.g. HF source id lists read by reporting scripts).
MAX_NONEMPTY_LINES_FILE_BYTES = int(
    os.environ.get("AID_MAX_NONEMPTY_LINES_FILE_BYTES", str(32 * 1024 * 1024))
)
MAX_NONEMPTY_LINES_COUNT = int(os.environ.get("AID_MAX_NONEMPTY_LINES_COUNT", str(500_000)))

# Video files (OpenCV decode DoS mitigation).
MAX_VIDEO_FILE_BYTES = int(os.environ.get("AID_MAX_VIDEO_FILE_BYTES", str(2 * 1024**3)))
MAX_VIDEO_DECODE_FRAMES = int(os.environ.get("AID_MAX_VIDEO_DECODE_FRAMES", str(500_000)))

# Safetensors string metadata (checkpoint sidecar JSON).
MAX_SAFETENSORS_METADATA_BYTES = int(os.environ.get("AID_MAX_SAFETENSORS_METADATA_BYTES", str(256 * 1024)))

# Full .safetensors checkpoint files on disk (DoS / TOCTOU mitigation; aligns with training .pt cap).
MAX_SAFETENSORS_FILE_BYTES = int(os.environ.get("AID_MAX_SAFETENSORS_FILE_BYTES", str(2 * 1024**3)))

_pil_limits_applied = False


def configure_pil_limits() -> None:
    global _pil_limits_applied
    if _pil_limits_applied:
        return
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    _pil_limits_applied = True


def check_file_size(path: str | Path, *, max_bytes: int = MAX_IMAGE_FILE_BYTES) -> int:
    p = Path(path)
    st = p.stat()
    if st.st_size > max_bytes:
        raise ValueError(f"file_too_large path={p} size={st.st_size} max={max_bytes}")
    return int(st.st_size)


def read_bytes_limited(path: str | Path, *, max_bytes: int = MAX_IMAGE_FILE_BYTES) -> bytes:
    p = Path(path)
    check_file_size(p, max_bytes=max_bytes)
    return p.read_bytes()


def path_must_be_under(path: str | Path, root: str | Path) -> Path:
    """Resolve path and require it lies under root (no path escape via symlinks)."""
    p = Path(path).expanduser()
    r = Path(root).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(p)
    resolved = p.resolve()
    try:
        resolved.relative_to(r)
    except ValueError as exc:
        raise ValueError(f"path_escapes_data_root path={path} root={root}") from exc
    return resolved


def reject_symlink(path: str | Path) -> Path:
    """Reject if the final path component is a symlink (mitigate swap/tofu attacks)."""
    p = Path(path)
    if p.is_symlink():
        raise ValueError(f"symlink_not_allowed path={p}")
    return p


def prepare_video_path(path: str | Path) -> Path:
    """Validate a video path before OpenCV: no symlink on leaf, bounded file size."""
    p = Path(path)
    reject_symlink(p)
    check_file_size(p, max_bytes=MAX_VIDEO_FILE_BYTES)
    return p


def open_image_rgb(
    path: str | Path,
    *,
    root: Path | None = None,
    allow_symlink: bool = False,
) -> "Image.Image":
    """Open image with size limits and optional dataset-root jail."""
    configure_pil_limits()
    from PIL import Image

    p = Path(path)
    if root is not None:
        reject_symlink(p)
        p = path_must_be_under(p, root)
    elif not allow_symlink:
        reject_symlink(p)
    check_file_size(p, max_bytes=MAX_IMAGE_FILE_BYTES)
    img = Image.open(p)
    try:
        return img.convert("RGB")
    finally:
        img.close()


def read_json_file_limited(path: str | Path, *, max_bytes: int = MAX_JSON_CONFIG_BYTES) -> dict[str, Any]:
    p = Path(path)
    reject_symlink(p)
    if not p.exists():
        return {}
    st = p.stat()
    if st.st_size == 0:
        return {}
    raw = read_bytes_limited(p, max_bytes=max_bytes)
    try:
        data = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"invalid_json_config path={p}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"json_config_must_be_object path={p}")
    return data


def validate_ensemble_config(cfg: dict[str, Any]) -> None:
    w = cfg.get("weights")
    if w is not None:
        if not isinstance(w, list) or not w:
            raise ValueError("ensemble.weights must be a non-empty list")
        for x in w:
            f = float(x)
            if f < 0 or f > 1e6 or not math.isfinite(f):
                raise ValueError("ensemble.weights must be finite non-negative")
    if "threshold" in cfg:
        t = float(cfg["threshold"])
        if t < 0.0 or t > 1.0 or not math.isfinite(t):
            raise ValueError("ensemble.threshold must be in [0,1]")
    if "temperature" in cfg:
        t = float(cfg["temperature"])
        if t < 1e-4 or t > 100.0 or not math.isfinite(t):
            raise ValueError("ensemble.temperature must be in [1e-4, 100]")


def validate_domain_config(cfg: dict[str, Any]) -> None:
    th = cfg.get("thresholds")
    if th is None:
        return
    if not isinstance(th, dict):
        raise ValueError("domain.thresholds must be an object")
    for k, v in th.items():
        if not isinstance(k, str):
            raise ValueError("domain.thresholds keys must be strings")
        f = float(v)
        if f < 0.0 or f > 1.0 or not math.isfinite(f):
            raise ValueError(f"domain.thresholds[{k!r}] must be in [0,1]")


def validate_tools_config(cfg: dict[str, Any]) -> None:
    for key in ("risk_bias", "prob_bias"):
        if key not in cfg:
            continue
        f = float(cfg[key])
        if f < -1.0 or f > 1.0 or not math.isfinite(f):
            raise ValueError(f"tools.{key} must be in [-1,1]")
