from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image


DOMAIN_NAMES = ["photo", "screenshot", "document", "art_cgi", "face"]


def classify_domain(image: "Image.Image", text_score: float = 0.0) -> str:
    import numpy as np

    w, h = image.size
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    gray = np.asarray(image.convert("L"), dtype=np.float32)

    if min(w, h) <= 0:
        return "photo"

    # Approximate saturation stats.
    rgb = arr / 255.0
    cmax = rgb.max(axis=2)
    cmin = rgb.min(axis=2)
    sat = np.where(cmax > 1e-6, (cmax - cmin) / np.maximum(cmax, 1e-6), 0.0)
    sat_mean = float(np.mean(sat))

    # Edge density for UI/text-like content.
    gy, gx = np.gradient(gray)
    edge = np.hypot(gx, gy)
    edge_density = float((edge > 20).mean())

    aspect = max(float(w) / max(h, 1), float(h) / max(w, 1))

    # Rule-based routing for robust threshold specialization.
    if text_score >= 0.62 and edge_density > 0.17:
        return "screenshot"
    if text_score >= 0.72:
        return "document"
    if sat_mean > 0.44 and edge_density < 0.18:
        return "art_cgi"
    if aspect > 1.75 and text_score > 0.38:
        return "screenshot"
    return "photo"


def load_domain_config(path: str = "") -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    from .io_limits import read_json_file_limited, validate_domain_config

    cfg = read_json_file_limited(p)
    validate_domain_config(cfg)
    return cfg


def resolve_domain_threshold(base_threshold: float, domain: str, cfg: dict[str, Any]) -> float:
    per = cfg.get("thresholds", {})
    if not isinstance(per, dict):
        return float(base_threshold)
    if domain in per:
        return float(per[domain])
    return float(base_threshold)
