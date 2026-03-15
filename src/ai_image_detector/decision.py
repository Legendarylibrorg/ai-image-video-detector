from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


def image_ood_score(image: Image.Image) -> dict[str, Any]:
    w, h = image.size
    flags: list[str] = []
    score = 0.0

    if min(w, h) < 128:
        score += 0.35
        flags.append("very_low_resolution")

    aspect = max(w / max(h, 1), h / max(w, 1))
    if aspect > 3.0:
        score += 0.25
        flags.append("extreme_aspect_ratio")

    g = np.asarray(image.convert("L"), dtype=np.float32)
    # Lightweight blur heuristic via pixel-difference variance.
    lap = (
        -4 * g
        + np.roll(g, 1, axis=0)
        + np.roll(g, -1, axis=0)
        + np.roll(g, 1, axis=1)
        + np.roll(g, -1, axis=1)
    )
    sharpness = float(np.var(lap))
    if sharpness < 8.0:
        score += 0.2
        flags.append("very_blurry")

    if sharpness > 4000.0:
        score += 0.2
        flags.append("oversharpened_or_noisy")

    score = min(1.0, score)
    return {"ood_score": score, "ood_flags": flags, "sharpness": sharpness}


def combined_risk(
    prob_ai: float,
    metadata_score: float = 0.0,
    provenance_score: float = 0.0,
) -> float:
    return float((0.75 * prob_ai) + (0.15 * metadata_score) + (0.10 * provenance_score))


def decide_label(
    prob_ai: float,
    threshold: float,
    unknown_margin: float,
    ood_score: float,
) -> str:
    if abs(prob_ai - threshold) <= unknown_margin:
        return "Unknown"
    if ood_score >= 0.70:
        return "Unknown"
    return "AI-generated" if prob_ai >= threshold else "Real"
