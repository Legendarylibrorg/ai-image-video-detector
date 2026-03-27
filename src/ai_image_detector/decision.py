from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image


def image_ood_score(image: "Image.Image") -> dict[str, Any]:
    import numpy as np

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
    text_score: float = 0.0,
) -> float:
    return float((0.84 * prob_ai) + (0.06 * metadata_score) + (0.05 * provenance_score) + (0.05 * text_score))


def decide_label(
    prob_ai: float,
    threshold: float,
    unknown_margin: float,
    ood_score: float,
    borderline_ood_score: float = 0.45,
    hard_ood_score: float = 0.80,
    ai_unknown_margin: float | None = None,
    real_unknown_margin: float | None = None,
) -> str:
    ai_margin = max(float(ai_unknown_margin if ai_unknown_margin is not None else unknown_margin), 0.0)
    real_margin = max(float(real_unknown_margin if real_unknown_margin is not None else unknown_margin), 0.0)
    borderline_ood_score = max(float(borderline_ood_score), 0.0)
    hard_ood_score = max(float(hard_ood_score), borderline_ood_score)

    if ood_score >= hard_ood_score:
        return "Unknown"
    if prob_ai >= threshold:
        if (prob_ai - threshold) <= ai_margin and ood_score >= borderline_ood_score:
            return "Unknown"
        return "AI-generated"
    if (threshold - prob_ai) <= real_margin and ood_score >= borderline_ood_score:
        return "Unknown"
    return "Real"
