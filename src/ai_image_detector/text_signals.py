from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image


def analyze_text_signals(image: "Image.Image") -> dict[str, Any]:
    import numpy as np

    try:
        import cv2  # type: ignore
    except Exception:
        cv2 = None

    arr = np.asarray(image.convert("L"), dtype=np.uint8)
    h, w = arr.shape[:2]
    if h == 0 or w == 0:
        return {"text_score": 0.0, "text_flags": ["invalid_image"], "text_regions": 0}

    if cv2 is None:
        gy, gx = np.gradient(arr.astype(np.float32))
        edge_strength = np.hypot(gx, gy)
        edge_density = float((edge_strength > 20).mean())
        score = min(1.0, edge_density / 0.22)
        flags: list[str] = []
        if score > 0.65:
            flags.append("possible_text_overlay")
        return {"text_score": float(score), "text_flags": flags, "text_regions": 0}

    blur = cv2.GaussianBlur(arr, (3, 3), 0)
    grad_x = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    grad = cv2.convertScaleAbs(grad_x)

    _, bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 3))
    merged = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = 0
    text_pixels = 0
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if ch < 8 or cw < 18:
            continue
        area = cw * ch
        if area < 120:
            continue
        aspect = cw / max(ch, 1)
        if aspect < 1.2 or aspect > 28.0:
            continue
        fill_ratio = float(cv2.contourArea(cnt)) / max(area, 1)
        if fill_ratio < 0.08 or fill_ratio > 0.95:
            continue
        regions += 1
        text_pixels += area

    coverage = float(text_pixels) / float(max(h * w, 1))
    region_term = min(1.0, regions / 14.0)
    coverage_term = min(1.0, coverage / 0.22)
    score = float(min(1.0, 0.65 * region_term + 0.35 * coverage_term))

    flags: list[str] = []
    if regions >= 2:
        flags.append("text_regions_detected")
    if coverage >= 0.06:
        flags.append("heavy_text_overlay")
    elif coverage >= 0.025:
        flags.append("moderate_text_overlay")

    return {
        "text_score": score,
        "text_flags": flags,
        "text_regions": int(regions),
        "text_coverage": float(coverage),
    }
