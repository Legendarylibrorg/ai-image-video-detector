from __future__ import annotations

import io
from typing import Any


def _safe_text(v: Any) -> str:
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="ignore")
    return str(v)


def analyze_provenance(image_bytes: bytes) -> dict[str, Any]:
    from PIL import Image

    flags: list[str] = []
    score = 0.0

    blob = image_bytes.lower()
    if b"c2pa" in blob or b"content credentials" in blob:
        flags.append("has_content_credentials")
        # Presence of credentials lowers ambiguity.
        score = max(score - 0.1, 0.0)

    ai_markers = [
        b"stable diffusion",
        b"midjourney",
        b"dall-e",
        b"dall e",
        b"comfyui",
        b"automatic1111",
        b"firefly",
    ]
    if any(m in blob for m in ai_markers):
        flags.append("embedded_ai_tool_marker")
        score += 0.6

    try:
        img = Image.open(io.BytesIO(image_bytes))
        info = {str(k).lower(): _safe_text(v).lower() for k, v in img.info.items()}
        joined = " ".join([f"{k}:{v}" for k, v in info.items()])
        if "c2pa" in joined or "content credentials" in joined:
            if "has_content_credentials" not in flags:
                flags.append("has_content_credentials")
        if any(s in joined for s in ("stable diffusion", "midjourney", "dall", "firefly")):
            flags.append("metadata_ai_tool_marker")
            score += 0.35
    except Exception:
        flags.append("unreadable_provenance")
        score += 0.10

    score = float(max(0.0, min(1.0, score)))
    return {
        "provenance_score": score,
        "provenance_flags": sorted(set(flags)),
    }
