from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_tools_config(path: str = "") -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def apply_risk_tools(
    prob_ai: float,
    combined_risk: float,
    metadata_flags: list[str] | None = None,
    ood_flags: list[str] | None = None,
    text_flags: list[str] | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = cfg or {}
    out_prob = float(prob_ai)
    out_risk = float(combined_risk)
    reasons: list[str] = []

    meta = set(metadata_flags or [])
    ood = set(ood_flags or [])
    txt = set(text_flags or [])

    # Rule overrides.
    if {"edited_with_software_tag", "embedded_software_tag"} & meta and "oversharpened_or_noisy" in ood:
        out_risk = min(1.0, out_risk + 0.08)
        reasons.append("rule_meta_plus_ood")
    if "heavy_text_overlay" in txt and "extreme_aspect_ratio" in ood:
        out_risk = min(1.0, out_risk + 0.05)
        reasons.append("rule_text_overlay_combo")

    # Configurable nudges.
    risk_bias = float(cfg.get("risk_bias", 0.0))
    prob_bias = float(cfg.get("prob_bias", 0.0))
    if risk_bias != 0.0:
        out_risk = max(0.0, min(1.0, out_risk + risk_bias))
        reasons.append("cfg_risk_bias")
    if prob_bias != 0.0:
        out_prob = max(0.0, min(1.0, out_prob + prob_bias))
        reasons.append("cfg_prob_bias")

    return {
        "prob_ai": float(out_prob),
        "combined_risk": float(out_risk),
        "tool_adjustments": reasons,
    }
