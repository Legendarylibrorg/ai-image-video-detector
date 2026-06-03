from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .decision import combined_risk, decide_label
from .risk_tools import apply_risk_tools


@dataclass(frozen=True)
class DecisionOptions:
    unknown_margin: float
    unknown_margin_ai: float
    unknown_margin_real: float
    borderline_ood_threshold: float
    hard_ood_threshold: float
    tta_views: int


@dataclass(frozen=True)
class ModelReport:
    model_ids: list[str]
    weights: list[float]
    temperature: float
    ensemble_config: str = ""


@dataclass(frozen=True)
class ConfigReport:
    domain_config: str = ""
    tools_config: str = ""


def build_inference_report(
    *,
    prob_ai: float,
    threshold: float,
    metadata: dict[str, Any],
    provenance: dict[str, Any],
    text: dict[str, Any],
    ood: dict[str, Any],
    domain: str,
    decision: DecisionOptions,
    model: ModelReport,
    config: ConfigReport,
    tools_cfg: dict[str, Any],
) -> dict[str, Any]:
    metadata_score = float(metadata["metadata_score"])
    provenance_score = float(provenance["provenance_score"])
    text_score = float(text["text_score"])
    ood_score = float(ood["ood_score"])
    c_risk = combined_risk(prob_ai, metadata_score, provenance_score, text_score)
    adjusted = apply_risk_tools(
        prob_ai=prob_ai,
        combined_risk=c_risk,
        metadata_flags=metadata["metadata_flags"],
        ood_flags=ood["ood_flags"],
        text_flags=text["text_flags"],
        cfg=tools_cfg,
    )
    adjusted_prob_ai = float(adjusted["prob_ai"])
    adjusted_risk = float(adjusted["combined_risk"])
    label = decide_label(
        adjusted_prob_ai,
        threshold,
        decision.unknown_margin,
        ood_score,
        borderline_ood_score=decision.borderline_ood_threshold,
        hard_ood_score=decision.hard_ood_threshold,
        ai_unknown_margin=decision.unknown_margin_ai,
        real_unknown_margin=decision.unknown_margin_real,
    )

    return {
        "label": label,
        "prob_ai": adjusted_prob_ai,
        "threshold": float(threshold),
        "unknown_margin": float(decision.unknown_margin),
        "unknown_margin_ai": float(decision.unknown_margin_ai),
        "unknown_margin_real": float(decision.unknown_margin_real),
        "borderline_ood_threshold": float(decision.borderline_ood_threshold),
        "hard_ood_threshold": float(decision.hard_ood_threshold),
        "combined_risk": adjusted_risk,
        "metadata_score": metadata_score,
        "metadata_flags": metadata["metadata_flags"],
        "metadata_fields": metadata["metadata_fields"],
        "provenance_score": provenance_score,
        "provenance_flags": provenance["provenance_flags"],
        "text_score": text_score,
        "text_flags": text["text_flags"],
        "text_regions": int(text.get("text_regions", 0)),
        "ood_score": ood_score,
        "ood_flags": ood["ood_flags"],
        "model_ids": model.model_ids,
        "model_count": len(model.model_ids),
        "temperature": float(model.temperature),
        "ensemble_weights": [float(w) for w in model.weights],
        "ensemble_config": model.ensemble_config or None,
        "domain": domain,
        "domain_config": config.domain_config or None,
        "tools_config": config.tools_config or None,
        "tool_adjustments": adjusted["tool_adjustments"],
        "tta_views": int(max(decision.tta_views, 1)),
    }
