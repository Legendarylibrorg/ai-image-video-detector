from __future__ import annotations

from pathlib import Path
from typing import Any

from script_support import iter_member_dirs, read_json_dict, resolve_checkpoint


def _bool_metric(obj: dict[str, Any], key: str, default: bool = False) -> bool:
    value = obj.get(key, default)
    if isinstance(value, bool):
        return value
    return bool(value)


def _float_metric(obj: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in obj:
            value = obj.get(key, default)
            try:
                return float(value)
            except Exception:
                break
    return float(default)


def _use_metadata_features(config: dict[str, Any]) -> bool:
    args = config.get("args", {})
    if isinstance(args, dict):
        return bool(args.get("use_metadata_features", False))
    return False


def _selection_sort_key(item: dict[str, Any]) -> tuple[float | int, ...]:
    return (
        1 if item["has_test_metrics"] else 0,
        item["auc"],
        item["balanced_accuracy"],
        item["precision_ai"],
        item["recall_ai"],
        item["precision_real"],
        item["recall_real"],
        1 if item["use_metadata_features"] else 0,
    )


def public_model_candidates(ens_out: Path) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for model_dir in iter_member_dirs(ens_out):
        checkpoint = resolve_checkpoint(model_dir / "best.safetensors")
        if checkpoint is None:
            continue

        metrics = read_json_dict(model_dir / "best_metrics.json")
        test_metrics = read_json_dict(model_dir / "test_metrics.json")
        calibration = read_json_dict(model_dir / "calibration.json")
        config = read_json_dict(model_dir / "config.json")

        threshold = calibration.get("threshold")
        temperature = calibration.get("temperature")
        promotable = _bool_metric(metrics, "promotion_eligible", False)
        threshold_operable = _bool_metric(metrics, "threshold_operable", False)
        predicts_single_class = _bool_metric(metrics, "predicts_single_class", True)
        use_metadata_features = _use_metadata_features(config)

        candidates.append(
            {
                "name": model_dir.name,
                "artifact_dir": str(model_dir.resolve()),
                "source_checkpoint": str(checkpoint.resolve()),
                "source_checkpoint_path": checkpoint,
                "metrics": metrics,
                "test_metrics": test_metrics,
                "calibration": calibration,
                "config": config,
                "use_metadata_features": use_metadata_features,
                "promotion_eligible": promotable,
                "promotion_reason": metrics.get("promotion_reason"),
                "threshold_operable": threshold_operable,
                "predicts_single_class": predicts_single_class,
                "has_test_metrics": bool(test_metrics),
                "auc": _float_metric(test_metrics or metrics, "auc"),
                "balanced_accuracy": _float_metric(test_metrics or metrics, "balanced_accuracy"),
                "precision_ai": _float_metric(test_metrics or metrics, "precision", "precision_ai", default=0.0),
                "recall_ai": _float_metric(test_metrics or metrics, "recall", "recall_ai", default=0.0),
                "precision_real": _float_metric(test_metrics or metrics, "precision_real", default=0.0),
                "recall_real": _float_metric(test_metrics or metrics, "recall_real", default=0.0),
                "eligible_public": (
                    promotable
                    and threshold_operable
                    and not predicts_single_class
                    and threshold is not None
                    and temperature is not None
                    and bool(test_metrics)
                ),
            }
        )
    return candidates


def select_public_model(ens_out: Path) -> dict[str, Any] | None:
    candidates = [item for item in public_model_candidates(ens_out) if item.get("eligible_public")]
    if not candidates:
        return None

    candidates.sort(key=_selection_sort_key, reverse=True)
    chosen = dict(candidates[0])
    metadata_candidates_exist = any(bool(item["use_metadata_features"]) for item in candidates)
    if chosen["use_metadata_features"]:
        chosen["selection_reason"] = "best_promotable_metadata_aware_model"
    elif metadata_candidates_exist:
        chosen["selection_reason"] = "best_quality_promotable_pixel_model"
    else:
        chosen["selection_reason"] = "best_promotable_pixel_model"
    return chosen


def build_public_model_manifest(
    candidate: dict[str, Any] | None,
    *,
    public_checkpoint: str | None = None,
) -> dict[str, Any] | None:
    if not candidate:
        return None

    metrics = candidate.get("metrics", {})
    test_metrics = candidate.get("test_metrics", {})
    calibration = candidate.get("calibration", {})
    config = candidate.get("config", {})
    args = config.get("args", {}) if isinstance(config.get("args", {}), dict) else {}
    composite = metrics.get("composite_metrics", {}) if isinstance(metrics.get("composite_metrics", {}), dict) else {}
    test_composite = test_metrics.get("composite_metrics", {}) if isinstance(test_metrics.get("composite_metrics", {}), dict) else {}
    threshold = calibration.get("threshold")
    temperature = calibration.get("temperature")
    return {
        "member": candidate["name"],
        "selection_reason": candidate.get("selection_reason"),
        "source_checkpoint": candidate["source_checkpoint"],
        "public_checkpoint": public_checkpoint,
        "artifact_dir": candidate["artifact_dir"],
        "use_metadata_features": bool(candidate["use_metadata_features"]),
        "promotion_eligible": bool(candidate["promotion_eligible"]),
        "promotion_reason": candidate.get("promotion_reason"),
        "threshold_operable": bool(candidate["threshold_operable"]),
        "predicts_single_class": bool(candidate["predicts_single_class"]),
        "threshold": threshold,
        "temperature": temperature,
        "threshold_objective": calibration.get("objective") or metrics.get("threshold_objective"),
        "backbone": args.get("backbone"),
        "img_size": args.get("img_size"),
        "inference": {
            "input": {
                "image_mode": "RGB",
                "img_size": args.get("img_size"),
            },
            "calibration": {
                "threshold": threshold,
                "temperature": temperature,
            },
            "recommended_output": {
                "label": "ai|real|unknown",
                "prob_ai": "float",
                "confidence": "low|medium|high",
                "threshold": "float",
            },
            "example_output": {
                "label": "ai",
                "prob_ai": 0.97,
                "confidence": "high",
                "threshold": threshold,
            },
            "confidence_rule": "based_on_distance_from_threshold",
        },
        "validation_metrics": {
            "auc": metrics.get("auc"),
            "balanced_accuracy": metrics.get("balanced_accuracy"),
            "precision_ai": metrics.get("precision_ai"),
            "recall_ai": metrics.get("recall_ai"),
            "precision_real": metrics.get("precision_real"),
            "recall_real": metrics.get("recall_real"),
            "ece": composite.get("ece"),
            "brier": composite.get("brier"),
        },
        "test_metrics": {
            "auc": test_metrics.get("auc"),
            "balanced_accuracy": test_metrics.get("balanced_accuracy"),
            "precision_ai": test_metrics.get("precision_ai", test_metrics.get("precision")),
            "recall_ai": test_metrics.get("recall_ai", test_metrics.get("recall")),
            "precision_real": test_metrics.get("precision_real"),
            "recall_real": test_metrics.get("recall_real"),
            "ece": test_metrics.get("ece", test_composite.get("ece")),
            "brier": test_metrics.get("brier", test_composite.get("brier")),
        },
    }


def build_inference_profile(candidate: dict[str, Any] | None) -> dict[str, Any] | None:
    manifest = build_public_model_manifest(candidate)
    if manifest is None:
        return None
    return {
        "schema": "ai-image-detector-inference-profile-v1",
        "model": {
            "member": manifest["member"],
            "backbone": manifest.get("backbone"),
            "img_size": manifest.get("img_size"),
            "use_metadata_features": manifest.get("use_metadata_features"),
        },
        "calibration": manifest.get("inference", {}).get("calibration", {}),
        "recommended_output": manifest.get("inference", {}).get("recommended_output", {}),
        "example_output": manifest.get("inference", {}).get("example_output", {}),
        "classes": ["ai", "real"],
    }
