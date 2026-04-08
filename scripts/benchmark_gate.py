from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from release_selection import select_public_model
from script_support import read_json_dict


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return read_json_dict(path)


def _metric(obj: dict, *names: str, default: float | None = None) -> float:
    for name in names:
        if name in obj:
            return float(obj[name])
    if default is None:
        raise KeyError(names[0] if names else "metric")
    return float(default)


def _check_model_metrics(
    *,
    prefix: str,
    metrics: dict,
    min_auc: float,
    min_f1: float,
    min_precision: float,
    min_recall: float,
    max_ece: float,
    max_brier: float,
    checks: dict[str, float | str],
    failures: list[str],
) -> None:
    auc = _metric(metrics, "auc", default=0.0)
    f1 = _metric(metrics, "f1", default=0.0)
    precision = _metric(metrics, "precision", "precision_ai", default=0.0)
    recall = _metric(metrics, "recall", "recall_ai", default=0.0)
    ece = _metric(metrics, "ece", default=1.0)
    brier = _metric(metrics, "brier", default=1.0)
    checks[f"{prefix}_auc"] = auc
    checks[f"{prefix}_f1"] = f1
    checks[f"{prefix}_precision"] = precision
    checks[f"{prefix}_recall"] = recall
    checks[f"{prefix}_ece"] = ece
    checks[f"{prefix}_brier"] = brier
    if auc < min_auc:
        failures.append(f"{prefix}_auc {auc:.4f} < {min_auc:.4f}")
    if f1 < min_f1:
        failures.append(f"{prefix}_f1 {f1:.4f} < {min_f1:.4f}")
    if precision < min_precision:
        failures.append(f"{prefix}_precision {precision:.4f} < {min_precision:.4f}")
    if recall < min_recall:
        failures.append(f"{prefix}_recall {recall:.4f} < {min_recall:.4f}")
    if ece > max_ece:
        failures.append(f"{prefix}_ece {ece:.4f} > {max_ece:.4f}")
    if brier > max_brier:
        failures.append(f"{prefix}_brier {brier:.4f} > {max_brier:.4f}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Promotion gate for benchmark metrics")
    ap.add_argument("--ens-out", default="./artifacts_ens")
    ap.add_argument("--video-out", default="./video_artifacts")
    ap.add_argument("--min-image-auc", type=float, default=0.96)
    ap.add_argument("--min-image-f1", type=float, default=0.92)
    ap.add_argument("--min-image-precision", type=float, default=0.90)
    ap.add_argument("--min-image-recall", type=float, default=0.90)
    ap.add_argument("--max-image-ece", type=float, default=0.05)
    ap.add_argument("--max-image-brier", type=float, default=0.08)
    ap.add_argument("--min-robust-worst-auc", type=float, default=0.90)
    ap.add_argument("--min-robust-worst-f1", type=float, default=0.85)
    ap.add_argument("--max-robust-auc-drop", type=float, default=0.08)
    ap.add_argument("--min-video-acc", type=float, default=0.86)
    ap.add_argument("--allow-missing-video", action="store_true", default=False)
    ap.add_argument("--skip-video", action="store_true", default=False)
    args = ap.parse_args()

    ens = Path(args.ens_out)
    video = Path(args.video_out)
    failures: list[str] = []
    checks: dict[str, float | str] = {}

    test_metrics = _read_json(ens / "test_metrics.json")
    _check_model_metrics(
        prefix="image",
        metrics=test_metrics,
        min_auc=args.min_image_auc,
        min_f1=args.min_image_f1,
        min_precision=args.min_image_precision,
        min_recall=args.min_image_recall,
        max_ece=args.max_image_ece,
        max_brier=args.max_image_brier,
        checks=checks,
        failures=failures,
    )

    robust_eval = _read_json(ens / "robust_eval.json")
    clean_metrics = robust_eval.get("clean")
    if not isinstance(clean_metrics, dict):
        failures.append("missing clean robust_eval metrics")
    else:
        robust_variants = [
            (name, metrics)
            for name, metrics in robust_eval.items()
            if name != "clean" and isinstance(metrics, dict)
        ]
        if not robust_variants:
            failures.append("missing degraded robust_eval variants")
        else:
            clean_auc = _metric(clean_metrics, "auc", default=0.0)
            worst_name, worst_metrics = min(
                robust_variants,
                key=lambda item: _metric(item[1], "auc", default=0.0),
            )
            worst_auc = _metric(worst_metrics, "auc", default=0.0)
            worst_f1 = min(_metric(metrics, "f1", default=0.0) for _, metrics in robust_variants)
            max_auc_drop = max(0.0, clean_auc - worst_auc)
            checks["robust_clean_auc"] = clean_auc
            checks["robust_worst_variant"] = worst_name
            checks["robust_worst_auc"] = worst_auc
            checks["robust_worst_f1"] = worst_f1
            checks["robust_max_auc_drop"] = max_auc_drop
            if worst_auc < args.min_robust_worst_auc:
                failures.append(f"robust_worst_auc {worst_auc:.4f} < {args.min_robust_worst_auc:.4f}")
            if worst_f1 < args.min_robust_worst_f1:
                failures.append(f"robust_worst_f1 {worst_f1:.4f} < {args.min_robust_worst_f1:.4f}")
            if max_auc_drop > args.max_robust_auc_drop:
                failures.append(f"robust_auc_drop {max_auc_drop:.4f} > {args.max_robust_auc_drop:.4f}")

    vlog = video / "training_log.jsonl"
    if args.skip_video:
        checks["video_best_val_acc"] = "skipped"
    elif vlog.exists():
        best_acc = 0.0
        for line in vlog.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            best_acc = max(best_acc, float(row.get("val_acc", 0.0)))
        checks["video_best_val_acc"] = best_acc
        if best_acc < args.min_video_acc:
            failures.append(f"video_best_val_acc {best_acc:.4f} < {args.min_video_acc:.4f}")
    elif args.allow_missing_video:
        checks["video_best_val_acc"] = "skipped_missing_video"
    else:
        failures.append(f"missing {vlog}")

    required = [
        ens / "prod_manifest.json",
        ens / "ensemble_config.json",
        ens / "domain_config.json",
        ens / "robust_eval.json",
    ]
    for p in required:
        if not p.exists():
            failures.append(f"missing {p}")
    if (
        not args.skip_video
        and not (video / "best_video.safetensors").exists()
        and not args.allow_missing_video
    ):
        failures.append(f"missing {video / 'best_video.safetensors'}")

    public_model = select_public_model(ens)
    if public_model is None:
        failures.append("missing promotable public model candidate")
    else:
        public_dir = Path(str(public_model["artifact_dir"]))
        checks["public_model_member"] = public_model["name"]
        checks["public_model_threshold"] = public_model["calibration"].get("threshold")
        checks["public_model_temperature"] = public_model["calibration"].get("temperature")
        checks["public_model_metadata_aware"] = bool(public_model["use_metadata_features"])
        for required_name in ("calibration.json", "config.json", "inference_spec.json", "best_model_summary.json", "test_metrics.json"):
            path = public_dir / required_name
            if not path.exists():
                failures.append(f"missing {path}")
        public_test_metrics = public_model.get("test_metrics", {})
        if not public_test_metrics:
            failures.append(f"missing {public_dir / 'test_metrics.json'}")
        else:
            _check_model_metrics(
                prefix="public_model",
                metrics=public_test_metrics,
                min_auc=args.min_image_auc,
                min_f1=args.min_image_f1,
                min_precision=args.min_image_precision,
                min_recall=args.min_image_recall,
                max_ece=args.max_image_ece,
                max_brier=args.max_image_brier,
                checks=checks,
                failures=failures,
            )

    out = {"ok": len(failures) == 0, "checks": checks, "failures": failures}
    print(json.dumps(out, indent=2))
    return 0 if out["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
