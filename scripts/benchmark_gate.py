from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(obj: dict, *names: str, default: float | None = None) -> float:
    for name in names:
        if name in obj:
            return float(obj[name])
    if default is None:
        raise KeyError(names[0] if names else "metric")
    return float(default)


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
    image_auc = _metric(test_metrics, "auc", default=0.0)
    image_f1 = _metric(test_metrics, "f1", default=0.0)
    image_precision = _metric(test_metrics, "precision", "precision_ai", default=0.0)
    image_recall = _metric(test_metrics, "recall", "recall_ai", default=0.0)
    image_ece = _metric(test_metrics, "ece", default=1.0)
    image_brier = _metric(test_metrics, "brier", default=1.0)
    checks["image_auc"] = image_auc
    checks["image_f1"] = image_f1
    checks["image_precision"] = image_precision
    checks["image_recall"] = image_recall
    checks["image_ece"] = image_ece
    checks["image_brier"] = image_brier
    if image_auc < args.min_image_auc:
        failures.append(f"image_auc {image_auc:.4f} < {args.min_image_auc:.4f}")
    if image_f1 < args.min_image_f1:
        failures.append(f"image_f1 {image_f1:.4f} < {args.min_image_f1:.4f}")
    if image_precision < args.min_image_precision:
        failures.append(f"image_precision {image_precision:.4f} < {args.min_image_precision:.4f}")
    if image_recall < args.min_image_recall:
        failures.append(f"image_recall {image_recall:.4f} < {args.min_image_recall:.4f}")
    if image_ece > args.max_image_ece:
        failures.append(f"image_ece {image_ece:.4f} > {args.max_image_ece:.4f}")
    if image_brier > args.max_image_brier:
        failures.append(f"image_brier {image_brier:.4f} > {args.max_image_brier:.4f}")

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

    out = {"ok": len(failures) == 0, "checks": checks, "failures": failures}
    print(json.dumps(out, indent=2))
    return 0 if out["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
