from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Promotion gate for benchmark metrics")
    ap.add_argument("--ens-out", default="./artifacts_ens")
    ap.add_argument("--video-out", default="./video_artifacts")
    ap.add_argument("--min-image-auc", type=float, default=0.93)
    ap.add_argument("--min-image-f1", type=float, default=0.90)
    ap.add_argument("--min-video-acc", type=float, default=0.82)
    args = ap.parse_args()

    ens = Path(args.ens_out)
    video = Path(args.video_out)
    failures: list[str] = []
    checks: dict[str, float | str] = {}

    test_metrics = _read_json(ens / "test_metrics.json")
    image_auc = float(test_metrics.get("auc", 0.0))
    image_f1 = float(test_metrics.get("f1", 0.0))
    checks["image_auc"] = image_auc
    checks["image_f1"] = image_f1
    if image_auc < args.min_image_auc:
        failures.append(f"image_auc {image_auc:.4f} < {args.min_image_auc:.4f}")
    if image_f1 < args.min_image_f1:
        failures.append(f"image_f1 {image_f1:.4f} < {args.min_image_f1:.4f}")

    vlog = video / "training_log.jsonl"
    if vlog.exists():
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
    else:
        failures.append(f"missing {vlog}")

    required = [
        ens / "prod_manifest.json",
        ens / "ensemble_config.json",
        ens / "domain_config.json",
        video / "best_video.pt",
    ]
    for p in required:
        if not p.exists():
            failures.append(f"missing {p}")

    out = {"ok": len(failures) == 0, "checks": checks, "failures": failures}
    print(json.dumps(out, indent=2))
    return 0 if out["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
