from __future__ import annotations

import argparse
from datetime import datetime, timezone
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Export best model artifacts into a versioned release bundle")
    ap.add_argument("--out", required=True, help="Training output directory")
    ap.add_argument("--model", default="best.pt", help="Model filename in --out")
    args = ap.parse_args()

    out = Path(args.out)
    model = out / args.model
    if not model.exists():
        raise FileNotFoundError(f"missing model: {model}")

    rel = out / "releases" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rel.mkdir(parents=True, exist_ok=True)

    names = [
        args.model,
        "best_metrics.json",
        "best_group_metrics.json",
        "calibration.json",
        "test_metrics.json",
        "config.json",
        "last_metrics.json",
        "training_log.jsonl",
    ]
    for name in names:
        src = out / name
        if src.exists():
            shutil.copy2(src, rel / name)

    (out / "latest_release.txt").write_text(str(rel), encoding="utf-8")
    print(f"saved_release={rel}")


if __name__ == "__main__":
    main()
