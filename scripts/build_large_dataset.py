from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def main() -> None:
    ap = argparse.ArgumentParser(description="Compatibility wrapper around build_best_dataset.py")
    ap.add_argument("--dataset", default="Hemg/AI-Generated-vs-Real-Images-Datasets")
    ap.add_argument("--out", default="data")
    ap.add_argument("--train-per-class", type=int, default=20000)
    ap.add_argument("--val-per-class", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quality", type=int, default=92)
    args = ap.parse_args()

    target_script = Path(__file__).with_name("build_best_dataset.py")
    cmd = [
        sys.executable,
        str(target_script),
        "--out",
        args.out,
        "--seed",
        str(args.seed),
        "--train-per-class",
        str(args.train_per_class),
        "--val-per-class",
        str(args.val_per_class),
        "--test-per-class",
        "0",
        "--jpeg-quality",
        str(args.quality),
        "--extra-source",
        args.dataset,
        "--hf-only",
        "--no-default-sources",
        "--require-full-targets",
    ]
    print("deprecated_wrapper=build_large_dataset.py target=build_best_dataset.py")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
