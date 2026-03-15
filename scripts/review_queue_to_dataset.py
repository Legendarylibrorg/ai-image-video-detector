from __future__ import annotations

import argparse
from pathlib import Path
import shutil


def main() -> None:
    ap = argparse.ArgumentParser(description="Move reviewed queue items into labeled dataset folders")
    ap.add_argument("--queue", default="./incoming_review_queue")
    ap.add_argument("--dst", default="./data_new/train")
    ap.add_argument("--archive", default="./incoming_review_queue/_processed")
    ap.add_argument("--copy", action="store_true", default=False, help="Copy instead of move")
    args = ap.parse_args()

    q = Path(args.queue)
    dst = Path(args.dst)
    archive = Path(args.archive)
    for cls in ("ai", "real"):
        (dst / cls).mkdir(parents=True, exist_ok=True)
        src_dir = q / cls
        if not src_dir.exists():
            continue
        for p in src_dir.glob("*.jpg"):
            out = dst / cls / p.name
            if args.copy:
                shutil.copy2(p, out)
            else:
                p.rename(out)
            j = p.with_suffix(".json")
            if j.exists():
                aj = archive / cls
                aj.mkdir(parents=True, exist_ok=True)
                if args.copy:
                    shutil.copy2(j, aj / j.name)
                else:
                    j.rename(aj / j.name)
    print(f"review_queue_ingested dst={dst}")


if __name__ == "__main__":
    main()
