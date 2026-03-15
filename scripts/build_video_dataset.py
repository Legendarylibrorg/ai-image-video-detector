from __future__ import annotations

import argparse
import os
from pathlib import Path
import random
import shutil
import time
from typing import Dict, List

from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download


SOURCES = [
    {
        "repo": "angads24/deepfake-video",
        "real_prefixes": ["dataset/real/"],
        "fake_prefixes": ["dataset/fake/"],
    },
    {
        "repo": "Sarim-Hash/video_DEEPFAKE_dataset",
        "real_prefixes": ["real_video/"],
        "fake_prefixes": ["fake_video/"],
    },
    {
        "repo": "UniDataPro/deepfake-videos-dataset",
        "real_prefixes": ["video/"],
        "fake_prefixes": ["deepfake/"],
    },
]

EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".MOV")


def done(have: Dict[str, Dict[str, int]], need: Dict[str, Dict[str, int]]) -> bool:
    return all(have[s][c] >= need[s][c] for s in ["train", "val"] for c in ["ai", "real"])


def count_existing(out: Path) -> Dict[str, Dict[str, int]]:
    have = {"train": {"ai": 0, "real": 0}, "val": {"ai": 0, "real": 0}}
    for split in ["train", "val"]:
        for cls in ["ai", "real"]:
            have[split][cls] = len(list((out / split / cls).glob("*")))
    return have


def _collect_snapshot_files(root: Path, prefixes: List[str]) -> List[Path]:
    files: List[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix not in EXTS and p.suffix.lower() not in tuple(e.lower() for e in EXTS):
            continue
        rel = p.relative_to(root).as_posix()
        if any(rel.startswith(pref) for pref in prefixes):
            files.append(p)
    return files


def _download_with_retry(repo: str, filename: str, token: str | None, retries: int, sleep_ms: int) -> str | None:
    for attempt in range(retries):
        try:
            return hf_hub_download(repo_id=repo, repo_type="dataset", filename=filename, token=token)
        except Exception:
            if attempt + 1 >= retries:
                return None
            time.sleep((2**attempt) * (sleep_ms / 1000.0))
    return None


def main():
    ap = argparse.ArgumentParser(description="Build video deepfake train/val dataset from HF")
    ap.add_argument("--out", default="video_data")
    ap.add_argument("--train-per-class", type=int, default=220)
    ap.add_argument("--val-per-class", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=["snapshot", "per-file"], default="snapshot")
    ap.add_argument("--chunk-size", type=int, default=20)
    ap.add_argument("--sleep-ms", type=int, default=120)
    ap.add_argument("--jitter-ms", type=int, default=80)
    ap.add_argument("--chunk-pause-ms", type=int, default=1000)
    ap.add_argument("--repo-cooldown-ms", type=int, default=3000)
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--token-env", default="HF_TOKEN")
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    token = os.getenv(args.token_env)
    if token:
        print(f"using_token_env={args.token_env}")
    else:
        print(f"warning_no_token env={args.token_env} (works, but lower rate limits)")

    random.seed(args.seed)
    out = Path(args.out)
    for split in ["train", "val"]:
        for cls in ["ai", "real"]:
            (out / split / cls).mkdir(parents=True, exist_ok=True)

    need = {
        "train": {"ai": args.train_per_class, "real": args.train_per_class},
        "val": {"ai": args.val_per_class, "real": args.val_per_class},
    }
    have = count_existing(out)

    for src in SOURCES:
        if done(have, need):
            break

        repo = src["repo"]
        print(f"repo={repo} mode={args.mode}")

        fake_files: List[str] = []
        real_files: List[str] = []

        if args.mode == "snapshot":
            # Minimal HF API calls: one snapshot download call per repo.
            allow_patterns = [f"{pref}*" for pref in src["fake_prefixes"] + src["real_prefixes"]]
            try:
                snap = snapshot_download(
                    repo_id=repo,
                    repo_type="dataset",
                    token=token,
                    allow_patterns=allow_patterns,
                    resume_download=True,
                    max_workers=4,
                    cache_dir=args.cache_dir,
                )
            except Exception as e:
                print(f"skip repo={repo} err={e}")
                continue

            snap_root = Path(snap)
            fake_local = _collect_snapshot_files(snap_root, src["fake_prefixes"])
            real_local = _collect_snapshot_files(snap_root, src["real_prefixes"])
            random.shuffle(fake_local)
            random.shuffle(real_local)

            print(f"snapshot_files repo={repo} fake={len(fake_local)} real={len(real_local)}")

            for cls, locals_ in [("ai", fake_local), ("real", real_local)]:
                for p in locals_:
                    split = "train" if have["train"][cls] < need["train"][cls] else ("val" if have["val"][cls] < need["val"][cls] else None)
                    if split is None:
                        break
                    n = have[split][cls]
                    dst = out / split / cls / f"src={repo.split('/')[-1]}__{n:05d}{p.suffix.lower()}"
                    if not dst.exists():
                        shutil.copy2(p, dst)
                        have[split][cls] += 1

                if done(have, need):
                    break

        else:
            try:
                files = list_repo_files(repo, repo_type="dataset", token=token)
            except Exception as e:
                print(f"skip repo={repo} err={e}")
                continue

            def pick_files(prefixes: List[str], seed_offset: int) -> List[str]:
                cands = [f for f in files if any(f.startswith(p) for p in prefixes) and f.endswith(EXTS)]
                rng = random.Random(args.seed + seed_offset)
                rng.shuffle(cands)
                return cands

            fake_files = pick_files(src["fake_prefixes"], 0)
            real_files = pick_files(src["real_prefixes"], 1)
            print(f"per_file_list repo={repo} fake={len(fake_files)} real={len(real_files)}")

            for cls, file_list in [("ai", fake_files), ("real", real_files)]:
                i = 0
                consecutive_failures = 0
                while i < len(file_list) and (have["train"][cls] < need["train"][cls] or have["val"][cls] < need["val"][cls]):
                    batch = file_list[i : i + args.chunk_size]
                    pulled = 0
                    for f in batch:
                        split = "train" if have["train"][cls] < need["train"][cls] else ("val" if have["val"][cls] < need["val"][cls] else None)
                        if split is None:
                            break

                        local = _download_with_retry(repo, f, token, retries=args.retries, sleep_ms=args.sleep_ms)
                        if not local:
                            consecutive_failures += 1
                            if consecutive_failures >= 3:
                                time.sleep(args.repo_cooldown_ms / 1000.0)
                                consecutive_failures = 0
                            continue
                        consecutive_failures = 0

                        n = have[split][cls]
                        dst = out / split / cls / f"src={repo.split('/')[-1]}__{n:05d}{Path(f).suffix.lower()}"
                        if not dst.exists():
                            shutil.copy2(local, dst)
                            have[split][cls] += 1
                            pulled += 1

                        jitter = random.randint(0, max(args.jitter_ms, 0))
                        time.sleep((args.sleep_ms + jitter) / 1000.0)

                    i += len(batch)
                    print(f"repo={repo} cls={cls} idx={i} pulled_batch={pulled} train={have['train'][cls]} val={have['val'][cls]}")
                    time.sleep(args.chunk_pause_ms / 1000.0)

                    if done(have, need):
                        break

        for split in ["train", "val"]:
            for cls in ["ai", "real"]:
                print(f"progress {split}/{cls}={have[split][cls]}")

    for split in ["train", "val"]:
        for cls in ["ai", "real"]:
            n = len(list((out / split / cls).glob("*")))
            print(f"{split}/{cls}={n}")


if __name__ == "__main__":
    main()
