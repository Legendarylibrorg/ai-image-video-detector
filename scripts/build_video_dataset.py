from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import random
import shutil
import time
from typing import Dict, List

from ai_image_detector.collection_paths import validate_collection_io_paths

from dataset_builder_common import HF_CACHE_DIR_DEFAULT, configure_hf_cache_env, count_existing_split_classes, targets_met
from hf_data import download_dataset_file, list_dataset_repo_files, resolve_hf_token_value, snapshot_dataset_repo

hf_hub_download = download_dataset_file
list_repo_files = list_dataset_repo_files
snapshot_download = snapshot_dataset_repo


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
EXTS_LOWER = {e.lower() for e in EXTS}


def done(have: Dict[str, Dict[str, int]], need: Dict[str, Dict[str, int]]) -> bool:
    return targets_met(have, need, ("train", "val"), ("ai", "real"))


def count_existing(out: Path) -> Dict[str, Dict[str, int]]:
    have = count_existing_split_classes(out, ("train", "val"), ("ai", "real"), "*")
    for split in ("train", "val"):
        for cls in ("ai", "real"):
            split_dir = out / split / cls
            if not split_dir.exists():
                have[split][cls] = 0
                continue
            have[split][cls] = sum(
                1
                for path in split_dir.glob("*")
                if path.is_file() and path.suffix.lower() in EXTS_LOWER
            )
    return have


def _collect_snapshot_files(root: Path, prefixes: List[str]) -> List[Path]:
    files: List[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in EXTS_LOWER:
            continue
        rel = p.relative_to(root).as_posix()
        if any(rel.startswith(pref) for pref in prefixes):
            files.append(p)
    return files


def _download_with_retry(
    repo: str,
    filename: str,
    token: str | None,
    cache_dir: str | None,
    retries: int,
    sleep_ms: int,
) -> str | None:
    for attempt in range(retries):
        try:
            return hf_hub_download(repo, filename, token=token, cache_dir=cache_dir)
        except Exception:
            if attempt + 1 >= retries:
                return None
            time.sleep((2**attempt) * (sleep_ms / 1000.0))
    return None


def _passes_video_quality(path: Path, min_bytes: int, max_bytes: int) -> bool:
    try:
        n = path.stat().st_size
    except Exception:
        return False
    if n < int(min_bytes):
        return False
    if int(max_bytes) > 0 and n > int(max_bytes):
        return False
    return True


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_existing_video_hashes(out: Path) -> set[str]:
    paths: list[Path] = []
    for split in ("train", "val"):
        for cls in ("ai", "real"):
            split_dir = out / split / cls
            if not split_dir.exists():
                continue
            for path in split_dir.glob("*"):
                if path.is_file():
                    paths.append(path)
    seen: set[str] = set()
    if not paths:
        return seen
    max_workers = min(8, max(1, (os.cpu_count() or 4) // 2))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for digest in ex.map(_sha256, paths):
            seen.add(digest)
    return seen


def main():
    ap = argparse.ArgumentParser(description="Build video deepfake train/val dataset from HF")
    ap.add_argument("--out", default="video_data")
    ap.add_argument("--train-per-class", type=int, default=220)
    ap.add_argument("--val-per-class", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=["snapshot", "per-file"], default="snapshot")
    ap.add_argument("--snapshot-max-workers", type=int, default=8)
    ap.add_argument("--chunk-size", type=int, default=20)
    ap.add_argument("--sleep-ms", type=int, default=40)
    ap.add_argument("--jitter-ms", type=int, default=80)
    ap.add_argument("--chunk-pause-ms", type=int, default=250)
    ap.add_argument("--repo-cooldown-ms", type=int, default=12000)
    ap.add_argument("--repo-base-pause-ms", type=int, default=150)
    ap.add_argument("--repo-jitter-ms", type=int, default=150)
    ap.add_argument("--copy-sleep-ms", type=int, default=15)
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--min-video-bytes", type=int, default=100000)
    ap.add_argument("--max-video-bytes", type=int, default=0)
    ap.add_argument("--cache-dir", default=HF_CACHE_DIR_DEFAULT)
    ap.add_argument("--token-env", default="HF_TOKEN")
    args = ap.parse_args()
    validate_collection_io_paths(out=args.out, cache_dir=args.cache_dir or None)

    cache_dir = configure_hf_cache_env(args.cache_dir)
    if cache_dir is not None:
        print(f"hf_cache_dir={cache_dir}")

    token, token_source = resolve_hf_token_value(args.token_env)
    if token:
        if token_source.startswith("env:"):
            print(f"using_token_env={token_source.split(':', 1)[1]}")
        else:
            print(f"using_token_source={token_source}")
    else:
        print(f"warning_no_token env={args.token_env} (works, but lower rate limits; hf auth login also works)")
    print(f"video_quality_filters min_bytes={args.min_video_bytes} max_bytes={args.max_video_bytes}")

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
    seen_hashes = load_existing_video_hashes(out)

    for src in SOURCES:
        if done(have, need):
            break

        repo = src["repo"]
        # Stagger repo requests so we don't burst across datasets.
        repo_pause = args.repo_base_pause_ms + random.randint(0, max(args.repo_jitter_ms, 0))
        time.sleep(repo_pause / 1000.0)
        print(f"repo={repo} mode={args.mode}")

        if args.mode == "snapshot":
            # Minimal HF API calls: one snapshot download call per repo.
            allow_patterns = [f"{pref}*" for pref in src["fake_prefixes"] + src["real_prefixes"]]
            try:
                snap = snapshot_download(
                    repo,
                    token=token,
                    allow_patterns=allow_patterns,
                    max_workers=args.snapshot_max_workers,
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
                        if _passes_video_quality(p, args.min_video_bytes, args.max_video_bytes):
                            src_hash = _sha256(p)
                            if src_hash in seen_hashes:
                                continue
                            shutil.copy2(p, dst)
                            seen_hashes.add(src_hash)
                            have[split][cls] += 1
                            if args.copy_sleep_ms > 0:
                                time.sleep(args.copy_sleep_ms / 1000.0)

                if done(have, need):
                    break

        else:
            try:
                files = list_repo_files(repo, token=token)
            except Exception as e:
                print(f"skip repo={repo} err={e}")
                continue

            def pick_files(prefixes: List[str], seed_offset: int) -> List[str]:
                cands = [f for f in files if any(f.startswith(p) for p in prefixes) and Path(f).suffix.lower() in EXTS_LOWER]
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

                        local = _download_with_retry(
                            repo,
                            f,
                            token,
                            args.cache_dir or None,
                            retries=args.retries,
                            sleep_ms=args.sleep_ms,
                        )
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
                            local_path = Path(local)
                            if _passes_video_quality(local_path, args.min_video_bytes, args.max_video_bytes):
                                src_hash = _sha256(local_path)
                                if src_hash in seen_hashes:
                                    continue
                                shutil.copy2(local_path, dst)
                                seen_hashes.add(src_hash)
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
