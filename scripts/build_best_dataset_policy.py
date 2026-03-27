from __future__ import annotations

import random
from datetime import datetime, timezone
from typing import Optional


SPLITS = ("train", "val", "test")


def next_split_for_class(
    have: dict[str, dict[str, int]],
    need: dict[str, dict[str, int]],
    cls: str,
    rng: random.Random,
) -> Optional[str]:
    remaining = {s: max(0, need[s][cls] - have[s][cls]) for s in SPLITS}
    choices = [s for s, rem in remaining.items() if rem > 0]
    if not choices:
        return None
    total = float(sum(remaining[s] for s in choices))
    pick = rng.random() * total
    acc = 0.0
    for split in choices:
        acc += float(remaining[split])
        if pick <= acc:
            return split
    return choices[-1]


def next_split_for_source_class(
    have: dict[str, dict[str, int]],
    need: dict[str, dict[str, int]],
    source_split_counts: dict[str, dict[str, int]],
    cls: str,
    rng: random.Random,
    max_per_source_split_class: int,
) -> Optional[str]:
    existing_splits = [split for split in SPLITS if int(source_split_counts[split][cls]) > 0]
    if existing_splits:
        assigned_split = max(existing_splits, key=lambda split: int(source_split_counts[split][cls]))
        remaining = max(0, need[assigned_split][cls] - have[assigned_split][cls])
        if remaining <= 0:
            return None
        if source_split_counts[assigned_split][cls] >= max_per_source_split_class:
            return None
        return assigned_split

    choices: list[str] = []
    weighted_remaining: dict[str, int] = {}
    for split in SPLITS:
        remaining = max(0, need[split][cls] - have[split][cls])
        if remaining <= 0:
            continue
        if source_split_counts[split][cls] >= max_per_source_split_class:
            continue
        choices.append(split)
        weighted_remaining[split] = remaining

    if not choices:
        return None
    if len(choices) == 1:
        return choices[0]

    best_split = choices[0]
    best_score = float("-inf")
    for split in choices:
        score = float(weighted_remaining[split]) - (0.75 * float(source_split_counts[split][cls]))
        score += rng.random() * 1e-3
        if score > best_score:
            best_split = split
            best_score = score
    return best_split


def source_manifest_policy(args) -> dict[str, object]:
    return {
        "streaming": bool(args.streaming),
        "stream_buffer_size": int(args.stream_buffer_size),
        "max_samples_per_source": int(args.max_samples_per_source),
        "acceptance_warmup_samples": int(args.acceptance_warmup_samples),
        "min_acceptance_rate": float(args.min_acceptance_rate),
        "quiet_progress": bool(args.quiet_progress),
    }


def should_skip_source_from_manifest(entry: dict | None, policy: dict[str, object]) -> bool:
    if not entry:
        return False
    if not bool(entry.get("skip_future_runs", False)):
        return False
    return entry.get("policy") == policy


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
