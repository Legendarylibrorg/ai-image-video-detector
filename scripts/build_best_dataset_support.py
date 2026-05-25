from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, DefaultDict, Iterable

from ai_image_detector.utils import write_json_dict


SPLITS = ("train", "val", "test")
CLASSES = ("ai", "real")


def run_source_acceptance_loop(
    examples: Iterable[object],
    *,
    is_done: Callable[[], bool],
    max_unique_per_source: int,
    global_rejects: dict[str, int],
    try_accept_and_save: Callable[[Any, str, str, dict[str, int], dict[str, dict[str, int]]], bool],
    extract_payload: Callable[[object], tuple[str | None, Any | None, str | None]],
    source_name: str,
    source_counts: dict[str, int],
    source_split_counts: dict[str, dict[str, int]],
    acceptance_warmup_samples: int | None = None,
    min_acceptance_rate: float | None = None,
) -> dict[str, Any]:
    source_rejects: DefaultDict[str, int] = defaultdict(int)
    accepted_total = 0
    processed_total = 0
    exhausted_for_future = False

    for ex in examples:
        if is_done():
            break
        processed_total += 1
        if acceptance_warmup_samples is not None and min_acceptance_rate is not None and processed_total >= int(acceptance_warmup_samples):
            acceptance_rate = accepted_total / float(max(1, processed_total))
            if acceptance_rate < float(min_acceptance_rate):
                source_rejects["low_acceptance_rate"] += 1
                exhausted_for_future = accepted_total == 0
                break
        if accepted_total >= int(max_unique_per_source):
            source_rejects["source_total_cap"] += 1
            break

        cls, img, reject_reason = extract_payload(ex)
        if reject_reason is not None:
            source_rejects[reject_reason] += 1
            continue

        before = dict(global_rejects)
        if try_accept_and_save(img, cls, source_name, source_counts, source_split_counts):
            accepted_total += 1
            continue

        changed = [k for k, v in global_rejects.items() if v != before.get(k, 0)]
        if changed:
            source_rejects[changed[0]] += 1
        else:
            source_rejects["rejected_other"] += 1

    if accepted_total == 0 and processed_total >= int(max_unique_per_source):
        exhausted_for_future = True

    return {
        "accepted_total": int(accepted_total),
        "processed_total": int(processed_total),
        "rejections": dict(source_rejects),
        "skip_future_runs": bool(exhausted_for_future),
        "acceptance_rate": accepted_total / float(max(1, processed_total)),
    }


def make_source_report(
    *,
    source: str,
    source_type: str,
    source_counts: dict[str, int],
    source_split_counts: dict[str, dict[str, int]],
    loop_result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "source": source,
        "type": source_type,
        "status": "completed",
        "accepted_ai": int(source_counts["ai"]),
        "accepted_real": int(source_counts["real"]),
        "accepted_by_split": source_split_counts,
        "accepted_total": int(loop_result["accepted_total"]),
        "processed_total": int(loop_result.get("processed_total", 0)),
        "rejections": dict(loop_result["rejections"]),
        "skip_future_runs": bool(loop_result.get("skip_future_runs", False)),
        "acceptance_rate": float(loop_result["acceptance_rate"]),
    }


def build_summary(
    *,
    targets: dict[str, dict[str, int]],
    raw_counts: dict[str, dict[str, int]],
    global_rejections: dict[str, int],
    source_reports: list[dict[str, Any]],
    hf_sources: list[str],
    cache_dir: str,
    args,
    source_manifest_path: Path,
    manifest_policy: dict[str, object],
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    summary = {
        "targets": targets,
        "final_counts": raw_counts,
        "global_rejections": dict(global_rejections),
        "source_reports": source_reports,
        "source_candidates": hf_sources,
        "cache": {
            "cache_dir": cache_dir,
            "streaming": bool(args.streaming),
            "stream_buffer_size": int(args.stream_buffer_size),
            "max_samples_per_source": int(args.max_samples_per_source),
            "quiet_progress": bool(args.quiet_progress),
        },
        "source_manifest_path": str(source_manifest_path.resolve()),
        "source_manifest_policy": manifest_policy,
    }

    hf_reports = [r for r in source_reports if r.get("type") == "hf"]
    hf_sources_with_accepted = sum(1 for r in hf_reports if int(r.get("accepted_ai", 0)) + int(r.get("accepted_real", 0)) > 0)
    hf_sources_ai = sum(1 for r in hf_reports if int(r.get("accepted_ai", 0)) > 0)
    hf_sources_real = sum(1 for r in hf_reports if int(r.get("accepted_real", 0)) > 0)
    hf_sources_per_split_class = {
        split: {
            cls: sum(
                1
                for r in hf_reports
                if int((((r.get("accepted_by_split") or {}).get(split) or {}).get(cls, 0))) > 0
            )
            for cls in CLASSES
        }
        for split in SPLITS
    }
    summary["hf_sources_with_accepted"] = int(hf_sources_with_accepted)
    summary["hf_sources_ai"] = int(hf_sources_ai)
    summary["hf_sources_real"] = int(hf_sources_real)
    summary["hf_sources_per_split_class"] = hf_sources_per_split_class

    shortfalls: list[str] = []
    for split in SPLITS:
        for cls in CLASSES:
            have_n = summary["final_counts"][split][cls]
            need_n = targets[split][cls]
            if have_n < need_n:
                shortfalls.append(f"{split}/{cls}:{have_n}<{need_n}")
    summary["full_targets_ok"] = len(shortfalls) == 0

    return summary, hf_sources_per_split_class, shortfalls


def write_summary_files(out: Path, summary: dict[str, Any], run_summary: dict[str, Any]) -> None:
    write_json_dict(out / "dataset_build_report.json", summary)
    write_json_dict(out / "dataset_state.json", summary)
    write_json_dict(out / "dataset_run_summary.json", run_summary)
