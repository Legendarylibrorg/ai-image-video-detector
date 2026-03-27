from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
from typing import Any

from release_selection import build_public_model_manifest, select_public_model
from script_support import git_commit, iter_member_dirs, read_json_dict, read_nonempty_lines, resolve_preferred_checkpoint, write_json_dict


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
SPLITS = ("train", "val", "test")
VIDEO_SPLITS = ("train", "val")
CLASSES = ("ai", "real")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _count_files(root: Path, splits: tuple[str, ...], exts: set[str]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for split in splits:
        counts[split] = {}
        for cls in CLASSES:
            bucket = root / split / cls
            if not bucket.exists():
                counts[split][cls] = 0
                continue
            counts[split][cls] = sum(1 for p in bucket.iterdir() if p.is_file() and p.suffix.lower() in exts)
    return counts


def _complete_split_class_counts(counts: dict[str, dict[str, int]], splits: tuple[str, ...]) -> bool:
    return all(int(counts.get(split, {}).get(cls, 0)) > 0 for split in splits for cls in CLASSES)


def _disk_free_gb(root: Path) -> float:
    try:
        usage = shutil.disk_usage(root)
    except Exception:
        return -1.0
    return round(usage.free / (1024 ** 3), 2)


def _selected_env() -> dict[str, str]:
    exact = {
        "DATA_DIR",
        "EPOCHS",
        "SWEEP_EPOCHS",
        "ENS_OUT",
        "ENS_CONFIG_PATH",
        "VIDEO_OUT",
        "VIDEO_ARTIFACTS_OUT",
        "PIPELINE_MIN_FREE_GB",
        "PIPELINE_STAGE",
    }
    prefixes = ("BEST_DS_", "FAST_", "VIDEO_", "RUN_", "SKIP_", "HARD_", "DISTILL_", "ENS_", "TRAIN_")
    out: dict[str, str] = {}
    for key, value in os.environ.items():
        if key in exact or key.startswith(prefixes):
            out[key] = value
    return dict(sorted(out.items()))


def write_dataset_report(args: argparse.Namespace) -> int:
    data_root = Path(args.data)
    prepared_root = Path(args.prepared)
    incremental_root = Path(args.incremental) if args.incremental else None
    video_root = Path(args.video)
    cache_file = Path(args.cache_file)

    dataset_build_report_path = data_root / "dataset_build_report.json"
    dataset_run_summary_path = data_root / "dataset_run_summary.json"
    training_data_report_path = prepared_root / "training_data_report.json"
    dataset_build_report = read_json_dict(dataset_build_report_path)
    dataset_run_summary = read_json_dict(dataset_run_summary_path)
    training_data_report = read_json_dict(training_data_report_path)
    hf_sources = read_nonempty_lines(cache_file)

    data_counts = _count_files(data_root, SPLITS, IMAGE_EXTS)
    prepared_counts = _count_files(prepared_root, SPLITS, IMAGE_EXTS)
    video_counts = _count_files(video_root, VIDEO_SPLITS, VIDEO_EXTS)

    qa_summary = {
        "generated_at": _now(),
        "git_commit": git_commit(),
        "disk_free_gb": _disk_free_gb(Path(".")),
        "paths": {
            "data": str(data_root.resolve()),
            "prepared": str(prepared_root.resolve()),
            "incremental": str(incremental_root.resolve()) if incremental_root else None,
            "video": str(video_root.resolve()),
            "hf_cache_file": str(cache_file.resolve()),
        },
        "image_counts": {
            "collected": data_counts,
            "prepared": prepared_counts,
        },
        "video_counts": video_counts,
        "qa_checks": {
            "collected_complete": bool(dataset_build_report.get("full_targets_ok", _complete_split_class_counts(data_counts, SPLITS))),
            "prepared_complete": bool(training_data_report.get("complete_image_dataset", _complete_split_class_counts(prepared_counts, SPLITS))),
            "video_complete": _complete_split_class_counts(video_counts, VIDEO_SPLITS),
        },
        "report_paths": {
            "dataset_build_report": str(dataset_build_report_path.resolve()) if dataset_build_report_path.exists() else None,
            "dataset_run_summary": str(dataset_run_summary_path.resolve()) if dataset_run_summary_path.exists() else None,
            "training_data_report": str(training_data_report_path.resolve()) if training_data_report_path.exists() else None,
        },
    }

    provenance = {
        "generated_at": _now(),
        "git_commit": git_commit(),
        "hf_discovery_sources": {
            "cache_file": str(cache_file.resolve()),
            "count": len(hf_sources),
            "sources": hf_sources,
        },
        "dataset_build_report": dataset_build_report,
        "dataset_run_summary": dataset_run_summary,
        "training_data_report": training_data_report,
    }

    write_json_dict(Path(args.out), qa_summary, indent=2)
    write_json_dict(Path(args.provenance_out), provenance, indent=2)
    return 0


def _read_member_summaries(ens_out: Path) -> list[dict[str, Any]]:
    members: list[dict[str, Any]] = []
    for root in iter_member_dirs(ens_out):
        name = root.name
        if not root.exists():
            continue
        best_safe = root / "best.safetensors"
        members.append(
            {
                "name": name,
                "artifact_dir": str(root.resolve()),
                "preferred_checkpoint": str(best_safe.resolve()) if best_safe.exists() else None,
                "best_metrics": read_json_dict(root / "best_metrics.json"),
                "group_metrics": read_json_dict(root / "best_group_metrics.json"),
                "calibration": read_json_dict(root / "calibration.json"),
            }
        )
    return members


def write_final_report(args: argparse.Namespace) -> int:
    data_root = Path(args.data)
    prepared_root = Path(args.prepared)
    ens_out = Path(args.ens_out)
    ensemble_config_path = Path(args.ensemble_config)
    domain_config_path = Path(args.domain_config)
    video_data_root = Path(args.video)
    video_artifacts = Path(args.video_artifacts)

    image_members = _read_member_summaries(ens_out)
    ensemble_config = read_json_dict(ensemble_config_path)
    domain_config = read_json_dict(domain_config_path)
    robust_eval = read_json_dict(Path(args.robust_eval))
    image_test_metrics = read_json_dict(ens_out / "test_metrics.json")
    video_metrics = read_json_dict(video_artifacts / "best_video_metrics.json")
    dataset_qa = read_json_dict(Path(args.dataset_qa))
    distill_dir = ens_out / "distill"
    distill_summary = read_json_dict(distill_dir / "best_model_summary.json")
    distill_checkpoint = resolve_preferred_checkpoint(distill_dir / "best.safetensors")
    release_bundle = Path(args.release_bundle) if getattr(args, "release_bundle", "") else None
    public_model = select_public_model(ens_out)
    public_model_release_path = None
    if release_bundle is not None and public_model is not None:
        public_model_release_path = str((release_bundle / "public_model" / "best.safetensors").resolve())
    public_model_summary = build_public_model_manifest(public_model, public_checkpoint=public_model_release_path)

    thresholds = {
        "image_models": {
            item["name"]: item.get("calibration", {}).get("threshold")
            for item in image_members
            if item.get("calibration")
        },
        "ensemble": ensemble_config.get("threshold"),
        "ensemble_fit_objective": ensemble_config.get("fit", {}).get("objective"),
        "domain_base_threshold": domain_config.get("base_threshold"),
        "domain_thresholds": domain_config.get("thresholds", {}),
        "video": video_metrics.get("threshold"),
    }

    video_model = resolve_preferred_checkpoint(video_artifacts / "best_video.safetensors")
    preferred_models = [item["preferred_checkpoint"] for item in image_members if item.get("preferred_checkpoint")]
    if video_model.exists():
        preferred_video_model = str(video_model.resolve())
    else:
        preferred_video_model = None
    if distill_checkpoint.exists():
        preferred_distill_model = str(distill_checkpoint.resolve())
    else:
        preferred_distill_model = None

    final_summary = {
        "generated_at": _now(),
        "git_commit": git_commit(),
        "disk_free_gb": _disk_free_gb(Path(".")),
        "datasets": {
            "data_root": str(data_root.resolve()),
            "prepared_root": str(prepared_root.resolve()),
            "video_data_root": str(video_data_root.resolve()),
            "dataset_qa": dataset_qa,
        },
        "image_models": image_members,
        "image_test_metrics": image_test_metrics,
        "ensemble_fit": ensemble_config.get("fit", {}),
        "robust_eval": robust_eval,
        "video_metrics": video_metrics,
        "distilled_model": distill_summary,
        "public_model": public_model_summary,
        "thresholds": thresholds,
        "preferred_checkpoints": {
            "image_models": preferred_models,
            "video_model": preferred_video_model,
            "distilled_model": preferred_distill_model,
        },
        "release_bundle": str(release_bundle.resolve()) if release_bundle else None,
    }

    run_manifest = {
        "schema": "ai-image-detector-run-manifest-v1",
        "generated_at": _now(),
        "git_commit": git_commit(),
        "env": _selected_env(),
        "datasets": {
            "data": str(data_root.resolve()),
            "prepared": str(prepared_root.resolve()),
            "video_data": str(video_data_root.resolve()),
        },
        "artifacts": {
            "ensemble_dir": str(ens_out.resolve()),
            "video_artifacts_dir": str(video_artifacts.resolve()),
            "distill_dir": str(distill_dir.resolve()),
            "ensemble_config": str(ensemble_config_path.resolve()),
            "domain_config": str(domain_config_path.resolve()),
            "final_run_summary": str(Path(args.summary_out).resolve()),
            "threshold_summary": str(Path(args.thresholds_out).resolve()),
            "robust_eval": str(Path(args.robust_eval).resolve()),
            "release_bundle": str(release_bundle.resolve()) if release_bundle else None,
        },
        "preferred_checkpoints": final_summary["preferred_checkpoints"],
    }

    prod_manifest = {
        "models": preferred_models,
        "test_metrics": str((ens_out / "test_metrics.json").resolve()) if (ens_out / "test_metrics.json").exists() else None,
        "ensemble_config": str(ensemble_config_path.resolve()) if ensemble_config_path.exists() else None,
        "domain_config": str(domain_config_path.resolve()) if domain_config_path.exists() else None,
        "robust_eval": str(Path(args.robust_eval).resolve()) if Path(args.robust_eval).exists() else None,
        "video_model": preferred_video_model,
        "distilled_model": preferred_distill_model,
        "public_model": public_model_summary,
        "final_run_summary": str(Path(args.summary_out).resolve()),
        "run_manifest": str(Path(args.manifest_out).resolve()),
        "threshold_summary": str(Path(args.thresholds_out).resolve()),
        "release_bundle": str(release_bundle.resolve()) if release_bundle else None,
    }

    write_json_dict(Path(args.summary_out), final_summary, indent=2)
    write_json_dict(Path(args.manifest_out), run_manifest, indent=2)
    write_json_dict(Path(args.thresholds_out), thresholds, indent=2)
    write_json_dict(Path(args.prod_manifest), prod_manifest, indent=2)
    return 0


def write_failure_report(args: argparse.Namespace) -> int:
    payload = {
        "generated_at": _now(),
        "git_commit": git_commit(),
        "exit_code": int(args.exit_code),
        "stage": args.stage,
        "disk_free_gb": _disk_free_gb(Path(".")),
        "env": _selected_env(),
        "paths": {
            "data": str(Path(args.data).resolve()),
            "ens_out": str(Path(args.ens_out).resolve()),
            "video_out": str(Path(args.video).resolve()),
            "video_artifacts": str(Path(args.video_artifacts).resolve()),
        },
    }
    write_json_dict(Path(args.out), payload, indent=2)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Write pipeline QA, manifest, and summary artifacts")
    sub = ap.add_subparsers(dest="mode", required=True)

    dataset = sub.add_parser("dataset")
    dataset.add_argument("--data", required=True)
    dataset.add_argument("--prepared", required=True)
    dataset.add_argument("--incremental", default="")
    dataset.add_argument("--video", required=True)
    dataset.add_argument("--cache-file", required=True)
    dataset.add_argument("--out", required=True)
    dataset.add_argument("--provenance-out", required=True)

    final = sub.add_parser("final")
    final.add_argument("--data", required=True)
    final.add_argument("--prepared", required=True)
    final.add_argument("--video", required=True)
    final.add_argument("--ens-out", required=True)
    final.add_argument("--ensemble-config", required=True)
    final.add_argument("--domain-config", required=True)
    final.add_argument("--video-artifacts", required=True)
    final.add_argument("--dataset-qa", required=True)
    final.add_argument("--robust-eval", required=True)
    final.add_argument("--prod-manifest", required=True)
    final.add_argument("--summary-out", required=True)
    final.add_argument("--manifest-out", required=True)
    final.add_argument("--thresholds-out", required=True)
    final.add_argument("--release-bundle", default="")

    failure = sub.add_parser("failure")
    failure.add_argument("--stage", required=True)
    failure.add_argument("--exit-code", required=True, type=int)
    failure.add_argument("--data", required=True)
    failure.add_argument("--ens-out", required=True)
    failure.add_argument("--video", required=True)
    failure.add_argument("--video-artifacts", required=True)
    failure.add_argument("--out", required=True)

    args = ap.parse_args()
    if args.mode == "dataset":
        return write_dataset_report(args)
    if args.mode == "final":
        return write_final_report(args)
    return write_failure_report(args)


if __name__ == "__main__":
    raise SystemExit(main())
