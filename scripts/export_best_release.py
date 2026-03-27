from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from release_selection import build_inference_profile, build_public_model_manifest, select_public_model
from script_support import iter_member_dirs, read_json_dict, resolve_checkpoint, write_json_dict


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _copy_tree_file(path: Path, rel: str, out_dir: Path, copied: list[str]) -> None:
    if _copy_if_exists(path, out_dir / rel):
        copied.append(rel)


def _copy_named_files(src_dir: Path, dst_dir: Path, names: tuple[str, ...], copied: list[str], *, rel_prefix: str) -> None:
    for name in names:
        target_rel = f"{rel_prefix}/{name}" if rel_prefix else name
        _copy_tree_file(src_dir / name, target_rel, dst_dir, copied)


def main() -> None:
    ap = argparse.ArgumentParser(description="Export a release-ready artifact bundle from the ensemble output directory")
    ap.add_argument("--ens-out", required=True, help="Ensemble output directory")
    ap.add_argument("--video-artifacts", default="", help="Optional video artifact directory")
    ap.add_argument("--out", default="", help="Optional explicit release bundle directory")
    args = ap.parse_args()

    ens_out = Path(args.ens_out)
    if not ens_out.exists():
        raise FileNotFoundError(f"missing ensemble output directory: {ens_out}")

    if args.out:
        release_dir = Path(args.out)
    else:
        release_dir = ens_out / "releases" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    release_dir.mkdir(parents=True, exist_ok=True)

    prod_manifest = read_json_dict(ens_out / "prod_manifest.json")
    copied: list[str] = []

    for name in (
        "prod_manifest.json",
        "run_manifest.json",
        "final_run_summary.json",
        "final_thresholds.json",
        "test_metrics.json",
        "robust_eval.json",
        "ensemble_config.json",
        "domain_config.json",
    ):
        _copy_tree_file(ens_out / name, name, release_dir, copied)

    for model_dir in iter_member_dirs(ens_out):
        rel_base = model_dir.name
        preferred = resolve_checkpoint(model_dir / "best.safetensors")
        if preferred is not None:
            _copy_tree_file(preferred, f"{rel_base}/{preferred.name}", release_dir, copied)
        _copy_named_files(
            model_dir,
            release_dir,
            (
                "calibration.json",
                "best_model_summary.json",
                "best_metrics.json",
                "best_group_metrics.json",
                "config.json",
                "inference_spec.json",
                "best_checkpoint.txt",
                "test_metrics.json",
            ),
            copied,
            rel_prefix=rel_base,
        )

    distill_dir = ens_out / "distill"
    if distill_dir.exists():
        preferred = resolve_checkpoint(distill_dir / "best.safetensors")
        if preferred is not None:
            _copy_tree_file(preferred, f"distill/{preferred.name}", release_dir, copied)
        _copy_named_files(
            distill_dir,
            release_dir,
            ("best_model_summary.json", "best_checkpoint.txt", "config.json"),
            copied,
            rel_prefix="distill",
        )

    if args.video_artifacts:
        video_dir = Path(args.video_artifacts)
        preferred = resolve_checkpoint(video_dir / "best_video.safetensors")
        if preferred is not None:
            _copy_tree_file(preferred, preferred.name, release_dir, copied)
        _copy_named_files(
            video_dir,
            release_dir,
            ("best_video_metrics.json", "config.json"),
            copied,
            rel_prefix="",
        )

    public_model = select_public_model(ens_out)
    public_model_manifest = None
    if public_model is not None:
        public_dir = release_dir / "public_model"
        public_dir.mkdir(parents=True, exist_ok=True)
        public_checkpoint = public_dir / "best.safetensors"
        if _copy_if_exists(public_model["source_checkpoint_path"], public_checkpoint):
            copied.append("public_model/best.safetensors")
        _copy_named_files(
            Path(public_model["artifact_dir"]),
            release_dir,
            ("calibration.json", "best_metrics.json", "test_metrics.json", "config.json", "inference_spec.json", "best_model_summary.json"),
            copied,
            rel_prefix="public_model",
        )
        public_model_manifest = build_public_model_manifest(
            public_model,
            public_checkpoint=str(public_checkpoint.resolve()),
        )
        if public_model_manifest is not None:
            write_json_dict(public_dir / "model_manifest.json", public_model_manifest, indent=2)
            copied.append("public_model/model_manifest.json")
        inference_profile = build_inference_profile(public_model)
        if inference_profile is not None:
            write_json_dict(public_dir / "inference_profile.json", inference_profile, indent=2)
            copied.append("public_model/inference_profile.json")
        (ens_out / "latest_public_model.txt").write_text(str(public_checkpoint.resolve()), encoding="utf-8")

    copied_with_manifest = [*copied, "release_manifest.json"]
    release_manifest = {
        "schema": "ai-image-detector-release-bundle-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_ensemble_dir": str(ens_out.resolve()),
        "release_dir": str(release_dir.resolve()),
        "prod_manifest": str((release_dir / "prod_manifest.json").resolve()) if (release_dir / "prod_manifest.json").exists() else None,
        "public_models": prod_manifest.get("models", []),
        "public_model": public_model_manifest,
        "copied_files": copied_with_manifest,
    }
    write_json_dict(release_dir / "release_manifest.json", release_manifest, indent=2)

    (ens_out / "latest_release.txt").write_text(str(release_dir.resolve()), encoding="utf-8")
    print(f"saved_release={release_dir}")


if __name__ == "__main__":
    main()
