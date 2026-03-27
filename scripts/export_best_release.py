from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _resolve_checkpoint(path: Path) -> Path | None:
    safe = path.with_suffix(".safetensors")
    if safe.exists():
        return safe
    if path.exists():
        return path
    return None


def _copy_tree_file(path: Path, rel: str, out_dir: Path, copied: list[str]) -> None:
    if _copy_if_exists(path, out_dir / rel):
        copied.append(rel)


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

    prod_manifest = _read_json(ens_out / "prod_manifest.json")
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

    for model_dir in sorted((p for p in ens_out.glob("m*") if p.is_dir()), key=lambda p: p.name):
        rel_base = model_dir.name
        preferred = _resolve_checkpoint(model_dir / "best.safetensors")
        if preferred is not None:
            _copy_tree_file(preferred, f"{rel_base}/{preferred.name}", release_dir, copied)
        for name in (
            "calibration.json",
            "best_model_summary.json",
            "best_metrics.json",
            "best_group_metrics.json",
            "config.json",
            "best_checkpoint.txt",
            "test_metrics.json",
        ):
            _copy_tree_file(model_dir / name, f"{rel_base}/{name}", release_dir, copied)

    distill_dir = ens_out / "distill"
    if distill_dir.exists():
        preferred = _resolve_checkpoint(distill_dir / "best.safetensors")
        if preferred is not None:
            _copy_tree_file(preferred, f"distill/{preferred.name}", release_dir, copied)
        for name in ("best_model_summary.json", "best_checkpoint.txt", "config.json"):
            _copy_tree_file(distill_dir / name, f"distill/{name}", release_dir, copied)

    if args.video_artifacts:
        video_dir = Path(args.video_artifacts)
        preferred = _resolve_checkpoint(video_dir / "best_video.safetensors")
        if preferred is not None:
            _copy_tree_file(preferred, preferred.name, release_dir, copied)
        for name in ("best_video_metrics.json", "config.json"):
            _copy_tree_file(video_dir / name, name, release_dir, copied)

    copied_with_manifest = [*copied, "release_manifest.json"]
    release_manifest = {
        "schema": "ai-image-detector-release-bundle-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_ensemble_dir": str(ens_out.resolve()),
        "release_dir": str(release_dir.resolve()),
        "prod_manifest": str((release_dir / "prod_manifest.json").resolve()) if (release_dir / "prod_manifest.json").exists() else None,
        "public_models": prod_manifest.get("models", []),
        "copied_files": copied_with_manifest,
    }
    (release_dir / "release_manifest.json").write_text(json.dumps(release_manifest, indent=2), encoding="utf-8")

    (ens_out / "latest_release.txt").write_text(str(release_dir.resolve()), encoding="utf-8")
    print(f"saved_release={release_dir}")


if __name__ == "__main__":
    main()
