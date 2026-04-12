"""Training run directory setup: config JSON, dataset manifest, inference spec."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .checkpoints import args_dict_for_checkpoint
from .dataset_integrity import (
    assert_no_train_val_hash_overlap,
    build_manifest_records,
    write_dataset_manifest,
)
from .model import model_runtime_spec
from .train_support import _dataset_counts
from .utils import git_commit
from .utils.jsonio import write_json_atomic


def prepare_training_output_dir(
    out: Path,
    data_root: Path,
    args: argparse.Namespace,
    *,
    train_samples: list[tuple[str, int]],
    val_samples: list[tuple[str, int]],
    classes: list[str],
    metadata_dim: int,
) -> dict[str, Any]:
    """Write ``config.json``, optional ``dataset_manifest.json``, and ``inference_spec.json``."""
    run_config: dict[str, Any] = {
        "args": args_dict_for_checkpoint(args),
        "git_commit": git_commit(),
        "dataset_counts": _dataset_counts(data_root),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_spec": model_runtime_spec(
            backbone=args.backbone,
            img_size=args.img_size,
            metadata_feature_dim=metadata_dim,
        ),
    }

    def _attach_manifest(train_recs: list[dict[str, Any]], val_recs: list[dict[str, Any]]) -> None:
        if args.dataset_manifest == "off":
            return
        write_dataset_manifest(
            out / "dataset_manifest.json",
            data_root=data_root,
            train_records=train_recs,
            val_records=val_recs,
        )
        run_config["dataset_manifest"] = "dataset_manifest.json"

    if args.strict_dataset:
        train_recs = build_manifest_records(train_samples, classes, data_root, hash_files=True)
        val_recs = build_manifest_records(val_samples, classes, data_root, hash_files=True)
        assert_no_train_val_hash_overlap(train_recs, val_recs)
        _attach_manifest(train_recs, val_recs)
    elif args.dataset_manifest != "off":
        hash_train = args.dataset_manifest == "full"
        train_recs = build_manifest_records(train_samples, classes, data_root, hash_files=hash_train)
        val_recs = build_manifest_records(val_samples, classes, data_root, hash_files=True)
        _attach_manifest(train_recs, val_recs)

    write_json_atomic(out / "config.json", run_config, indent=2)
    write_json_atomic(out / "inference_spec.json", run_config["runtime_spec"], indent=2)
    return run_config
