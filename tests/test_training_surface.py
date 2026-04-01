from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TrainingSurfaceTests(unittest.TestCase):
    def test_pyproject_keeps_only_pipeline_entrypoints_and_no_web_dependencies(self) -> None:
        text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertNotIn("aid-serve", text)
        self.assertNotIn("aid-detect", text)
        self.assertNotIn("aid-explain", text)
        self.assertNotIn("aid-metadata", text)
        self.assertNotIn("aid-robust-eval", text)
        self.assertNotIn("aid-video-detect", text)
        self.assertNotIn("aid-video-detect-temporal", text)
        self.assertNotIn("aid-train-advanced", text)
        self.assertIn('aid-train = "ai_image_detector.cli:train_main"', text)
        self.assertIn('aid-video-train = "ai_image_detector.cli:video_train_main"', text)
        self.assertNotIn("aid-dataset =", text)
        self.assertNotIn("fastapi", text.lower())
        self.assertNotIn("uvicorn", text.lower())

    def test_api_module_is_removed(self) -> None:
        self.assertFalse((ROOT / "src" / "ai_image_detector" / "api.py").exists())
        self.assertFalse((ROOT / "src" / "ai_image_detector" / "multimodal.py").exists())

    def test_cli_wrapper_import_stays_lightweight(self) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "import ai_image_detector.cli as c; "
                    "print(int('torch' in sys.modules)); "
                    "print(int('cv2' in sys.modules)); "
                    "print(hasattr(c, 'train_main')); "
                    "print(hasattr(c, 'video_train_main')); "
                    "print(hasattr(c, 'dataset_main'))"
                ),
            ],
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
            check=True,
            capture_output=True,
            text=True,
        )
        lines = proc.stdout.strip().splitlines()
        self.assertEqual(lines[0], "0")
        self.assertEqual(lines[1], "0")
        self.assertEqual(lines[2], "True")
        self.assertEqual(lines[3], "True")
        self.assertEqual(lines[4], "False")

    def test_retrain_logic_lives_in_shared_training_helpers(self) -> None:
        text = (ROOT / "scripts" / "lib" / "training.sh").read_text(encoding="utf-8")
        self.assertIn("run_prepared_max_quality_pipeline()", text)
        self.assertIn("run_benchmark_gate()", text)
        self.assertIn("run_retrain_pipeline()", text)
        self.assertIn("bash scripts/do.sh train-existing", text)
        self.assertIn("--skip-video", text)
        self.assertIn('--min-image-auc "${GATE_MIN_IMAGE_AUC:-0.96}"', text)
        self.assertIn('--min-image-f1 "${GATE_MIN_IMAGE_F1:-0.92}"', text)
        self.assertIn('--min-robust-worst-auc "${GATE_MIN_ROBUST_WORST_AUC:-0.90}"', text)
        self.assertIn('--max-robust-auc-drop "${GATE_MAX_ROBUST_AUC_DROP:-0.08}"', text)
        self.assertIn("run_repo_python scripts/benchmark_gate.py", text)
        self.assertIn('run_prepared_max_quality_pipeline "$collected_root" 1', text)
        self.assertIn('run_prepared_max_quality_pipeline "$collected_root"', text)
        self.assertNotIn("train_image_pipeline()", text)
        self.assertNotIn("train_all_pipeline()", text)
        self.assertNotIn("train_video_only()", text)
        self.assertFalse((ROOT / "scripts" / "local_retrain_4090.sh").exists())

    def test_metadata_finetune_wrapper_uses_metadata_features_and_existing_checkpoint(self) -> None:
        text = (ROOT / "scripts" / "metadata_finetune_4090.sh").read_text(encoding="utf-8")
        self.assertIn("prepare_training_image_data", text)
        self.assertIn("--use-metadata-features", text)
        self.assertIn("--init-from", text)
        self.assertIn("artifacts_finetune_metadata", text)
        self.assertNotIn("best.pt", text)

    def test_train_module_skips_degenerate_best_checkpoint_promotion(self) -> None:
        text = (ROOT / "src" / "ai_image_detector" / "train.py").read_text(encoding="utf-8")
        self.assertIn("def _promotion_status(report: dict[str, Any]) -> tuple[bool, str]:", text)
        self.assertIn('choices=["tiny", "effb0", "effb2", "convnext_tiny", "convnext_small"]', text)
        self.assertIn("--warmup-epochs", text)
        self.assertIn("--mixup-alpha", text)
        self.assertIn("--label-smoothing", text)
        self.assertIn("configure_torch_runtime(device, args.deterministic)", text)
        self.assertIn('"git_commit": git_commit()', text)
        self.assertIn("build_adamw(model.parameters()", text)
        self.assertIn("save_training_checkpoint(", text)
        self.assertIn("build_loader_kwargs(num_workers=args.num_workers)", text)
        self.assertIn("make_eval_transform(args.img_size)", text)
        self.assertIn("unpack_image_batch", text)
        self.assertIn('return False, "no_operable_threshold"', text)
        self.assertIn('return False, "single_class_predictions"', text)
        self.assertIn('return False, "uninformative_balanced_threshold"', text)
        self.assertIn('report["promotion_eligible"] = bool(promotion_eligible)', text)
        self.assertIn('report["threshold_operable"] = bool(threshold_metrics.get("operable", True))', text)
        self.assertIn('report["threshold_search_status"] = str(threshold_metrics.get("search_status", "unknown"))', text)
        self.assertIn("skip_best_checkpoint epoch={}", text)
        self.assertIn('raise RuntimeError("no_promotable_checkpoint")', text)
        self.assertNotIn('torch.save(ckpt, out / "best.pt")', text)
        self.assertIn('save_safetensors_checkpoint(out / "best.safetensors", ckpt)', text)
        self.assertIn("write_timestamped_release(", text)
        self.assertIn('write_json_atomic(out / "inference_spec.json"', text)
        self.assertIn('runtime_spec": model_runtime_spec(', text)
        self.assertIn('(out / "best_checkpoint.txt").write_text(preferred_best.name', text)
        self.assertNotIn('--save-safetensors', text)
        self.assertNotIn('--no-save-safetensors', text)

    def test_reference_doc_stays_pipeline_focused(self) -> None:
        text = (ROOT / "docs" / "REFERENCE.md").read_text(encoding="utf-8")
        self.assertIn("[COMMANDS.md](COMMANDS.md)", text)
        self.assertIn("Documentation map", text)
        self.assertIn("Environment variables (`AID_*`)", text)
        self.assertIn("`aid-train` dataset integrity flags", text)
        self.assertIn("--strict-dataset", text)
        self.assertIn("Modern training stack (`aid-train`)", text)
        self.assertIn("Pipeline tools", text)
        self.assertIn("RTX 4090", text)
        self.assertNotIn("collect-diverse", text)
        self.assertNotIn("aid-detect --model", text)
        self.assertNotIn("aid-explain --model", text)
        self.assertNotIn("aid-video-detect-temporal", text)

    def test_max_quality_profile_env_enables_quality_thresholding_and_refinement(self) -> None:
        text = (ROOT / "scripts" / "lib" / "pipeline_max_quality_env.inc.sh").read_text(encoding="utf-8")
        self.assertIn('export RUN_HARD_RETRAIN="${RUN_HARD_RETRAIN:-1}"', text)
        self.assertIn('export RUN_DOMAIN_THRESHOLDS="${RUN_DOMAIN_THRESHOLDS:-1}"', text)
        self.assertIn('export RUN_ROBUST_EVAL="${RUN_ROBUST_EVAL:-1}"', text)
        self.assertIn('export BEST_DS_HF_MIN_QUALITY_SCORE="${BEST_DS_HF_MIN_QUALITY_SCORE:-1.45}"', text)
        self.assertIn('export BEST_DS_HF_DISCOVERY_WORKERS="${BEST_DS_HF_DISCOVERY_WORKERS:-12}"', text)
        self.assertIn('export VIDEO_SNAPSHOT_MAX_WORKERS="${VIDEO_SNAPSHOT_MAX_WORKERS:-8}"', text)
        self.assertIn('export BEST_DS_MAX_PER_SOURCE_CLASS="${BEST_DS_MAX_PER_SOURCE_CLASS:-7000}"', text)
        self.assertIn('export BEST_DS_MAX_PER_SOURCE_SPLIT_CLASS="${BEST_DS_MAX_PER_SOURCE_SPLIT_CLASS:-2000}"', text)
        self.assertIn('export BEST_DS_MIN_HF_SOURCES_PER_SPLIT_CLASS="${BEST_DS_MIN_HF_SOURCES_PER_SPLIT_CLASS:-16}"', text)
        self.assertIn('export BEST_DS_MIN_SIDE="${BEST_DS_MIN_SIDE:-160}"', text)
        self.assertIn('export BEST_DS_MAX_ASPECT_RATIO="${BEST_DS_MAX_ASPECT_RATIO:-4.0}"', text)
        self.assertIn('export TRAIN_PATIENCE="${TRAIN_PATIENCE:-5}"', text)
        self.assertIn('export VIDEO_MIN_BYTES="${VIDEO_MIN_BYTES:-200000}"', text)

    def test_train_ensemble_uses_shared_member_train_helper(self) -> None:
        text = (ROOT / "scripts" / "train_ensemble.sh").read_text(encoding="utf-8")
        self.assertIn("run_member_train()", text)
        self.assertIn('run_member_train "$OUT_DIR/m1"', text)
        self.assertIn('run_member_train "$OUT_DIR/m2"', text)
        self.assertIn('run_member_train "$OUT_DIR/m3"', text)
        self.assertIn('run_member_train "$OUT_DIR/m4"', text)
        self.assertIn("--backbone convnext_tiny", text)
        self.assertIn('--threshold-objective balanced', text)
        self.assertIn('"${common_args[@]}"', text)

    def test_full_pipeline_passes_quality_collection_guardrails(self) -> None:
        text = (ROOT / "scripts" / "full_pipeline_4090.sh").read_text(encoding="utf-8")
        self.assertIn('PIPELINE_PROFILE="${PIPELINE_PROFILE:-standard}"', text)
        self.assertIn('pipeline_max_quality_env.inc.sh', text)
        self.assertIn('--max-per-source-split-class "$BEST_DS_MAX_PER_SOURCE_SPLIT_CLASS"', text)
        self.assertIn('--min-hf-sources-per-split-class "$BEST_DS_MIN_HF_SOURCES_PER_SPLIT_CLASS"', text)
        self.assertIn('--hf-discovery-workers "$BEST_DS_HF_DISCOVERY_WORKERS"', text)
        self.assertIn('--hf-query-pause-ms "$BEST_DS_HF_QUERY_PAUSE_MS"', text)
        self.assertIn('--transient-error-cooldown-ms "$BEST_DS_TRANSIENT_ERROR_COOLDOWN_MS"', text)
        self.assertNotIn("BEST_DS_LOCAL_", text)
        self.assertNotIn("--local-source", text)
        self.assertIn('--min-video-bytes "$VIDEO_MIN_BYTES"', text)
        self.assertIn('VIDEO_SNAPSHOT_MAX_WORKERS="${VIDEO_SNAPSHOT_MAX_WORKERS:-8}"', text)
        self.assertIn('PIPELINE_RELEASE_OUT="${PIPELINE_RELEASE_OUT:-$ENS_OUT/release}"', text)
        self.assertIn("write_release_bundle()", text)
        self.assertIn("scripts/export_best_release.py", text)

    def test_report_writer_no_longer_tracks_removed_fast_profile_env(self) -> None:
        text = (ROOT / "scripts" / "write_pipeline_report.py").read_text(encoding="utf-8")
        self.assertNotIn('"FAST_"', text)

    def test_distill_script_stays_safetensors_only_for_best_artifacts(self) -> None:
        text = (ROOT / "scripts" / "train_distill.py").read_text(encoding="utf-8")
        self.assertIn('choices=["tiny", "effb0", "convnext_tiny", "convnext_small"]', text)
        self.assertIn("--num-workers", text)
        self.assertIn("--amp", text)
        self.assertIn('"git_commit": git_commit()', text)
        self.assertIn("build_loader_kwargs(num_workers=args.num_workers)", text)
        self.assertIn("make_eval_transform(args.img_size)", text)
        self.assertIn("build_adamw(student.parameters()", text)
        self.assertIn("write_timestamped_release(", text)
        self.assertIn('torch.amp.GradScaler("cuda", enabled=use_amp)', text)
        self.assertIn('save_safetensors_checkpoint(out / "best.safetensors", ckpt)', text)
        self.assertNotIn('--save-safetensors', text)
        self.assertNotIn('--no-save-safetensors', text)

    def test_video_training_uses_shared_runtime_helpers(self) -> None:
        text = (ROOT / "src" / "ai_image_detector" / "video_temporal.py").read_text(encoding="utf-8")
        self.assertIn("configure_torch_runtime(device, args.deterministic)", text)
        self.assertIn('"git_commit": git_commit()', text)
        self.assertIn("build_loader_kwargs(num_workers=args.num_workers, prefetch_factor=2)", text)
        self.assertIn("build_adamw(model.parameters()", text)
        self.assertIn("save_training_checkpoint(", text)
        self.assertIn("write_timestamped_release(", text)
        self.assertIn('torch.amp.GradScaler("cuda", enabled=use_amp)', text)

    def test_continuous_training_runs_collection_before_retrain(self) -> None:
        text = (ROOT / "scripts" / "continuous_training.sh").read_text(encoding="utf-8")
        self.assertIn("Continuous collection + retraining loop", text)
        self.assertIn('source "$ROOT_DIR/scripts/lib/core.sh"', text)
        self.assertIn('source "$ROOT_DIR/scripts/lib/training.sh"', text)
        self.assertIn('PIPELINE_WAIT_FOR_TRAINING_SEC="${PIPELINE_WAIT_FOR_TRAINING_SEC:-$CHECK_WHILE_TRAINING_SEC}"', text)
        self.assertIn('wait_for_training_to_finish "continuous_training"', text)
        self.assertIn("continuous_training_collect_start", text)
        self.assertIn("bash scripts/do.sh collect", text)
        self.assertIn("continuous_training_retrain_start", text)
        self.assertIn("run_weekly_retrain_cycle", text)
        self.assertNotIn("is_training_active()", text)

    def test_weekly_retrain_logic_uses_repo_python_for_review_queue_ingest(self) -> None:
        text = (ROOT / "scripts" / "lib" / "training.sh").read_text(encoding="utf-8")
        self.assertIn("run_review_queue_ingest()", text)
        self.assertIn("run_repo_python scripts/review_queue_to_dataset.py", text)
        self.assertNotIn("\npython scripts/review_queue_to_dataset.py", text)
        self.assertIn("run_weekly_retrain_cycle()", text)
        self.assertFalse((ROOT / "scripts" / "weekly_retrain_v3.sh").exists())


if __name__ == "__main__":
    unittest.main()
