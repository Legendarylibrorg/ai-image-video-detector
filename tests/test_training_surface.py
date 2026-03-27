from __future__ import annotations

from pathlib import Path
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
        self.assertIn('aid-train = "ai_image_detector.train:main"', text)
        self.assertIn('aid-video-train = "ai_image_detector.video_temporal:train_main"', text)
        self.assertIn('aid-dataset = "ai_image_detector.dataset_tools:main"', text)
        self.assertNotIn("fastapi", text.lower())
        self.assertNotIn("uvicorn", text.lower())

    def test_api_module_is_removed(self) -> None:
        self.assertFalse((ROOT / "src" / "ai_image_detector" / "api.py").exists())

    def test_local_retrain_defaults_to_train_existing(self) -> None:
        text = (ROOT / "scripts" / "local_retrain_4090.sh").read_text(encoding="utf-8")
        self.assertIn('PIPELINE_MODE="${PIPELINE_MODE:-}"', text)
        self.assertIn('pipeline_args=(bash scripts/do.sh train-existing)', text)
        self.assertNotIn("start-v2", text)
        self.assertIn("--skip-video", text)
        self.assertIn('--min-image-auc "${GATE_MIN_IMAGE_AUC:-0.96}"', text)
        self.assertIn('--min-image-f1 "${GATE_MIN_IMAGE_F1:-0.92}"', text)
        self.assertIn('--min-robust-worst-auc "${GATE_MIN_ROBUST_WORST_AUC:-0.90}"', text)
        self.assertIn('--max-robust-auc-drop "${GATE_MAX_ROBUST_AUC_DROP:-0.08}"', text)
        self.assertNotIn('eval "$PIPELINE_CMD"', text)

    def test_metadata_finetune_wrapper_uses_metadata_features_and_existing_checkpoint(self) -> None:
        text = (ROOT / "scripts" / "metadata_finetune_4090.sh").read_text(encoding="utf-8")
        self.assertIn("prepare_training_image_data", text)
        self.assertIn("--use-metadata-features", text)
        self.assertIn("--init-from", text)
        self.assertIn("artifacts_finetune_metadata", text)

    def test_train_module_skips_degenerate_best_checkpoint_promotion(self) -> None:
        text = (ROOT / "src" / "ai_image_detector" / "train.py").read_text(encoding="utf-8")
        self.assertIn("def _promotion_status(report: dict[str, Any]) -> tuple[bool, str]:", text)
        self.assertIn('return False, "no_operable_threshold"', text)
        self.assertIn('return False, "single_class_predictions"', text)
        self.assertIn('return False, "uninformative_balanced_threshold"', text)
        self.assertIn('report["promotion_eligible"] = bool(promotion_eligible)', text)
        self.assertIn('report["threshold_operable"] = bool(threshold_metrics.get("operable", True))', text)
        self.assertIn('report["threshold_search_status"] = str(threshold_metrics.get("search_status", "unknown"))', text)
        self.assertIn("skip_best_checkpoint epoch={}", text)
        self.assertIn('raise RuntimeError("no_promotable_checkpoint")', text)

    def test_reference_doc_stays_pipeline_focused(self) -> None:
        text = (ROOT / "docs" / "REFERENCE.md").read_text(encoding="utf-8")
        self.assertIn("Pipeline tools", text)
        self.assertIn("RTX 4090", text)
        self.assertNotIn("aid-detect --model", text)
        self.assertNotIn("aid-explain --model", text)
        self.assertNotIn("aid-video-detect-temporal", text)

    def test_max_quality_wrapper_enables_quality_thresholding_and_refinement(self) -> None:
        text = (ROOT / "scripts" / "max_quality_4090.sh").read_text(encoding="utf-8")
        self.assertIn('export RUN_HARD_RETRAIN="${RUN_HARD_RETRAIN:-1}"', text)
        self.assertIn('export RUN_DOMAIN_THRESHOLDS="${RUN_DOMAIN_THRESHOLDS:-1}"', text)
        self.assertIn('export RUN_ROBUST_EVAL="${RUN_ROBUST_EVAL:-1}"', text)
        self.assertIn('export BEST_DS_HF_MIN_QUALITY_SCORE="${BEST_DS_HF_MIN_QUALITY_SCORE:-1.95}"', text)
        self.assertIn('export BEST_DS_MAX_PER_SOURCE_CLASS="${BEST_DS_MAX_PER_SOURCE_CLASS:-12000}"', text)
        self.assertIn('export BEST_DS_MAX_PER_SOURCE_SPLIT_CLASS="${BEST_DS_MAX_PER_SOURCE_SPLIT_CLASS:-3500}"', text)
        self.assertIn('export BEST_DS_MIN_HF_SOURCES_PER_SPLIT_CLASS="${BEST_DS_MIN_HF_SOURCES_PER_SPLIT_CLASS:-10}"', text)
        self.assertIn('export TRAIN_PATIENCE="${TRAIN_PATIENCE:-5}"', text)
        self.assertIn('export VIDEO_MIN_BYTES="${VIDEO_MIN_BYTES:-200000}"', text)

    def test_full_pipeline_passes_quality_collection_guardrails(self) -> None:
        text = (ROOT / "scripts" / "full_pipeline_4090.sh").read_text(encoding="utf-8")
        self.assertIn('--max-per-source-split-class "$BEST_DS_MAX_PER_SOURCE_SPLIT_CLASS"', text)
        self.assertIn('--min-hf-sources-per-split-class "$BEST_DS_MIN_HF_SOURCES_PER_SPLIT_CLASS"', text)
        self.assertIn('--hf-query-pause-ms "$BEST_DS_HF_QUERY_PAUSE_MS"', text)
        self.assertIn('--transient-error-cooldown-ms "$BEST_DS_TRANSIENT_ERROR_COOLDOWN_MS"', text)
        self.assertIn('--min-video-bytes "$VIDEO_MIN_BYTES"', text)

    def test_continuous_training_runs_collection_before_retrain(self) -> None:
        text = (ROOT / "scripts" / "continuous_training.sh").read_text(encoding="utf-8")
        self.assertIn("Continuous collection + retraining loop", text)
        self.assertIn('source "$ROOT_DIR/scripts/lib/core.sh"', text)
        self.assertIn('PIPELINE_WAIT_FOR_TRAINING_SEC="${PIPELINE_WAIT_FOR_TRAINING_SEC:-$CHECK_WHILE_TRAINING_SEC}"', text)
        self.assertIn('wait_for_training_to_finish "continuous_training"', text)
        self.assertIn("continuous_training_collect_start", text)
        self.assertIn("bash scripts/do.sh collect", text)
        self.assertIn("continuous_training_retrain_start", text)
        self.assertIn("bash scripts/weekly_retrain_v3.sh", text)
        self.assertNotIn("is_training_active()", text)

    def test_weekly_retrain_uses_repo_python_for_review_queue_ingest(self) -> None:
        text = (ROOT / "scripts" / "weekly_retrain_v3.sh").read_text(encoding="utf-8")
        self.assertIn('source "$ROOT_DIR/scripts/lib/core.sh"', text)
        self.assertIn("run_repo_python scripts/review_queue_to_dataset.py", text)
        self.assertNotIn("\npython scripts/review_queue_to_dataset.py", text)


if __name__ == "__main__":
    unittest.main()
