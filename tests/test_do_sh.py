from __future__ import annotations

from pathlib import Path
import tempfile
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DoShTests(unittest.TestCase):
    def run_bash(self, script: str) -> str:
        proc = subprocess.run(
            ["bash", "-lc", script],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return proc.stdout

    def run_bash_proc(self, script: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", "-lc", script],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

    def test_best_profile_defaults_to_hf_sources_only(self) -> None:
        out = self.run_bash("source scripts/do.sh; print_image_collection_args best")
        self.assertNotIn("--local-source\n", out)
        self.assertNotIn("--local-source-real\n", out)
        self.assertNotIn("--local-source-ai\n", out)
        self.assertIn("--cache-dir\n./.local/hf\n", out)
        self.assertIn("--hf-cache-only-if-present\n", out)
        self.assertIn("--quiet-progress\n", out)

    def test_best_profile_emits_split_source_diversity_gate(self) -> None:
        out = self.run_bash("source scripts/do.sh; print_image_collection_args best")
        self.assertIn("--min-hf-sources-per-split-class\n12\n", out)
        self.assertIn("--max-per-source-class\n10000\n", out)
        self.assertIn("--max-per-source-split-class\n3000\n", out)
        self.assertIn("--hardneg-fraction\n0.35\n", out)
        self.assertIn("--min-side\n160\n", out)
        self.assertIn("--max-aspect-ratio\n4.0\n", out)

    def test_fast_profile_defaults_to_hf_sources_only_and_split_source_gate(self) -> None:
        out = self.run_bash("source scripts/do.sh; print_image_collection_args fast")
        self.assertNotIn("--local-source\n", out)
        self.assertNotIn("--local-source-real\n", out)
        self.assertNotIn("--local-source-ai\n", out)
        self.assertIn("--cache-dir\n./.local/hf\n", out)
        self.assertIn("--min-hf-sources-per-split-class\n6\n", out)
        self.assertIn("--max-per-source-class\n3000\n", out)
        self.assertIn("--max-per-source-split-class\n1000\n", out)
        self.assertIn("--hardneg-fraction\n0.25\n", out)
        self.assertIn("--min-side\n160\n", out)
        self.assertIn("--max-aspect-ratio\n4.0\n", out)

    def test_usage_lists_simple_aliases(self) -> None:
        out = self.run_bash("source scripts/do.sh; print_usage")
        self.assertIn("pipeline", out)
        self.assertIn("smoke", out)
        self.assertIn("smoke-real", out)
        self.assertIn("collection-status", out)
        self.assertIn("train-existing", out)
        self.assertIn("retrain", out)
        self.assertIn("continuous", out)
        self.assertNotIn("detect", out)

    def test_no_arg_do_script_shows_usage_without_removed_start_alias(self) -> None:
        proc = self.run_bash_proc("bash scripts/do.sh")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("usage: bash scripts/do.sh", proc.stdout)
        self.assertNotIn("start", proc.stdout)

    def test_diverse_profile_emits_rate_limit_tuned_flags(self) -> None:
        out = self.run_bash(
            "source scripts/do.sh; "
            "print_diverse_common_args; "
            "print_diverse_discovery_args"
        )
        self.assertIn("--cache-dir\n./.local/hf\n", out)
        self.assertIn("--hf-cache-only-if-present\n", out)
        self.assertIn("--max-samples-per-source\n40000\n", out)
        self.assertIn("--acceptance-warmup-samples\n192\n", out)
        self.assertIn("--min-acceptance-rate\n0.02\n", out)
        self.assertIn("--max-per-source-class\n12000\n", out)
        self.assertIn("--max-per-source-split-class\n4000\n", out)
        self.assertIn("--min-hf-sources-per-class\n20\n", out)
        self.assertIn("--min-hf-sources-per-split-class\n12\n", out)
        self.assertIn("--hf-discovery-limit\n180\n", out)
        self.assertIn("--hf-max-sources\n480\n", out)
        self.assertIn("--hf-min-quality-score\n1.95\n", out)
        self.assertIn("--hf-print-top\n24\n", out)
        self.assertIn("--repo-base-pause-ms\n150\n", out)
        self.assertIn("--repo-cooldown-ms\n15000\n", out)
        self.assertIn("--transient-error-cooldown-ms\n2500\n", out)
        self.assertIn("--hf-query-pause-ms\n900\n", out)
        self.assertIn("--min-side\n160\n", out)
        self.assertIn("--max-aspect-ratio\n4.0\n", out)

    def test_video_profile_uses_shared_hf_cache(self) -> None:
        out = self.run_bash("source scripts/do.sh; print_video_collection_args")
        self.assertIn("--cache-dir\n./.local/hf\n", out)
        self.assertIn("--repo-base-pause-ms\n150\n", out)
        self.assertIn("--repo-cooldown-ms\n12000\n", out)
        self.assertIn("--min-video-bytes\n200000\n", out)

    def test_collect_diverse_cycle_uses_two_snapshot_workers_by_default(self) -> None:
        out = self.run_bash(
            "source scripts/do.sh; "
            "collect_diverse_image_data(){ :; }; "
            "ingest_outputs(){ :; }; "
            "collect_video_data(){ printf 'workers=%s\\n' \"$VIDEO_SNAPSHOT_MAX_WORKERS\"; }; "
            "collect_diverse_cycle"
        )
        self.assertIn("workers=2", out)

    def test_collect_fast_data_builds_args_without_mapfile(self) -> None:
        out = self.run_bash(
            "source scripts/do.sh; "
            "run_image_dataset_build(){ printf '%q\\n' \"$@\"; }; "
            "FAST_HF_QUERIES='query one,query two'; "
            "collect_fast_data"
        )
        self.assertIn("./data_best_fast", out)
        self.assertIn("query\\ one\\,query\\ two", out)
        self.assertIn("--require-full-targets", out)

    def test_profiled_image_collection_helper_is_shared(self) -> None:
        out = self.run_bash("source scripts/do.sh; declare -f collect_profiled_image_data; declare -f collect_image_data; declare -f collect_fast_data")
        self.assertIn("collect_profiled_image_data ()", out)
        self.assertIn('collect_profiled_image_data best "$out" "$query_csv"', out)
        self.assertIn('collect_profiled_image_data fast "$out" "$query_csv"', out)

    def test_collect_diverse_image_data_builds_cached_args_without_mapfile(self) -> None:
        out = self.run_bash(
            "tmpdir=$(mktemp -d); "
            "source scripts/do.sh; "
            "run_image_dataset_builder(){ printf 'builder:%q\\n' \"$@\"; }; "
            "run_malware_scan(){ printf 'scan:%q\\n' \"$@\"; }; "
            "run_repo_python(){ printf 'python:%q\\n' \"$@\"; }; "
            "DIVERSE_SKIP_DISCOVERY=1; "
            "DIVERSE_HF_CACHE_FILE=\"$tmpdir/cache.txt\"; "
            "printf 'repo/name\\n' > \"$DIVERSE_HF_CACHE_FILE\"; "
            "collect_diverse_image_data"
        )
        self.assertIn("builder:./data_best", out)
        self.assertIn("builder:--sources-file", out)
        self.assertIn("builder:", out)
        self.assertIn("python:scripts/audit_diversity.py", out)
        self.assertIn("scan:./data_best", out)

    def test_run_collection_command_requires_scanner_before_work(self) -> None:
        out = self.run_bash(
            "source scripts/do.sh; "
            "ensure_malware_scan_ready(){ echo ready; }; "
            "my_collect(){ echo collect; }; "
            "run_collection_command my_collect"
        )
        self.assertEqual(out.strip().splitlines(), ["ready", "collect"])

    def test_run_image_dataset_build_scans_before_and_after_build(self) -> None:
        out = self.run_bash(
            "tmpdir=.tmp_scan_fixture; "
            "rm -rf \"$tmpdir\"; "
            "mkdir -p \"$tmpdir/out\"; "
            "source scripts/do.sh; "
            "run_malware_scan(){ printf 'scan:%q\\n' \"$@\"; }; "
            "run_image_dataset_builder(){ printf 'builder:%q\\n' \"$@\"; }; "
            "run_image_dataset_build \"$tmpdir/out\" \"query one,query two\" --flag value; "
            "rm -rf \"$tmpdir\""
        )
        lines = out.strip().splitlines()
        self.assertGreaterEqual(len(lines), 3)
        self.assertEqual(lines[0], "scan:.tmp_scan_fixture/out")
        self.assertTrue(lines[1].startswith("builder:.tmp_scan_fixture/out"))
        self.assertEqual(lines[-1], "scan:.tmp_scan_fixture/out")

    def test_prepare_training_image_data_works_without_copy_flag(self) -> None:
        out = self.run_bash(
            "tmpdir=$(mktemp -d); "
            "source scripts/do.sh; "
            "ensure_env(){ :; }; "
            "python(){ python3 \"$@\"; }; "
            "DATA_DIR=\"$tmpdir/data_best\"; "
            "NEW_DATA_DST=\"$tmpdir/data_new/train\"; "
            "TRAIN_READY_DATA_DIR=\"$tmpdir/training_data\"; "
            "mkdir -p \"$tmpdir/data_best/train/ai\" \"$tmpdir/data_best/train/real\" "
            "\"$tmpdir/data_best/val/ai\" \"$tmpdir/data_best/val/real\" "
            "\"$tmpdir/data_best/test/ai\" \"$tmpdir/data_best/test/real\" "
            "\"$tmpdir/data_new/train/ai\" \"$tmpdir/data_new/train/real\"; "
            "touch \"$tmpdir/data_best/train/ai/base_ai.jpg\" "
            "\"$tmpdir/data_best/train/real/base_real.jpg\" "
            "\"$tmpdir/data_best/val/ai/val_ai.jpg\" "
            "\"$tmpdir/data_best/val/real/val_real.jpg\" "
            "\"$tmpdir/data_best/test/ai/test_ai.jpg\" "
            "\"$tmpdir/data_best/test/real/test_real.jpg\" "
            "\"$tmpdir/data_new/train/ai/inc_ai.jpg\" "
            "\"$tmpdir/data_new/train/real/inc_real.jpg\"; "
            "prepare_training_image_data >/dev/null; "
            "[[ -f \"$tmpdir/training_data/train/ai/base_ai.jpg\" ]] && "
            "[[ -f \"$tmpdir/training_data/train/ai/inc_ai.jpg\" ]] && "
            "[[ -f \"$tmpdir/training_data/train/real/inc_real.jpg\" ]] && "
            "echo ok"
        )
        self.assertEqual(out.strip().splitlines()[-1], "ok")

    def test_load_env_file_preserves_explicit_env_overrides(self) -> None:
        out = self.run_bash(
            "tmpenv=$(mktemp); "
            "printf 'MALWARE_SCAN=1\\nFAST_NO_DEFAULT_SOURCES=1\\n' > \"$tmpenv\"; "
            "ENV_FILE=\"$tmpenv\"; "
            "MALWARE_SCAN=0; "
            "FAST_NO_DEFAULT_SOURCES=0; "
            "source scripts/do.sh; "
            "load_env_file; "
            "printf '%s %s\\n' \"$MALWARE_SCAN\" \"$FAST_NO_DEFAULT_SOURCES\""
        )
        self.assertEqual(out.strip().splitlines()[-1], "0 0")

    def test_load_env_file_uses_env_file_when_override_is_empty_string(self) -> None:
        out = self.run_bash(
            "tmpenv=$(mktemp); "
            "printf 'HF_TOKEN=from_file\\n' > \"$tmpenv\"; "
            "ENV_FILE=\"$tmpenv\"; "
            "HF_TOKEN=''; "
            "source scripts/do.sh; "
            "load_env_file; "
            "printf '%s\\n' \"$HF_TOKEN\""
        )
        self.assertEqual(out.strip().splitlines()[-1], "from_file")

    def test_load_env_file_treats_env_as_data_not_shell_code(self) -> None:
        out = self.run_bash(
            "tmpenv=$(mktemp); "
            "marker=$(mktemp); rm -f \"$marker\"; "
            "printf 'HF_TOKEN=from_file\\nprintf hacked > %s\\n' \"$marker\" > \"$tmpenv\"; "
            "ENV_FILE=\"$tmpenv\"; "
            "source scripts/do.sh; "
            "load_env_file; "
            "[[ ! -e \"$marker\" ]] && printf '%s %s\\n' \"$HF_TOKEN\" safe"
        )
        self.assertEqual(out.strip().splitlines()[-1], "from_file safe")

    def test_load_env_file_ignores_inline_comments_for_plain_and_quoted_values(self) -> None:
        out = self.run_bash(
            "tmpenv=$(mktemp); "
            "printf 'HF_TOKEN=from_file # note\\nHUGGINGFACE_HUB_TOKEN=\"from_hub\" # note\\n' > \"$tmpenv\"; "
            "ENV_FILE=\"$tmpenv\"; "
            "HF_TOKEN=''; "
            "HUGGINGFACE_HUB_TOKEN=''; "
            "source scripts/do.sh; "
            "load_env_file; "
            "printf '%s %s\\n' \"$HF_TOKEN\" \"$HUGGINGFACE_HUB_TOKEN\""
        )
        self.assertEqual(out.strip().splitlines()[-1], "from_file from_hub")

    def test_train_existing_pipeline_passes_collected_and_prepared_roots_to_4090_wrapper(self) -> None:
        out = self.run_bash("source scripts/do.sh; declare -f run_prepared_max_quality_pipeline; declare -f train_existing_pipeline")
        self.assertIn('run_prepared_max_quality_pipeline ()', out)
        self.assertIn('PIPELINE_COLLECTED_DATA_DIR="$collected_root"', out)
        self.assertIn('PIPELINE_PREPARED_DATA_DIR="$PREPARED_IMAGE_DATA_DIR"', out)
        self.assertIn('TRAIN_READY_DATA_DIR="$PREPARED_IMAGE_DATA_DIR"', out)
        self.assertIn('run_prepared_max_quality_pipeline "$collected_root"', out)

    def test_require_pipeline_collection_data_rejects_partial_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "data_best"
            for split in ("train", "val", "test"):
                for cls in ("ai", "real"):
                    bucket = root / split / cls
                    bucket.mkdir(parents=True, exist_ok=True)
                    (bucket / f"{split}_{cls}.jpg").write_bytes(b"x")

            proc = self.run_bash_proc(
                f"source scripts/do.sh; "
                f"DATA_DIR='{root}'; "
                "PIPELINE_MIN_TRAIN_PER_CLASS=2; "
                "PIPELINE_MIN_VAL_PER_CLASS=1; "
                "PIPELINE_MIN_TEST_PER_CLASS=1; "
                "require_pipeline_collection_data \"$DATA_DIR\""
            )

        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("insufficient_image_bucket=", proc.stdout + proc.stderr)

    def test_require_pipeline_collection_data_accepts_successful_build_report_without_explicit_minima(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "data_best"
            for split in ("train", "val", "test"):
                for cls in ("ai", "real"):
                    bucket = root / split / cls
                    bucket.mkdir(parents=True, exist_ok=True)
                    (bucket / f"{split}_{cls}.jpg").write_bytes(b"x")
            (root / "dataset_build_report.json").write_text('{"full_targets_ok": true}\n', encoding="utf-8")

            proc = self.run_bash_proc(
                f"source scripts/do.sh; "
                f"DATA_DIR='{root}'; "
                "unset PIPELINE_MIN_TRAIN_PER_CLASS PIPELINE_MIN_VAL_PER_CLASS PIPELINE_MIN_TEST_PER_CLASS; "
                "unset TRAIN_PER_CLASS VAL_PER_CLASS TEST_PER_CLASS; "
                "require_pipeline_collection_data \"$DATA_DIR\""
            )

        self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
        self.assertIn("collection_min_counts=skipped reason=build_report_ok", proc.stdout + proc.stderr)

    def test_run_full_pipeline_uses_canonical_quality_wrapper(self) -> None:
        out = self.run_bash("source scripts/do.sh; declare -f run_full_pipeline")
        self.assertIn('with_training_lock bash scripts/max_quality_4090.sh', out)
        self.assertNotIn('run_pipeline_stage', out)

    def test_wait_for_training_to_finish_clears_stale_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_path = Path(tmpdir) / "training.lock"
            lock_path.write_text("old\n", encoding="utf-8")
            out = self.run_bash(
                f"source scripts/do.sh; "
                f"TRAIN_LOCK='{lock_path}'; "
                "TRAIN_LOCK_STALE_SEC=0; "
                "wait_for_training_to_finish test; "
                f"[[ ! -f '{lock_path}' ]] && echo cleared"
            )

        self.assertIn("training_lock=stale_cleared", out)
        self.assertEqual(out.strip().splitlines()[-1], "cleared")

    def test_preflight_command_requires_gpu(self) -> None:
        proc = subprocess.run(
            ["bash", "scripts/do.sh", "preflight"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

        self.assertEqual(proc.returncode, 2)
        self.assertIn("doctor_fail: nvidia_smi_missing gpu_required=1", proc.stdout)

    def test_pipeline_command_fails_fast_when_gpu_missing(self) -> None:
        proc = subprocess.run(
            ["bash", "scripts/do.sh", "pipeline"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

        self.assertEqual(proc.returncode, 1)
        self.assertIn("gpu_required=1 reason=nvidia_smi_missing", proc.stderr)


if __name__ == "__main__":
    unittest.main()
