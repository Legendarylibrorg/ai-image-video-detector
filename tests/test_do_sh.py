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

    def test_best_profile_includes_local_sources_when_hf_only_disabled(self) -> None:
        out = self.run_bash(
            "source scripts/do.sh; "
            "BEST_DS_HF_ONLY=0; "
            "BEST_DS_LOCAL_SOURCES='/tmp/a,/tmp/b'; "
            "print_image_collection_args best"
        )

        self.assertIn("--local-source\n/tmp/a\n", out)
        self.assertIn("--local-source\n/tmp/b\n", out)
        self.assertNotIn("--hf-only\n", out)

    def test_best_profile_defaults_to_hf_only(self) -> None:
        out = self.run_bash("source scripts/do.sh; print_image_collection_args best")
        self.assertIn("--hf-only\n", out)
        self.assertIn("--quiet-progress\n", out)

    def test_best_profile_emits_split_source_diversity_gate(self) -> None:
        out = self.run_bash("source scripts/do.sh; print_image_collection_args best")
        self.assertIn("--min-hf-sources-per-split-class\n10\n", out)
        self.assertIn("--max-per-source-class\n12000\n", out)
        self.assertIn("--max-per-source-split-class\n4000\n", out)
        self.assertIn("--hardneg-fraction\n0.35\n", out)

    def test_fast_profile_defaults_to_hf_only_and_split_source_gate(self) -> None:
        out = self.run_bash("source scripts/do.sh; print_image_collection_args fast")
        self.assertIn("--hf-only\n", out)
        self.assertIn("--min-hf-sources-per-split-class\n6\n", out)
        self.assertIn("--max-per-source-class\n3000\n", out)
        self.assertIn("--max-per-source-split-class\n1000\n", out)
        self.assertIn("--hardneg-fraction\n0.25\n", out)

    def test_usage_lists_simple_aliases(self) -> None:
        out = self.run_bash("source scripts/do.sh; print_usage")
        self.assertIn("pipeline", out)
        self.assertIn("run", out)
        self.assertIn("smoke", out)
        self.assertIn("smoke-real", out)
        self.assertIn("collection-status", out)
        self.assertIn("check", out)
        self.assertIn("train-existing", out)
        self.assertIn("retrain", out)
        self.assertIn("continuous", out)
        self.assertNotIn("detect", out)

    def test_pipeline_stage_is_resumable(self) -> None:
        out = self.run_bash(
            "tmpdir=$(mktemp -d); "
            "source scripts/do.sh; "
            "PIPELINE_STAGE_DIR=\"$tmpdir\"; "
            "call_file=\"$tmpdir/calls\"; "
            "run_pipeline_stage sample bash -lc 'echo first >> \"$0\"' \"$call_file\"; "
            "run_pipeline_stage sample bash -lc 'echo second >> \"$0\"' \"$call_file\"; "
            "wc -l < \"$call_file\""
        )
        self.assertEqual(out.strip().splitlines()[-1].strip(), "1")

    def test_pipeline_stage_retries_before_success(self) -> None:
        out = self.run_bash(
            "tmpdir=$(mktemp -d); "
            "source scripts/do.sh; "
            "PIPELINE_STAGE_DIR=\"$tmpdir\"; "
            "PIPELINE_MAX_ATTEMPTS=3; "
            "PIPELINE_RETRY_SLEEP_SEC=0; "
            "attempt_file=\"$tmpdir/attempts\"; "
            "run_pipeline_stage retry bash -lc "
            "'count=0; "
            "[[ -f \"$0\" ]] && count=$(cat \"$0\"); "
            "count=$((count + 1)); "
            "printf \"%s\\n\" \"$count\" > \"$0\"; "
            "[[ \"$count\" -ge 2 ]]' "
            "\"$attempt_file\"; "
            "cat \"$attempt_file\""
        )
        self.assertEqual(out.strip().splitlines()[-1], "2")

    def test_diverse_profile_emits_rate_limit_tuned_flags(self) -> None:
        out = self.run_bash(
            "source scripts/do.sh; "
            "print_diverse_common_args; "
            "print_diverse_discovery_args"
        )
        self.assertIn("--repo-base-pause-ms\n150\n", out)
        self.assertIn("--repo-cooldown-ms\n15000\n", out)
        self.assertIn("--transient-error-cooldown-ms\n2500\n", out)
        self.assertIn("--hf-query-pause-ms\n900\n", out)

    def test_collect_fast_data_builds_args_without_mapfile(self) -> None:
        out = self.run_bash(
            "source scripts/do.sh; "
            "run_image_dataset_build(){ printf '%q\\n' \"$@\"; }; "
            "FAST_HF_QUERIES='query one,query two'; "
            "collect_fast_data"
        )
        self.assertIn("./data_best_fast", out)
        self.assertIn("query\\ one\\,query\\ two", out)
        self.assertIn("--hf-only", out)
        self.assertIn("--require-full-targets", out)

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
            "printf 'MALWARE_SCAN=1\\nFAST_HF_ONLY=1\\n' > \"$tmpenv\"; "
            "ENV_FILE=\"$tmpenv\"; "
            "MALWARE_SCAN=0; "
            "FAST_HF_ONLY=0; "
            "source scripts/do.sh; "
            "load_env_file; "
            "printf '%s %s\\n' \"$MALWARE_SCAN\" \"$FAST_HF_ONLY\""
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

    def test_run_command_fails_fast_when_gpu_missing(self) -> None:
        proc = subprocess.run(
            ["bash", "scripts/do.sh", "run"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

        self.assertEqual(proc.returncode, 1)
        self.assertIn("gpu_required=1 reason=nvidia_smi_missing", proc.stderr)


if __name__ == "__main__":
    unittest.main()
