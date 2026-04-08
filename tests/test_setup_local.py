from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import unittest

from _support import ROOT


class SetupLinuxSurfaceTests(unittest.TestCase):
    def run_bash(self, script: str, *, input_text: str = "") -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", "-lc", script],
            cwd=ROOT,
            input=input_text,
            capture_output=True,
            text=True,
            check=True,
        )

    def test_prompt_for_hf_token_saves_value_from_stdin(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text("HF_TOKEN=''\n", encoding="utf-8")
            proc = self.run_bash(
                "source scripts/setup_linux.sh; "
                f"ENV_FILE='{env_file}'; "
                "HF_TOKEN=''; "
                "HUGGINGFACE_HUB_TOKEN=''; "
                "SETUP_PROMPT_FOR_HF_TOKEN=1; "
                "SETUP_ALLOW_STDIN_TOKEN=1; "
                "prompt_for_hf_token_if_missing; "
                "printf 'token=%s\\n' \"$HF_TOKEN\"; "
                f"cat '{env_file}'",
                input_text="from_prompt\n",
            )

        self.assertIn("setup_stage=env_token status=done", proc.stdout)
        self.assertIn("token=from_prompt", proc.stdout)
        self.assertIn("HF_TOKEN='from_prompt'", proc.stdout)

    def test_prompt_for_hf_token_is_skipped_when_opted_out(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text("HF_TOKEN=''\n", encoding="utf-8")
            proc = self.run_bash(
                "source scripts/setup_linux.sh; "
                f"ENV_FILE='{env_file}'; "
                "HF_TOKEN=''; "
                "HUGGINGFACE_HUB_TOKEN=''; "
                "SETUP_PROMPT_FOR_HF_TOKEN=0; "
                "prompt_for_hf_token_if_missing",
            )

        self.assertIn("setup_stage=env_token status=skip_opt_out", proc.stdout)

    def test_setup_linux_defaults_to_nonblocking_token_behavior(self) -> None:
        proc = self.run_bash(
            "source scripts/setup_linux.sh; "
            "printf 'prompt=%s attempts=%s sleep=%s\\n' "
            "\"$SETUP_PROMPT_FOR_HF_TOKEN\" \"$SETUP_MAX_ATTEMPTS\" \"$SETUP_RETRY_SLEEP_SEC\""
        )

        self.assertIn("prompt=0 attempts=2 sleep=5", proc.stdout)

    def test_persist_env_hf_token_if_present_uses_hub_token_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text("HF_TOKEN=''\n", encoding="utf-8")
            proc = self.run_bash(
                "source scripts/setup_linux.sh; "
                f"ENV_FILE='{env_file}'; "
                "HF_TOKEN=''; "
                "HUGGINGFACE_HUB_TOKEN='from_hub_env'; "
                "persist_env_hf_token_if_present; "
                "printf 'token=%s\\n' \"$HF_TOKEN\"; "
                f"cat '{env_file}'",
            )

        self.assertIn("setup_stage=env_token status=done", proc.stdout)
        self.assertIn("token=from_hub_env", proc.stdout)
        self.assertIn("HF_TOKEN='from_hub_env'", proc.stdout)

    def test_persist_env_hf_token_if_present_respects_save_env_opt_out(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text("HF_TOKEN=''\n", encoding="utf-8")
            proc = self.run_bash(
                "source scripts/setup_linux.sh; "
                f"ENV_FILE='{env_file}'; "
                "HF_TOKEN='from_env'; "
                "HF_SETUP_SAVE_ENV=0; "
                "persist_env_hf_token_if_present; "
                f"cat '{env_file}'",
            )

        self.assertEqual(proc.stdout.strip(), "HF_TOKEN=''")

    def test_persist_env_hf_token_if_present_uses_hf_login_cache_without_copying_to_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            env_file = tmp / ".env"
            env_file.write_text("HF_TOKEN=''\n", encoding="utf-8")
            token_dir = tmp / ".cache" / "huggingface"
            token_dir.mkdir(parents=True)
            (token_dir / "token").write_text("from_cached_login\n", encoding="utf-8")
            proc = self.run_bash(
                "source scripts/setup_linux.sh; "
                f"HOME='{tmp}'; "
                "HF_HOME=''; "
                "HF_TOKEN=''; "
                "HUGGINGFACE_HUB_TOKEN=''; "
                f"ENV_FILE='{env_file}'; "
                "persist_env_hf_token_if_present; "
                "printf 'token=%s\\n' \"$HF_TOKEN\"; "
                f"cat '{env_file}'",
            )

        self.assertIn("token=from_cached_login", proc.stdout)
        self.assertNotIn("setup_stage=env_token status=done", proc.stdout)
        self.assertTrue(proc.stdout.strip().endswith("HF_TOKEN=''"))

    def test_print_next_step_prefers_smoke_then_run(self) -> None:
        proc = self.run_bash(
            "source scripts/setup_linux.sh; "
            "HF_TOKEN='from_env'; "
            "HUGGINGFACE_HUB_TOKEN='from_env'; "
            "print_next_step"
        )

        self.assertIn("setup_next=run ./local.sh smoke, then ./local.sh run", proc.stdout)

    def test_print_next_step_for_collection_profile_prefers_collect_flow(self) -> None:
        proc = self.run_bash(
            "source scripts/setup_linux.sh; "
            "DEPS_EXTRA='collection'; "
            "HF_TOKEN='from_env'; "
            "HUGGINGFACE_HUB_TOKEN='from_env'; "
            "print_next_step"
        )

        self.assertIn("setup_next=run ./local.sh collect, then ./local.sh collect-status", proc.stdout)
        self.assertNotIn("./local.sh smoke", proc.stdout)
        self.assertNotIn("./local.sh run", proc.stdout)

    def test_print_next_step_for_training_profile_requires_persistent_data(self) -> None:
        proc = self.run_bash(
            "source scripts/setup_linux.sh; "
            "DEPS_EXTRA='training'; "
            "print_next_step"
        )

        self.assertIn(
            "setup_next=prepare ./data_best and optional ./data_new/train (plus ./video_data if you want video training), then run ./local.sh train",
            proc.stdout,
        )
        self.assertNotIn("./local.sh smoke", proc.stdout)

    def test_install_python_deps_reuses_stored_profile_when_env_is_unset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            venv_dir = Path(tmpdir) / "venv"
            venv_dir.mkdir()
            (venv_dir / ".deps_profile").write_text("collection\n", encoding="utf-8")
            proc = self.run_bash(
                "source scripts/setup_linux.sh; "
                f"VENV_DIR='{venv_dir}'; "
                "DEPS_EXTRA=''; "
                "run_setup_step_with_retry(){ printf 'cmd=%s\\n' \"$*\"; }; "
                "install_python_deps"
            )

        self.assertIn("cmd=python_deps env DEPS_EXTRA=collection UPGRADE_TOOLCHAIN=0 bash scripts/install_deps.sh", proc.stdout)

    def test_install_python_deps_skips_toolchain_upgrade_by_default(self) -> None:
        proc = self.run_bash(
            "source scripts/setup_linux.sh; "
            "run_setup_step_with_retry(){ printf 'upgrade=%s cmd=%s\\n' "
            "\"${UPGRADE_TOOLCHAIN:-}\" \"$*\"; }; "
            "install_python_deps"
        )

        self.assertIn("upgrade= cmd=python_deps env UPGRADE_TOOLCHAIN=0 bash scripts/install_deps.sh", proc.stdout)

    def test_install_python_deps_allows_explicit_toolchain_upgrade(self) -> None:
        proc = self.run_bash(
            "source scripts/setup_linux.sh; "
            "run_setup_step_with_retry(){ printf 'upgrade=%s cmd=%s\\n' "
            "\"${UPGRADE_TOOLCHAIN:-}\" \"$*\"; }; "
            "UPGRADE_TOOLCHAIN=1; "
            "install_python_deps"
        )

        self.assertIn("upgrade=1 cmd=python_deps env UPGRADE_TOOLCHAIN=1 bash scripts/install_deps.sh", proc.stdout)

    def test_prepare_local_dirs_creates_repo_runtime_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            proc = self.run_bash(
                "source scripts/setup_linux.sh; "
                f"ROOT_DIR='{tmpdir}'; "
                "prepare_local_dirs; "
                "printf 'paths=%s,%s,%s,%s\\n' "
                "\"$(test -d \"$ROOT_DIR/.local/reports\" && echo 1 || echo 0)\" "
                "\"$(test -d \"$ROOT_DIR/data_best\" && echo 1 || echo 0)\" "
                "\"$(test -d \"$ROOT_DIR/artifacts_ens\" && echo 1 || echo 0)\" "
                "\"$(test -d \"$ROOT_DIR/incoming_review_queue\" && echo 1 || echo 0)\""
            )

        self.assertIn("setup_stage=local_dirs status=done", proc.stdout)
        self.assertIn("paths=1,1,1,1", proc.stdout)


if __name__ == "__main__":
    unittest.main()
