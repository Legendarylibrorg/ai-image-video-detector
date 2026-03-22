from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class SetupLocalTests(unittest.TestCase):
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
                "source scripts/setup_local.sh; "
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
                "source scripts/setup_local.sh; "
                f"ENV_FILE='{env_file}'; "
                "HF_TOKEN=''; "
                "HUGGINGFACE_HUB_TOKEN=''; "
                "SETUP_PROMPT_FOR_HF_TOKEN=0; "
                "prompt_for_hf_token_if_missing",
            )

        self.assertIn("setup_stage=env_token status=skip_opt_out", proc.stdout)

    def test_setup_local_defaults_to_nonblocking_token_behavior(self) -> None:
        proc = self.run_bash(
            "source scripts/setup_local.sh; "
            "printf 'prompt=%s attempts=%s sleep=%s\\n' "
            "\"$SETUP_PROMPT_FOR_HF_TOKEN\" \"$SETUP_MAX_ATTEMPTS\" \"$SETUP_RETRY_SLEEP_SEC\""
        )

        self.assertIn("prompt=0 attempts=2 sleep=5", proc.stdout)

    def test_persist_env_hf_token_if_present_uses_hub_token_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text("HF_TOKEN=''\n", encoding="utf-8")
            proc = self.run_bash(
                "source scripts/setup_local.sh; "
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

    def test_print_next_step_prefers_smoke_then_run(self) -> None:
        proc = self.run_bash(
            "source scripts/setup_local.sh; "
            "HF_TOKEN='from_env'; "
            "HUGGINGFACE_HUB_TOKEN='from_env'; "
            "print_next_step"
        )

        self.assertIn("setup_next=run ./local.sh smoke, then ./local.sh run", proc.stdout)

    def test_install_python_deps_skips_toolchain_upgrade_by_default(self) -> None:
        proc = self.run_bash(
            "source scripts/setup_local.sh; "
            "run_setup_step_with_retry(){ printf 'upgrade=%s cmd=%s\\n' "
            "\"${UPGRADE_TOOLCHAIN:-}\" \"$*\"; }; "
            "install_python_deps"
        )

        self.assertIn("upgrade=0 cmd=python_deps bash scripts/install_deps.sh", proc.stdout)

    def test_install_python_deps_allows_explicit_toolchain_upgrade(self) -> None:
        proc = self.run_bash(
            "source scripts/setup_local.sh; "
            "run_setup_step_with_retry(){ printf 'upgrade=%s cmd=%s\\n' "
            "\"${UPGRADE_TOOLCHAIN:-}\" \"$*\"; }; "
            "UPGRADE_TOOLCHAIN=1; "
            "install_python_deps"
        )

        self.assertIn("upgrade=1 cmd=python_deps bash scripts/install_deps.sh", proc.stdout)


if __name__ == "__main__":
    unittest.main()
