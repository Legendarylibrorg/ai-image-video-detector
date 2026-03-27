from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]


class LocalShTests(unittest.TestCase):
    def test_help_shows_simple_workflow(self) -> None:
        proc = subprocess.run(
            ["bash", "./local.sh", "help"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        out = proc.stdout
        self.assertIn("usage: ./local.sh [setup|deps|doctor|docker-doctor|collect|run|status|smoke|smoke-real|collect-status|train|retrain|finetune|continuous]", out)
        self.assertIn("linux bash commands", out.lower())
        self.assertIn("native linux", out.lower())
        self.assertIn("sudo apt-get update", out)
        self.assertIn("git unzip python3", out)
        self.assertIn("./local.sh setup", out)
        self.assertIn("printf \"HF_TOKEN='your_token_here'\\n\" >> .env", out)
        self.assertIn("./local.sh smoke", out)
        self.assertIn("./local.sh deps", out)
        self.assertIn("./local.sh doctor", out)
        self.assertIn("./local.sh docker-doctor", out)
        self.assertIn("./local.sh smoke-real", out)
        self.assertIn("./local.sh collect", out)
        self.assertIn("./local.sh run", out)
        self.assertIn("./local.sh status", out)
        self.assertIn("./local.sh collect-status", out)
        self.assertIn("./local.sh train", out)
        self.assertIn("./local.sh retrain", out)
        self.assertIn("./local.sh finetune", out)
        self.assertIn("./local.sh continuous", out)
        self.assertIn("/opt/aid-venv", out)
        self.assertIn("source checkout is mounted read-only", out)
        self.assertIn("setup creates or reuses ./.venv", out)
        self.assertNotIn("advanced aliases still work", out.lower())
        self.assertNotIn("detect <image>", out)
        self.assertNotIn("./local.sh deps-update", out)
        self.assertNotIn("./local.sh venv", out)

    def test_setup_uses_linux_setup_path_in_dry_run(self) -> None:
        proc = subprocess.run(
            ["bash", "./local.sh", "setup"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "DRY_RUN": "1",
                "SETUP_INSTALL_SYSTEM_DEPS": "0",
                "SETUP_PROMPT_FOR_HF_TOKEN": "0",
                "HF_SETUP_REQUIRE_TOKEN": "0",
            },
        )

        out = proc.stdout
        self.assertIn("setup_stage=python_deps status=run", out)
        self.assertIn("[DRY_RUN] env UPGRADE_TOOLCHAIN=0 bash scripts/install_deps.sh", out)
        self.assertIn("[DRY_RUN] env DOCTOR_REQUIRE_TOKEN=0 bash scripts/doctor.sh", out)
        self.assertIn("setup_status=ready", out)

    def test_deps_command_runs_install_script(self) -> None:
        proc = subprocess.run(
            ["bash", "./local.sh", "deps"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("deps_status=up_to_date", proc.stdout)

    def test_collect_status_stdout_is_valid_json(self) -> None:
        proc = subprocess.run(
            ["bash", "./local.sh", "collect-status"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        payload = json.loads(proc.stdout)
        self.assertIn("data_root", payload)
        self.assertTrue(proc.stdout.lstrip().startswith("{"))

    def test_collect_command_routes_to_collection_pipeline(self) -> None:
        text = (ROOT / "local.sh").read_text(encoding="utf-8")
        self.assertIn("collect)", text)
        self.assertIn('run_do collect', text)

    def test_retrain_and_continuous_commands_route_to_supported_pipeline_paths(self) -> None:
        text = (ROOT / "local.sh").read_text(encoding="utf-8")
        self.assertIn("docker-doctor)", text)
        self.assertIn("retrain)", text)
        self.assertIn("finetune)", text)
        self.assertIn("continuous)", text)
        self.assertIn('DOCTOR_REQUIRE_DOCKER=1 bash scripts/doctor.sh "$@"', text)
        self.assertIn('run_do retrain "$@"', text)
        self.assertIn('run_do finetune "$@"', text)
        self.assertIn('run_do continuous "$@"', text)


if __name__ == "__main__":
    unittest.main()
