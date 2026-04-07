from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DocsSurfaceTests(unittest.TestCase):
    def test_docker_compose_surface_exists(self) -> None:
        compose_text = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
        entrypoint_text = (ROOT / "scripts" / "docker-entrypoint.sh").read_text(encoding="utf-8")

        self.assertTrue((ROOT / "Dockerfile").exists())
        self.assertTrue((ROOT / "Dockerfile.gpu").exists())
        self.assertTrue((ROOT / ".dockerignore").exists())
        self.assertIn("dockerfile: Dockerfile", compose_text)
        self.assertIn("dockerfile: Dockerfile.gpu", compose_text)
        self.assertIn("aid_venv:/opt/aid-venv", compose_text)
        self.assertIn('export VENV_DIR="${VENV_DIR:-/opt/aid-venv}"', entrypoint_text)

    def test_removed_legacy_wrappers_and_stale_scripts_stay_gone(self) -> None:
        for name in ("autocollect.sh", "collect.sh", "continuous.sh", "retrain.sh", "run.sh", "start.sh", "train.sh"):
            self.assertFalse((ROOT / name).exists(), name)
        for name in ("max_quality_4090.sh", "continuous_collect.sh", "incremental_refresh.sh", "weekly_retrain_v3.sh"):
            self.assertFalse((ROOT / "scripts" / name).exists(), name)

    def test_env_example_keeps_token_placeholder_without_old_tuning_noise(self) -> None:
        text = (ROOT / ".env.example").read_text(encoding="utf-8")
        self.assertIn("HF_TOKEN=''", text)
        self.assertIn("Collection, cache, and rate-limit tuning defaults now live in the scripts.", text)
        self.assertNotIn("FAST_NO_DEFAULT_SOURCES=", text)
        self.assertNotIn("DIVERSE_HF_MIN_QUALITY_SCORE=", text)

    def test_readme_maps_users_to_docs_and_core_commands(self) -> None:
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("## Documentation map", text)
        self.assertIn("[docs/STARTUP.md](docs/STARTUP.md)", text)
        self.assertIn("[docs/COMMANDS.md](docs/COMMANDS.md)", text)
        self.assertIn("[docs/REFERENCE.md](docs/REFERENCE.md)", text)
        self.assertIn("./local.sh setup", text)
        self.assertIn("./local.sh collect", text)
        self.assertIn("./local.sh run", text)
        self.assertIn("./local.sh retrain", text)
        self.assertIn("./local.sh continuous", text)
        self.assertIn("./local.sh smoke", text)

    def test_startup_and_commands_docs_cover_supported_platforms_and_entrypoints(self) -> None:
        startup = (ROOT / "docs" / "STARTUP.md").read_text(encoding="utf-8")
        commands = (ROOT / "docs" / "COMMANDS.md").read_text(encoding="utf-8")

        self.assertIn("## Dedicated Linux VM + Docker Compose startup", startup)
        self.assertIn("## Native Linux fallback", startup)
        self.assertIn("## macOS startup", startup)
        self.assertIn("## Windows startup", startup)
        self.assertIn("Do not use `sudo` for repo commands", startup)
        self.assertIn("## Dedicated Linux VM + Docker Compose commands", commands)
        self.assertIn("## macOS and Windows (short)", commands)
        self.assertIn("./local.sh collect-status", commands)
        self.assertIn("./local.sh train", commands)
        self.assertIn("./local.sh finetune", commands)
        self.assertIn("./local.sh docker-doctor", commands)


if __name__ == "__main__":
    unittest.main()
