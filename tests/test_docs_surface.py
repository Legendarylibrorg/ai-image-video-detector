from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DocsSurfaceTests(unittest.TestCase):
    def test_docker_compose_surface_exists(self) -> None:
        self.assertTrue((ROOT / "Dockerfile").exists())
        self.assertTrue((ROOT / "docker-compose.yml").exists())
        self.assertTrue((ROOT / ".dockerignore").exists())
        self.assertTrue((ROOT / "scripts" / "docker-entrypoint.sh").exists())
        self.assertFalse((ROOT / "src" / "ai_image_detector" / "explain.py").exists())
        self.assertFalse((ROOT / "src" / "ai_image_detector" / "video.py").exists())

    def test_redundant_root_wrappers_are_removed(self) -> None:
        for name in ["autocollect.sh", "collect.sh", "continuous.sh", "retrain.sh", "run.sh", "start.sh", "train.sh"]:
            self.assertFalse((ROOT / name).exists(), name)

    def test_stale_scripts_are_removed(self) -> None:
        for name in [
            "continuous_collect.sh",
            "incremental_refresh.sh",
            "linux_worker.sh",
            "privacy_cleanup.sh",
            "fit_multimodal_fusion.py",
            "build_large_dataset.py",
            "max_accuracy_v2.sh",
            "one_command_4090.sh",
        ]:
            self.assertFalse((ROOT / "scripts" / name).exists(), name)

    def test_env_example_does_not_override_collection_tuning_defaults(self) -> None:
        text = (ROOT / ".env.example").read_text(encoding="utf-8")
        self.assertIn("HF_TOKEN=''", text)
        self.assertIn("Collection, cache, and rate-limit tuning defaults now live in the scripts.", text)
        self.assertNotIn("DIVERSE_HF_MIN_QUALITY_SCORE=", text)
        self.assertNotIn("DIVERSE_REPO_BASE_PAUSE_MS=", text)
        self.assertNotIn("VIDEO_SNAPSHOT_MAX_WORKERS=", text)

    def test_readme_points_to_split_startup_docs(self) -> None:
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("[docs/STARTUP.md](docs/STARTUP.md)", text)
        self.assertIn("[docs/COMMANDS.md](docs/COMMANDS.md)", text)
        self.assertIn("[docs/REFERENCE.md](docs/REFERENCE.md)", text)
        self.assertIn("## Repo Layout", text)
        self.assertIn("## Command Map", text)
        self.assertIn("Most people only need these commands", text)
        self.assertIn("`./local.sh`", text)
        self.assertIn("`./install.sh`", text)
        self.assertIn("`./scripts/`", text)
        self.assertIn("`./src/ai_image_detector/`", text)
        self.assertIn("`./tests/`", text)
        self.assertIn("local Linux machine", text)
        self.assertIn("local virtualenv at `./.venv`", text)
        self.assertIn("curl -fsSL https://raw.githubusercontent.com/Legendarylibrorg/ai-image-video-detector/main/install.sh | bash", text)
        self.assertIn("git clone https://github.com/Legendarylibrorg/ai-image-video-detector.git", text)
        self.assertIn("cd ai-image-video-detector", text)
        self.assertIn("unzip ai-image-video-detector-main.zip", text)
        self.assertIn("cd ai-image-video-detector-main", text)
        self.assertIn("huggingface_hub", text)
        self.assertIn("sudo apt-get update", text)
        self.assertIn("curl ca-certificates git unzip python3", text)
        self.assertIn("./local.sh setup", text)
        self.assertIn("./local.sh collect", text)
        self.assertIn("./local.sh retrain", text)
        self.assertIn("./local.sh continuous", text)
        self.assertIn("bash ./install.sh", text)
        self.assertIn("printf \"HF_TOKEN='your_token_here'\\n\" >> .env", text)
        self.assertIn("./local.sh smoke", text)
        self.assertIn("python3 -m venv .venv", text)
        self.assertIn("source .venv/bin/activate", text)
        self.assertIn("./local.sh deps", text)
        self.assertIn("./local.sh doctor", text)
        self.assertIn("repo CLI commands are installed", text)
        self.assertIn("pip install -e '.[pipeline]'", text)
        self.assertIn("pip install -e .", text)
        self.assertIn("Dependency Profiles", text)
        self.assertIn("## Docker Compose", text)
        self.assertIn("docker compose run --rm pipeline ./local.sh doctor", text)
        self.assertIn("docker compose run --rm pipeline-gpu ./local.sh run", text)
        self.assertIn("cap_drop: [ALL]", text)
        self.assertIn("no-new-privileges", text)
        self.assertIn("./local.sh run", text)
        self.assertIn("./local.sh finetune", text)
        self.assertNotIn("cd /path/to/image-spam", text)
        self.assertNotIn("Everything else is advanced/internal", text)
        self.assertNotIn("## Advanced Reference", text)

    def test_startup_doc_marks_sudo_only_for_system_commands(self) -> None:
        text = (ROOT / "docs" / "STARTUP.md").read_text(encoding="utf-8")
        self.assertIn("Clone path:", text)
        self.assertIn("ZIP path:", text)
        self.assertIn("Already inside the repo root:", text)
        self.assertIn("sudo apt-get update", text)
        self.assertIn(
            "sudo apt-get install -y curl ca-certificates git unzip python3 python3-venv python3-pip build-essential clamav clamav-daemon",
            text,
        )
        self.assertIn("curl -fsSL https://raw.githubusercontent.com/Legendarylibrorg/ai-image-video-detector/main/install.sh | bash", text)
        self.assertIn("git clone https://github.com/Legendarylibrorg/ai-image-video-detector.git", text)
        self.assertIn("cd ai-image-video-detector", text)
        self.assertIn("unzip ai-image-video-detector-main.zip", text)
        self.assertIn("cd ai-image-video-detector-main", text)
        self.assertIn("bash ./install.sh", text)
        self.assertIn("Run `bash ./install.sh` only from inside the repo root", text)
        self.assertIn("Do not use `sudo` for repo commands", text)
        self.assertIn("pinned local virtualenv at `./.venv`", text)
        self.assertIn("python3 -m venv .venv", text)
        self.assertIn("source .venv/bin/activate", text)
        self.assertIn("It does not stop to prompt for `HF_TOKEN` by default.", text)
        self.assertIn("printf \"HF_TOKEN='your_token_here'\\n\" >> .env", text)
        self.assertIn("./local.sh smoke", text)
        self.assertIn("./local.sh collect", text)
        self.assertIn("./local.sh retrain", text)
        self.assertIn("./local.sh continuous", text)
        self.assertIn("./local.sh deps", text)
        self.assertIn("./local.sh doctor", text)
        self.assertIn("repo CLI commands and the `hf` CLI", text)
        self.assertIn("pip install -e '.[pipeline]'", text)
        self.assertIn("aid-*` commands are thin wrappers", text)
        self.assertIn("## Docker Compose startup", text)
        self.assertIn("docker compose run --rm pipeline-gpu ./local.sh run", text)
        self.assertIn("read-only container root filesystem", text)
        self.assertIn("VM path is intentionally not added", text)
        self.assertIn("## macOS startup", text)
        self.assertIn("## Windows startup", text)
        self.assertIn("WSL2 Ubuntu", text)
        self.assertIn("do not copy the `apt-get` commands below", text)
        self.assertNotIn("## Manual Linux bootstrap", text)
        self.assertNotIn("cd /path/to/image-spam", text)
        self.assertNotIn("## Setup options", text)
        self.assertNotIn(
            "sudo apt-get install -y curl ca-certificates git python3 python3-venv python3-pip build-essential clamav clamav-daemon",
            text,
        )

    def test_commands_doc_starts_with_linux_quick_start(self) -> None:
        text = (ROOT / "docs" / "COMMANDS.md").read_text(encoding="utf-8")
        self.assertIn("Clone path:", text)
        self.assertIn("ZIP path:", text)
        self.assertIn("Already inside the repo root:", text)
        self.assertIn("repo-local Python environment is `./.venv`", text)
        self.assertIn("Public command-to-path map:", text)
        self.assertIn("huggingface_hub", text)
        self.assertIn("curl -fsSL https://raw.githubusercontent.com/Legendarylibrorg/ai-image-video-detector/main/install.sh | bash", text)
        self.assertIn("git clone https://github.com/Legendarylibrorg/ai-image-video-detector.git", text)
        self.assertIn("cd ai-image-video-detector", text)
        self.assertIn("unzip ai-image-video-detector-main.zip", text)
        self.assertIn("cd ai-image-video-detector-main", text)
        self.assertIn("sudo apt-get update", text)
        self.assertIn("curl ca-certificates git unzip python3", text)
        self.assertIn("./local.sh setup", text)
        self.assertIn("./local.sh collect", text)
        self.assertIn("./local.sh retrain", text)
        self.assertIn("./local.sh continuous", text)
        self.assertIn("bash ./install.sh", text)
        self.assertIn("Run `bash ./install.sh` only from inside the repo root", text)
        self.assertIn("printf \"HF_TOKEN='your_token_here'\\n\" >> .env", text)
        self.assertIn("./local.sh smoke", text)
        self.assertIn("python3 -m venv .venv", text)
        self.assertIn("source .venv/bin/activate", text)
        self.assertIn("./local.sh deps", text)
        self.assertIn("./local.sh doctor", text)
        self.assertIn("./local.sh collect-status", text)
        self.assertIn("./local.sh train", text)
        self.assertIn("./local.sh finetune", text)
        self.assertIn("repo CLI commands and the `hf` CLI", text)
        self.assertIn("pip install -e '.[pipeline]'", text)
        self.assertIn("base install lightweight", text)
        self.assertIn("## Docker Compose commands", text)
        self.assertIn("docker compose run --rm pipeline ./local.sh doctor", text)
        self.assertIn("drop all Linux capabilities", text)
        self.assertIn("does not add a VM layer", text)
        self.assertIn("Linux commands", text)
        self.assertIn("macOS or Windows", text)
        self.assertIn("./local.sh run", text)
        self.assertNotIn("## Raw `scripts/do.sh` commands", text)
        self.assertNotIn(
            "sudo apt-get install -y curl ca-certificates git python3 python3-venv python3-pip build-essential clamav clamav-daemon",
            text,
        )


if __name__ == "__main__":
    unittest.main()
