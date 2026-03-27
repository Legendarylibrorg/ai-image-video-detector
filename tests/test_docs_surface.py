from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DocsSurfaceTests(unittest.TestCase):
    def test_redundant_root_wrappers_are_removed(self) -> None:
        for name in ["autocollect.sh", "collect.sh", "continuous.sh", "retrain.sh", "run.sh", "start.sh", "train.sh"]:
            self.assertFalse((ROOT / name).exists(), name)

    def test_stale_scripts_are_removed(self) -> None:
        for name in ["continuous_collect.sh", "incremental_refresh.sh", "linux_worker.sh", "privacy_cleanup.sh"]:
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
        self.assertIn("bash ./install.sh", text)
        self.assertIn("printf \"HF_TOKEN='your_token_here'\\n\" >> .env", text)
        self.assertIn("./local.sh smoke", text)
        self.assertIn("python3 -m venv .venv", text)
        self.assertIn("source .venv/bin/activate", text)
        self.assertIn("./local.sh deps", text)
        self.assertIn("./local.sh doctor", text)
        self.assertIn("repo CLI commands are installed", text)
        self.assertIn("./local.sh run", text)
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
        self.assertIn("./local.sh deps", text)
        self.assertIn("./local.sh doctor", text)
        self.assertIn("repo CLI commands and the `hf` CLI", text)
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
        self.assertIn("huggingface_hub", text)
        self.assertIn("curl -fsSL https://raw.githubusercontent.com/Legendarylibrorg/ai-image-video-detector/main/install.sh | bash", text)
        self.assertIn("git clone https://github.com/Legendarylibrorg/ai-image-video-detector.git", text)
        self.assertIn("cd ai-image-video-detector", text)
        self.assertIn("unzip ai-image-video-detector-main.zip", text)
        self.assertIn("cd ai-image-video-detector-main", text)
        self.assertIn("sudo apt-get update", text)
        self.assertIn("curl ca-certificates git unzip python3", text)
        self.assertIn("./local.sh setup", text)
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
        self.assertIn("repo CLI commands and the `hf` CLI", text)
        self.assertIn("./local.sh run", text)
        self.assertNotIn("## Raw `scripts/do.sh` commands", text)
        self.assertNotIn(
            "sudo apt-get install -y curl ca-certificates git python3 python3-venv python3-pip build-essential clamav clamav-daemon",
            text,
        )


if __name__ == "__main__":
    unittest.main()
