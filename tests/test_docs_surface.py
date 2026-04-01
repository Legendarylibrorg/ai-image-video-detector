from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def _assert_phrases(testcase: unittest.TestCase, text: str, phrases: tuple[str, ...], *, label: str = "") -> None:
    for phrase in phrases:
        testcase.assertIn(phrase, text, msg=f"{label}{phrase!r}")


# Shared across STARTUP.md and docs/COMMANDS.md (Compose walkthrough).
_COMPOSE_WALKTHROUGH_PHRASES = (
    "git clone https://github.com/Legendarylibrorg/ai-image-video-detector.git",
    "cd ai-image-video-detector",
    "docker compose build",
    "docker compose run --rm pipeline ./local.sh deps",
    "docker compose run --rm pipeline ./local.sh doctor",
    "printf \"HF_TOKEN='your_token_here'\\n\" >> .env",
    "docker compose run --rm pipeline-gpu ./local.sh run",
    "./local.sh docker-doctor",
)

# Shared native / tooling phrases across STARTUP and COMMANDS.
_COMMON_TOOLING_PHRASES = (
    "isolated container virtualenv at `/opt/aid-venv`",
    "printf \"HF_TOKEN='your_token_here'\\n\" >> .env",
    "./local.sh smoke",
    "./local.sh collect",
    "./local.sh retrain",
    "./local.sh continuous",
    "./local.sh deps",
    "./local.sh doctor",
    "repo CLI commands and the `hf` CLI",
    "pip install -e .",
)


class DocsSurfaceTests(unittest.TestCase):
    def test_docker_compose_surface_exists(self) -> None:
        self.assertTrue((ROOT / "Dockerfile").exists())
        self.assertTrue((ROOT / "Dockerfile.gpu").exists())
        self.assertTrue((ROOT / "docker-compose.yml").exists())
        self.assertTrue((ROOT / ".dockerignore").exists())
        self.assertTrue((ROOT / "scripts" / "docker-entrypoint.sh").exists())
        self.assertFalse((ROOT / "src" / "ai_image_detector" / "explain.py").exists())
        self.assertFalse((ROOT / "src" / "ai_image_detector" / "video.py").exists())

    def test_docker_compose_splits_cpu_and_gpu_images(self) -> None:
        compose_text = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
        cpu_text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        gpu_text = (ROOT / "Dockerfile.gpu").read_text(encoding="utf-8")
        self.assertIn("dockerfile: Dockerfile", compose_text)
        self.assertIn("dockerfile: Dockerfile.gpu", compose_text)
        self.assertIn("VENV_DIR: /opt/aid-venv", compose_text)
        self.assertIn("HF_HUB_CACHE: /workspace/.local/hf/hub", compose_text)
        self.assertIn("HF_DATASETS_CACHE: /workspace/.local/hf/datasets", compose_text)
        self.assertIn("HF_HUB_ENABLE_HF_TRANSFER: \"1\"", compose_text)
        self.assertIn("source: .", compose_text)
        self.assertIn("target: /workspace", compose_text)
        self.assertNotIn("env_file:", compose_text)
        self.assertNotIn("data_best_fast", compose_text)
        self.assertIn("aid_venv:/opt/aid-venv", compose_text)
        self.assertIn("FROM ubuntu:24.04", cpu_text)
        self.assertIn("FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04", gpu_text)

    def test_redundant_root_wrappers_are_removed(self) -> None:
        for name in ["autocollect.sh", "collect.sh", "continuous.sh", "retrain.sh", "run.sh", "start.sh", "train.sh"]:
            self.assertFalse((ROOT / name).exists(), name)

    def test_stale_scripts_are_removed(self) -> None:
        for name in [
            "max_quality_4090.sh",
            "continuous_collect.sh",
            "incremental_refresh.sh",
            "linux_worker.sh",
            "privacy_cleanup.sh",
            "fit_multimodal_fusion.py",
            "build_large_dataset.py",
            "max_accuracy_v2.sh",
            "one_command_4090.sh",
            "local_retrain_4090.sh",
            "weekly_retrain_v3.sh",
        ]:
            self.assertFalse((ROOT / "scripts" / name).exists(), name)

    def test_env_example_does_not_override_collection_tuning_defaults(self) -> None:
        text = (ROOT / ".env.example").read_text(encoding="utf-8")
        self.assertIn("HF_TOKEN=''", text)
        self.assertIn("Collection, cache, and rate-limit tuning defaults now live in the scripts.", text)
        self.assertNotIn("FAST_NO_DEFAULT_SOURCES=", text)
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
        self.assertIn("Day-to-day secure commands after setup", text)
        self.assertLess(text.index("## Secure Linux VM + Docker Compose"), text.index("## Python dependencies"))
        self.assertIn("`./local.sh`", text)
        self.assertIn("`./install.sh`", text)
        self.assertIn("`./scripts/`", text)
        self.assertIn("`./src/ai_image_detector/`", text)
        self.assertIn("`./tests/`", text)
        self.assertIn("local Linux machine", text)
        self.assertIn("isolated container virtualenv at `/opt/aid-venv`", text)
        self.assertIn("native fallback uses a local virtualenv at `./.venv`", text)
        self.assertIn("shell snippets in this README use Linux `bash` command syntax", text)
        self.assertIn("curl -fsSL https://raw.githubusercontent.com/Legendarylibrorg/ai-image-video-detector/main/install.sh | bash", text)
        self.assertIn("For the detailed clone path, ZIP path, and native Linux startup flow, use [docs/STARTUP.md](docs/STARTUP.md).", text)
        self.assertIn("Exact secure startup", text)
        self.assertIn("./local.sh setup", text)
        self.assertIn("./local.sh collect", text)
        self.assertIn("./local.sh retrain", text)
        self.assertIn("./local.sh continuous", text)
        self.assertIn("bash ./install.sh", text)
        self.assertIn("printf \"HF_TOKEN='your_token_here'\\n\" >> .env", text)
        self.assertIn("./local.sh smoke", text)
        self.assertIn("pip install -e .", text)
        self.assertIn("Python dependencies", text)
        self.assertNotIn("pip install -e '.[pipeline]'", text)
        self.assertIn("## Secure Linux VM + Docker Compose", text)
        self.assertIn("docker compose run --rm pipeline-gpu ./local.sh run", text)
        self.assertIn("./local.sh docker-doctor", text)
        self.assertIn("test -f Dockerfile", text)
        self.assertIn("Dockerfile.gpu", text)
        self.assertIn("cap_drop: [ALL]", text)
        self.assertIn("no-new-privileges", text)
        self.assertIn("Docker Compose is not a real VM.", text)
        self.assertIn("Best security with GPU", text)
        self.assertIn("dedicated Linux VM", text)
        self.assertIn("GPU passthrough", text)
        self.assertIn("NVIDIA Container Toolkit", text)
        self.assertIn("container repo root: `/workspace`", text)
        self.assertIn("container virtualenv: `/opt/aid-venv`", text)
        self.assertIn("general source tree under `/workspace`: writable", text)
        self.assertNotIn("data_best_fast", text)
        self.assertIn("for the full step-by-step walkthrough, use [docs/STARTUP.md](docs/STARTUP.md)", text)
        self.assertLess(text.index("## Secure Linux VM + Docker Compose"), text.index("## Quick Start"))
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
        self.assertIn("bash ./install.sh", text)
        self.assertIn("Run `bash ./install.sh` only from inside the repo root", text)
        self.assertIn("Do not use `sudo` for repo commands", text)
        self.assertIn("native Linux fallback", text)
        self.assertIn("shell snippets in this document use Linux `bash` command syntax", text)
        self.assertIn("It does not stop to prompt for `HF_TOKEN` by default.", text)
        self.assertIn("aid-*` commands are thin wrappers", text)
        self.assertNotIn("pip install -e '.[pipeline]'", text)
        self.assertIn("## Dedicated Linux VM + Docker Compose startup", text)
        self.assertIn("Docker Compose does not create a real VM inside Docker.", text)
        self.assertIn("Linux VM setup checklist:", text)
        self.assertIn("Detailed secure Docker config flow:", text)
        self.assertIn("Dockerfile.gpu", text)
        self.assertIn("the VM is the main isolation boundary", text)
        self.assertIn("isolated venv volume at `/opt/aid-venv`", text)
        self.assertIn("does not guarantee safety from malicious packages", text)
        self.assertIn("Best security with GPU", text)
        self.assertIn("GPU passthrough", text)
        self.assertIn("source checkout at `/workspace` so normal editing, setup, and patching still work", text)
        self.assertNotIn("data_best_fast", text)
        self.assertLess(text.index("## Dedicated Linux VM + Docker Compose startup"), text.index("## Native Linux fallback"))
        self.assertIn("## macOS startup", text)
        self.assertIn("## Windows startup", text)
        self.assertIn("WSL2 Ubuntu", text)
        self.assertIn("do not copy the Linux-native `apt-get` commands below", text)
        self.assertNotIn("## Manual Linux bootstrap", text)
        self.assertNotIn("cd /path/to/image-spam", text)
        self.assertNotIn("## Setup options", text)
        self.assertNotIn(
            "sudo apt-get install -y curl ca-certificates git python3 python3-venv python3-pip build-essential clamav clamav-daemon",
            text,
        )
        self.assertIn("unzip ai-image-video-detector-main.zip", text)
        self.assertIn("cd ai-image-video-detector-main", text)
        _assert_phrases(self, text, _COMPOSE_WALKTHROUGH_PHRASES, label="startup: ")
        _assert_phrases(self, text, _COMMON_TOOLING_PHRASES, label="startup: ")

    def test_commands_doc_starts_with_linux_quick_start(self) -> None:
        text = (ROOT / "docs" / "COMMANDS.md").read_text(encoding="utf-8")
        self.assertIn("Snippets use Linux `bash` syntax", text)
        self.assertIn("Public command-to-path map:", text)
        self.assertIn("huggingface_hub", text)
        self.assertIn("Python dependencies", text)
        self.assertNotIn("pip install -e '.[pipeline]'", text)
        self.assertIn("## Dedicated Linux VM + Docker Compose commands", text)
        self.assertIn("NVIDIA Container Toolkit for `pipeline-gpu`", text)
        self.assertIn("/opt/aid-venv", text)
        self.assertIn("/workspace/.local/hf", text)
        self.assertIn("For the full secure startup walkthrough, use [STARTUP.md](STARTUP.md).", text)
        self.assertIn("macOS or Windows", text)
        self.assertIn("./local.sh run", text)
        self.assertIn("./local.sh collect-status", text)
        self.assertIn("./local.sh train", text)
        self.assertIn("./local.sh finetune", text)
        self.assertNotIn("## Raw `scripts/do.sh` commands", text)
        self.assertNotIn(
            "sudo apt-get install -y curl ca-certificates git python3 python3-venv python3-pip build-essential clamav clamav-daemon",
            text,
        )
        self.assertIn("## Sudo guidance", text)
        self.assertIn("Do not add `sudo` to the repo commands", text)
        _assert_phrases(self, text, _COMPOSE_WALKTHROUGH_PHRASES, label="commands: ")
        _assert_phrases(self, text, _COMMON_TOOLING_PHRASES, label="commands: ")


if __name__ == "__main__":
    unittest.main()
