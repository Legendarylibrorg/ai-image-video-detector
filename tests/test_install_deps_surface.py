from __future__ import annotations

import os
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]


class InstallDepsSurfaceTests(unittest.TestCase):
    def test_install_deps_verifies_huggingface_python_and_cli(self) -> None:
        text = (ROOT / "scripts" / "install_deps.sh").read_text(encoding="utf-8")
        self.assertIn("import ai_image_detector", text)
        self.assertIn("import numpy", text)
        self.assertIn("import piexif", text)
        self.assertIn("import safetensors", text)
        self.assertIn("import sklearn", text)
        self.assertIn("import torchvision", text)
        self.assertIn("import huggingface_hub", text)
        self.assertIn("import cv2", text)
        self.assertIn("command -v hf", text)
        self.assertIn("command -v aid-train", text)
        self.assertIn("command -v aid-video-train", text)
        self.assertNotIn("command -v aid-dataset", text)
        self.assertIn("deps_fail=huggingface_cli_missing", text)
        self.assertIn("deps_fail=repo_cli_missing", text)
        self.assertIn("PIP_DISABLE_PIP_VERSION_CHECK=1", text)
        self.assertIn("python -m pip install --progress-bar off", text)
        self.assertIn('"$UPGRADE_TOOLCHAIN" != "1"', text)

    def test_update_deps_lock_emits_direct_dependency_lock(self) -> None:
        text = (ROOT / "scripts" / "update_deps_lock.sh").read_text(encoding="utf-8")
        self.assertIn('LOCK_EXTRA="${LOCK_EXTRA:-pipeline}"', text)
        self.assertIn('pip install --upgrade --upgrade-strategy eager -e ".[${LOCK_EXTRA}]"', text)
        self.assertIn("python -m pip install tomli", text)
        self.assertIn("import tomllib", text)
        self.assertIn('python - <<\'PY\' "$ROOT_DIR/pyproject.toml" "$LOCK_EXTRA"', text)
        self.assertIn("importlib.metadata", text)
        self.assertIn("tomllib", text)
        self.assertIn("tomli as tomllib", text)
        self.assertIn('project_cfg.get("optional-dependencies", {}).get(extra_name, [])', text)
        self.assertNotIn("pip freeze", text)

    def test_requirements_lock_stays_minimal_and_excludes_web_stack(self) -> None:
        text = (ROOT / "requirements.lock").read_text(encoding="utf-8")
        for name in [
            "datasets==",
            "huggingface_hub==",
            "numpy==",
            "opencv-python-headless==",
            "piexif==",
            "pillow==",
            "safetensors==",
            "scikit-learn==",
            "torch==",
            "torchvision==",
        ]:
            self.assertIn(name, text)
        for name in [
            "fastapi==",
            "uvicorn==",
            "starlette==",
            "pydantic==",
            "aiohttp==",
            "anyio==",
            "annotated-doc==",
        ]:
            self.assertNotIn(name, text)

    def test_pyproject_keeps_base_install_light_and_moves_runtime_stack_to_pipeline_extra(self) -> None:
        text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn("dependencies = []", text)
        self.assertIn("[project.optional-dependencies]", text)
        self.assertIn("inference = [", text)
        self.assertIn("training = [", text)
        self.assertIn("collection = [", text)
        self.assertIn("video = [", text)
        self.assertIn("pipeline = [", text)
        self.assertIn('"torch>=2.2"', text)
        self.assertIn('"datasets>=2.19"', text)
        self.assertIn('"opencv-python-headless>=4.10"', text)
        self.assertNotIn("fastapi", text.lower())
        self.assertNotIn("uvicorn", text.lower())

    def test_install_deps_fast_path_skips_work_when_current_by_default(self) -> None:
        subprocess.run(
            ["bash", "scripts/install_deps.sh"],
            cwd=ROOT,
            env={**os.environ, "UPGRADE_TOOLCHAIN": "0"},
            check=True,
            capture_output=True,
            text=True,
        )

        proc = subprocess.run(
            ["bash", "scripts/install_deps.sh"],
            cwd=ROOT,
            env={**os.environ, "UPGRADE_TOOLCHAIN": "0"},
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("deps_status=up_to_date", proc.stdout)
        self.assertNotIn("warning_toolchain_upgrade_failed", proc.stdout)


if __name__ == "__main__":
    unittest.main()
