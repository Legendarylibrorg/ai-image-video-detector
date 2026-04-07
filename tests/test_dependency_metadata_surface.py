from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import unittest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


def _locked_package_names() -> set[str]:
    names: set[str] = set()
    for line in (ROOT / "requirements.lock").read_text(encoding="utf-8").splitlines():
        if "==" not in line:
            continue
        name, _, _ = line.partition("==")
        if name:
            names.add(name)
    return names


class DependencyMetadataSurfaceTests(unittest.TestCase):
    def test_update_deps_lock_script_is_bash_valid(self) -> None:
        subprocess.run(["bash", "-n", "scripts/update_deps_lock.sh"], cwd=ROOT, check=True)

    def test_requirements_lock_tracks_expected_runtime_packages(self) -> None:
        locked = _locked_package_names()
        self.assertTrue(
            {
                "datasets",
                "huggingface_hub",
                "numpy",
                "opencv-python-headless",
                "piexif",
                "pillow",
                "safetensors",
                "scikit-learn",
                "torch",
                "torchvision",
            }.issubset(locked)
        )
        self.assertTrue(
            {
                "fastapi",
                "uvicorn",
                "starlette",
                "pydantic",
                "aiohttp",
                "anyio",
                "annotated-doc",
            }.isdisjoint(locked)
        )

    def test_pyproject_uses_profiled_optional_dependencies(self) -> None:
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        project_cfg = project["project"]
        optional = project_cfg["optional-dependencies"]

        self.assertEqual(project_cfg.get("dependencies"), [])
        self.assertIsNone(project_cfg.get("scripts"))
        self.assertEqual(sorted(optional), ["collection", "inference", "pipeline", "training", "video"])
        self.assertIn("torch>=2.2", optional["inference"])
        self.assertIn("datasets>=2.19", optional["collection"])
        self.assertIn("opencv-python-headless>=4.10", optional["video"])
        self.assertIn("safetensors>=0.4", optional["pipeline"])

    def test_deps_profile_script_emits_profile_specific_modules(self) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                "scripts/deps_profile.py",
                "--extras",
                "training,video",
                "--emit",
                "json",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["expanded_extras"], ["training", "video"])
        self.assertEqual(
            payload["python_modules"],
            ["ai_image_detector", "numpy", "PIL", "safetensors", "torch", "torchvision", "piexif", "sklearn", "cv2"],
        )


if __name__ == "__main__":
    unittest.main()
