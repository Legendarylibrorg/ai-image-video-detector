from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys
import unittest

from _support import ROOT


def _locked_package_names() -> set[str]:
    names: set[str] = set()
    for line in (ROOT / "requirements.lock").read_text(encoding="utf-8").splitlines():
        if "==" not in line:
            continue
        name, _, _ = line.partition("==")
        if name:
            names.add(name)
    return names


def _toml_section(text: str, section_name: str) -> str:
    pattern = re.compile(rf"(?ms)^\[{re.escape(section_name)}]\n(.*?)(?=^\[|\Z)")
    match = pattern.search(text)
    if match is None:
        raise AssertionError(f"missing [{section_name}] section")
    return match.group(1)


def _toml_array_items(section_text: str, key: str) -> list[str]:
    pattern = re.compile(rf'(?ms)^{re.escape(key)}\s*=\s*\[(.*?)^\]')
    match = pattern.search(section_text)
    if match is None:
        raise AssertionError(f"missing array for key {key}")
    return re.findall(r'"([^"]+)"', match.group(1))


class DependencyMetadataSurfaceTests(unittest.TestCase):
    def test_update_deps_lock_script_is_bash_valid(self) -> None:
        subprocess.run(["bash", "-n", "scripts/update_deps_lock.sh"], cwd=ROOT, check=True)
        subprocess.run([sys.executable, "-m", "py_compile", "scripts/update_deps_lock.py"], cwd=ROOT, check=True)

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
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        project_cfg = _toml_section(pyproject, "project")
        optional = _toml_section(pyproject, "project.optional-dependencies")
        urls = _toml_section(pyproject, "project.urls")

        self.assertIn("dependencies = []", project_cfg)
        self.assertIn('license = {file = "LICENSE"}', project_cfg)
        self.assertIn("authors = [", project_cfg)
        self.assertIn("keywords = [", project_cfg)
        self.assertIn("classifiers = [", project_cfg)
        self.assertNotRegex(project_cfg, r"(?m)^scripts\s*=")
        self.assertNotIn("[project.scripts]", pyproject)
        self.assertEqual(
            sorted(re.findall(r"(?m)^([a-z][a-z-]*)\s*=\s*\[", optional)),
            ["collection", "inference", "pipeline", "training", "video"],
        )
        self.assertIn('Homepage = "https://github.com/Legendarylibrorg/ai-image-video-detector"', urls)
        self.assertIn('Repository = "https://github.com/Legendarylibrorg/ai-image-video-detector"', urls)
        self.assertIn('Issues = "https://github.com/Legendarylibrorg/ai-image-video-detector/issues"', urls)
        self.assertIn("torch>=2.2", _toml_array_items(optional, "inference"))
        self.assertIn("datasets>=2.19", _toml_array_items(optional, "collection"))
        self.assertIn("opencv-python-headless>=4.10", _toml_array_items(optional, "video"))
        self.assertIn("safetensors>=0.4", _toml_array_items(optional, "pipeline"))

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

    def test_requirements_manifest_exists_and_tracks_lock_packages(self) -> None:
        manifest = json.loads((ROOT / "requirements.lock.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["generated_from"], "scripts/update_deps_lock.py")
        lock_names = [line.partition("==")[0] for line in (ROOT / "requirements.lock").read_text(encoding="utf-8").splitlines() if "==" in line]
        manifest_names = [entry["name"] for entry in manifest["packages"]]
        self.assertEqual(lock_names, manifest_names)


if __name__ == "__main__":
    unittest.main()
