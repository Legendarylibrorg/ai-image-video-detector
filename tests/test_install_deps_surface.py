from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class InstallDepsSurfaceTests(unittest.TestCase):
    def test_install_deps_scopes_dependency_checks_by_extra(self) -> None:
        text = (ROOT / "scripts" / "install_deps.sh").read_text(encoding="utf-8")
        self.assertIn('DEPS_EXTRA="${DEPS_EXTRA:-pipeline}"', text)
        self.assertIn('DEPS_STAMP_FILE="${DEPS_STAMP_FILE:-$VENV_DIR/.deps_stamp.${DEPS_PROFILE_TAG}}"', text)
        self.assertIn("extra_enabled()", text)
        self.assertIn("verify_python_deps()", text)
        self.assertIn("verify_required_commands()", text)
        self.assertIn('modules = {"ai_image_detector"}', text)
        self.assertIn('modules.update({"numpy", "PIL", "safetensors", "torch", "torchvision"})', text)
        self.assertIn('modules.update({"piexif", "sklearn"})', text)
        self.assertIn('modules.update({"datasets", "huggingface_hub", "PIL"})', text)
        self.assertIn('modules.add("cv2")', text)
        self.assertIn("command -v hf", text)
        self.assertIn("command -v aid-train", text)
        self.assertIn("command -v aid-video-train", text)
        self.assertNotIn("command -v aid-dataset", text)
        self.assertIn("deps_fail=huggingface_cli_missing extra=$DEPS_EXTRA", text)
        self.assertIn("deps_fail=repo_cli_missing cli=aid-train extra=$DEPS_EXTRA", text)
        self.assertIn("PIP_DISABLE_PIP_VERSION_CHECK=1", text)
        self.assertIn("python -m pip install --progress-bar off", text)
        self.assertIn('"$UPGRADE_TOOLCHAIN" != "1"', text)

    def test_install_deps_uses_lock_only_for_full_pipeline_profile(self) -> None:
        text = (ROOT / "scripts" / "install_deps.sh").read_text(encoding="utf-8")
        self.assertIn('if [[ -s "$LOCK_FILE" && "$DEPS_EXTRA" == "pipeline" ]]; then', text)
        self.assertIn("selected_lock_package_names()", text)
        self.assertIn("install_selected_locked_packages()", text)
        self.assertIn('echo "deps_lock=subset extra=$DEPS_EXTRA packages=${names[*]}"', text)
        self.assertIn('echo "deps_lock=missing file=$LOCK_FILE fallback=pyproject_resolve"', text)
        self.assertIn("pip_cmd install -e . --no-deps --no-build-isolation", text)
        self.assertIn("pip_cmd install --upgrade --upgrade-strategy eager -e .", text)

    def test_update_deps_lock_emits_direct_dependency_lock(self) -> None:
        text = (ROOT / "scripts" / "update_deps_lock.sh").read_text(encoding="utf-8")
        self.assertIn("pip install --upgrade --upgrade-strategy eager -e .", text)
        self.assertIn("python -m pip install tomli", text)
        self.assertIn("import tomllib", text)
        self.assertIn('python - <<\'PY\' "$ROOT_DIR/pyproject.toml" > "$TMP_FILE"', text)
        self.assertIn("importlib.metadata", text)
        self.assertIn("tomllib", text)
        self.assertIn("tomli as tomllib", text)
        self.assertIn('deps = list(project_cfg.get("dependencies", []))', text)
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

    def test_pyproject_lists_required_runtime_dependencies(self) -> None:
        text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn("dependencies = [", text)
        self.assertNotIn("[project.optional-dependencies]", text)
        self.assertIn('"torch>=2.2"', text)
        self.assertIn('"datasets>=2.19"', text)
        self.assertIn('"opencv-python-headless>=4.10"', text)
        self.assertIn('"huggingface_hub>=0.24"', text)
        self.assertNotIn("fastapi", text.lower())
        self.assertNotIn("uvicorn", text.lower())

    def test_install_deps_fast_path_skips_work_when_current_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            venv = Path(tmpdir) / "venv"
            bin_dir = venv / "bin"
            bin_dir.mkdir(parents=True, exist_ok=True)

            (bin_dir / "activate").write_text(
                f'export VIRTUAL_ENV="{venv}"\nexport PATH="{bin_dir}:$PATH"\n',
                encoding="utf-8",
            )
            fake_python = bin_dir / "python"
            fake_python.write_text(
                "#!/usr/bin/env bash\n"
                'if [[ "$1" == "-m" && "$2" == "pip" && "$3" == "install" ]]; then\n'
                '  echo "unexpected_install" >&2\n'
                "  exit 91\n"
                "fi\n"
                "exit 0\n",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            for name in ("hf", "aid-train", "aid-video-train"):
                stub = bin_dir / name
                stub.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
                stub.chmod(0o755)

            digest = hashlib.sha256()
            digest.update(b"deps_extra=pipeline\n")
            digest.update((ROOT / "requirements.lock").read_bytes())
            digest.update((ROOT / "pyproject.toml").read_bytes())
            (venv / ".deps_stamp.pipeline").write_text(digest.hexdigest() + "\n", encoding="utf-8")

            proc = subprocess.run(
                ["bash", "scripts/install_deps.sh"],
                cwd=ROOT,
                env={**os.environ, "VENV_DIR": str(venv), "UPGRADE_TOOLCHAIN": "0"},
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("deps_status=up_to_date", proc.stdout)
        self.assertNotIn("warning_toolchain_upgrade_failed", proc.stdout)
        self.assertNotIn("unexpected_install", proc.stderr)


if __name__ == "__main__":
    unittest.main()
