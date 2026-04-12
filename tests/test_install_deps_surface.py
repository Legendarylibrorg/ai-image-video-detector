from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from tests._support import ROOT


class InstallDepsSurfaceTests(unittest.TestCase):
    def test_invalid_deps_extra_token_rejected(self) -> None:
        proc = subprocess.run(
            ["bash", "scripts/install_deps.sh"],
            cwd=ROOT,
            env={**os.environ, "DEPS_EXTRA": "training,evil_extra", "UPGRADE_TOOLCHAIN": "0"},
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(proc.returncode, 0)
        combined = (proc.stdout or "") + (proc.stderr or "")
        self.assertIn("invalid_deps_extra_token", combined)
        self.assertIn("evil_extra", combined)

    def test_pipeline_profile_uses_full_lock_path_and_publishes_both_training_wrappers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            venv = tmp / "venv"
            bin_dir = venv / "bin"
            bin_dir.mkdir(parents=True, exist_ok=True)
            log_path = tmp / "pip.log"
            self._write_fake_activate(bin_dir, venv)
            self._write_fake_python(bin_dir)
            fake_hf = bin_dir / "hf"
            fake_hf.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            fake_hf.chmod(0o755)

            proc = subprocess.run(
                ["bash", "scripts/install_deps.sh"],
                cwd=ROOT,
                env={
                    **os.environ,
                    "FAKE_PYTHON_LOG": str(log_path),
                    "VENV_DIR": str(venv),
                    "DEPS_EXTRA": "pipeline",
                    "UPGRADE_TOOLCHAIN": "0",
                },
                check=True,
                capture_output=True,
                text=True,
            )

            pip_log = log_path.read_text(encoding="utf-8")
            profile_value = (venv / ".deps_profile").read_text(encoding="utf-8").strip()
            train_wrapper = (bin_dir / "aid-train").read_text(encoding="utf-8")
            video_wrapper = (bin_dir / "aid-video-train").read_text(encoding="utf-8")

        self.assertEqual(profile_value, "pipeline")
        self.assertIn("train_main", train_wrapper)
        self.assertIn("video_train_main", video_wrapper)
        self.assertNotIn("deps_lock=subset", proc.stdout)
        self.assertIn("-m pip install --progress-bar off -r ", pip_log)
        self.assertIn("torch==", pip_log)
        self.assertIn("torchvision==", pip_log)
        self.assertIn("-m pip install --progress-bar off -e .[pipeline] --no-deps --no-build-isolation", pip_log)

    def test_training_profile_succeeds_without_collection_cli_and_publishes_only_train_wrapper(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            venv = tmp / "venv"
            bin_dir = venv / "bin"
            bin_dir.mkdir(parents=True, exist_ok=True)
            log_path = tmp / "pip.log"
            self._write_fake_activate(bin_dir, venv)
            self._write_fake_python(bin_dir)

            proc = subprocess.run(
                ["bash", "scripts/install_deps.sh"],
                cwd=ROOT,
                env={
                    **os.environ,
                    "FAKE_PYTHON_LOG": str(log_path),
                    "VENV_DIR": str(venv),
                    "DEPS_EXTRA": "training",
                    "UPGRADE_TOOLCHAIN": "0",
                },
                check=True,
                capture_output=True,
                text=True,
            )

            pip_log = log_path.read_text(encoding="utf-8")
            profile_value = (venv / ".deps_profile").read_text(encoding="utf-8").strip()
            train_wrapper = (bin_dir / "aid-train").read_text(encoding="utf-8")
            video_exists = (bin_dir / "aid-video-train").exists()

        self.assertEqual(profile_value, "training")
        self.assertIn("train_main", train_wrapper)
        self.assertFalse(video_exists)
        self.assertIn("deps_lock=subset extra=training", proc.stdout)
        self.assertIn("-m pip install --progress-bar off -e .[training] --no-deps --no-build-isolation", pip_log)
        self.assertNotIn("huggingface_cli_missing", proc.stderr)

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

    def test_fast_path_requires_collection_cli_inside_repo_venv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            venv = tmp / "venv"
            bin_dir = venv / "bin"
            tools_dir = tmp / "tools"
            bin_dir.mkdir(parents=True, exist_ok=True)
            tools_dir.mkdir(parents=True, exist_ok=True)

            self._write_fake_activate(bin_dir, venv)
            self._write_fake_python(bin_dir)
            global_hf = tools_dir / "hf"
            global_hf.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            global_hf.chmod(0o755)

            digest = hashlib.sha256()
            digest.update(b"deps_extra=collection\n")
            digest.update((ROOT / "requirements.lock").read_bytes())
            digest.update((ROOT / "pyproject.toml").read_bytes())
            (venv / ".deps_stamp.collection").write_text(digest.hexdigest() + "\n", encoding="utf-8")

            proc = subprocess.run(
                ["bash", "scripts/install_deps.sh"],
                cwd=ROOT,
                env={
                    **os.environ,
                    "PATH": f"{tools_dir}:{os.environ['PATH']}",
                    "VENV_DIR": str(venv),
                    "DEPS_EXTRA": "collection",
                    "UPGRADE_TOOLCHAIN": "0",
                },
                capture_output=True,
                text=True,
            )

        self.assertEqual(proc.returncode, 1)
        self.assertIn("deps_fail=huggingface_cli_missing", proc.stderr)

    def test_missing_lock_file_falls_back_to_profiled_pyproject_install(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            venv = tmp / "venv"
            bin_dir = venv / "bin"
            bin_dir.mkdir(parents=True, exist_ok=True)
            log_path = tmp / "pip.log"
            self._write_fake_activate(bin_dir, venv)
            self._write_fake_python(bin_dir)

            proc = subprocess.run(
                ["bash", "scripts/install_deps.sh"],
                cwd=ROOT,
                env={
                    **os.environ,
                    "FAKE_PYTHON_LOG": str(log_path),
                    "LOCK_FILE": str(tmp / "missing.lock"),
                    "VENV_DIR": str(venv),
                    "DEPS_EXTRA": "training",
                    "UPGRADE_TOOLCHAIN": "0",
                },
                check=True,
                capture_output=True,
                text=True,
            )

            pip_log = log_path.read_text(encoding="utf-8")
            train_exists = (bin_dir / "aid-train").exists()

        self.assertIn("deps_lock=missing", proc.stdout)
        self.assertIn("-m pip install --progress-bar off --upgrade --upgrade-strategy eager -e .[training]", pip_log)
        self.assertTrue(train_exists)

    def _write_fake_activate(self, bin_dir: Path, venv: Path) -> None:
        (bin_dir / "activate").write_text(
            f'export VIRTUAL_ENV="{venv}"\nexport PATH="{bin_dir}:$PATH"\n',
            encoding="utf-8",
        )

    def _write_fake_python(self, bin_dir: Path) -> None:
        fake_python = bin_dir / "python"
        fake_python.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "if [[ \"$1\" == \"-m\" && \"$2\" == \"pip\" ]]; then\n"
            "  if [[ -n \"${FAKE_PYTHON_LOG:-}\" ]]; then\n"
            "    printf '%s\\n' \"$*\" >> \"$FAKE_PYTHON_LOG\"\n"
            "  fi\n"
            "  exit 0\n"
            "fi\n"
            "if [[ \"${1:-}\" == \"-\" ]]; then\n"
            "  cat >/dev/null || true\n"
            "  exit 0\n"
            "fi\n"
            "exit 0\n",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)

    def _write_fake_python3_venv_creator(self, tool_dir: Path) -> None:
        fake_python3 = tool_dir / "python3"
        fake_python3.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "if [[ \"$1\" == \"-m\" && \"$2\" == \"venv\" ]]; then\n"
            "  target=\"$3\"\n"
            "  mkdir -p \"$target/bin\"\n"
            "  cat > \"$target/bin/activate\" <<EOF\n"
            "export VIRTUAL_ENV=\"$target\"\n"
            "export PATH=\"$target/bin:$PATH\"\n"
            "EOF\n"
            "  cat > \"$target/bin/python\" <<'EOF'\n"
            "#!/usr/bin/env bash\n"
            "if [[ \"$1\" == \"-m\" && \"$2\" == \"pip\" ]]; then\n"
            "  exit 0\n"
            "fi\n"
            "exit 0\n"
            "EOF\n"
            "  chmod +x \"$target/bin/python\"\n"
            "  exit 0\n"
            "fi\n"
            "exit 99\n",
            encoding="utf-8",
        )
        fake_python3.chmod(0o755)

    def test_collection_profile_removes_stale_training_wrappers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            venv = tmp / "venv"
            bin_dir = venv / "bin"
            bin_dir.mkdir(parents=True, exist_ok=True)
            log_path = tmp / "pip.log"
            self._write_fake_activate(bin_dir, venv)
            self._write_fake_python(bin_dir)
            stale_train = bin_dir / "aid-train"
            stale_video = bin_dir / "aid-video-train"
            stale_train.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            stale_video.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            stale_train.chmod(0o755)
            stale_video.chmod(0o755)
            fake_hf = bin_dir / "hf"
            fake_hf.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            fake_hf.chmod(0o755)

            proc = subprocess.run(
                ["bash", "scripts/install_deps.sh"],
                cwd=ROOT,
                env={
                    **os.environ,
                    "FAKE_PYTHON_LOG": str(log_path),
                    "VENV_DIR": str(venv),
                    "DEPS_EXTRA": "collection",
                    "UPGRADE_TOOLCHAIN": "0",
                },
                check=True,
                capture_output=True,
                text=True,
            )

            profile_value = (venv / ".deps_profile").read_text(encoding="utf-8").strip()
            pip_log = log_path.read_text(encoding="utf-8")
            train_exists = stale_train.exists()
            video_exists = stale_video.exists()

        self.assertEqual(profile_value, "collection")
        self.assertFalse(train_exists)
        self.assertFalse(video_exists)
        self.assertIn("deps_lock=subset extra=collection", proc.stdout)
        self.assertIn("-m pip install --progress-bar off -e .[collection] --no-deps --no-build-isolation", pip_log)
        self.assertNotIn("torch==", pip_log)

    def test_training_video_profile_creates_required_wrappers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            venv = tmp / "venv"
            bin_dir = venv / "bin"
            bin_dir.mkdir(parents=True, exist_ok=True)
            log_path = tmp / "pip.log"
            self._write_fake_activate(bin_dir, venv)
            self._write_fake_python(bin_dir)

            proc = subprocess.run(
                ["bash", "scripts/install_deps.sh"],
                cwd=ROOT,
                env={
                    **os.environ,
                    "FAKE_PYTHON_LOG": str(log_path),
                    "VENV_DIR": str(venv),
                    "DEPS_EXTRA": "training,video",
                    "UPGRADE_TOOLCHAIN": "0",
                },
                check=True,
                capture_output=True,
                text=True,
            )

            train_wrapper = (bin_dir / "aid-train").read_text(encoding="utf-8")
            video_wrapper = (bin_dir / "aid-video-train").read_text(encoding="utf-8")
            profile_value = (venv / ".deps_profile").read_text(encoding="utf-8").strip()
            pip_log = log_path.read_text(encoding="utf-8")

        self.assertEqual(profile_value, "training,video")
        self.assertIn("train_main", train_wrapper)
        self.assertIn("video_train_main", video_wrapper)
        self.assertIn("deps_lock=subset extra=training,video", proc.stdout)
        self.assertIn("-m pip install --progress-bar off -e .[training,video] --no-deps --no-build-isolation", pip_log)

    def test_incomplete_venv_directory_is_recreated_and_bootstraps_successfully(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            venv = tmp / "venv"
            venv.mkdir()
            tool_dir = tmp / "tools"
            tool_dir.mkdir()
            self._write_fake_python3_venv_creator(tool_dir)

            proc = subprocess.run(
                ["bash", "scripts/install_deps.sh"],
                cwd=ROOT,
                env={
                    **os.environ,
                    "PATH": f"{tool_dir}:{os.environ['PATH']}",
                    "VENV_DIR": str(venv),
                    "DEPS_EXTRA": "training",
                    "UPGRADE_TOOLCHAIN": "0",
                },
                check=True,
                capture_output=True,
                text=True,
            )

            activate_exists = (venv / "bin" / "activate").exists()
            train_wrapper_exists = (venv / "bin" / "aid-train").exists()

        self.assertTrue(activate_exists)
        self.assertTrue(train_wrapper_exists)
        self.assertIn("deps_lock=subset extra=training", proc.stdout)

    def test_missing_collection_cli_hint_preserves_requested_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            venv = Path(tmpdir) / "venv"
            bin_dir = venv / "bin"
            bin_dir.mkdir(parents=True, exist_ok=True)
            self._write_fake_activate(bin_dir, venv)
            self._write_fake_python(bin_dir)

            proc = subprocess.run(
                ["bash", "scripts/install_deps.sh"],
                cwd=ROOT,
                env={
                    **os.environ,
                    "VENV_DIR": str(venv),
                    "DEPS_EXTRA": "collection",
                    "UPGRADE_TOOLCHAIN": "0",
                },
                capture_output=True,
                text=True,
            )

        self.assertEqual(proc.returncode, 1)
        self.assertIn("run=env DEPS_EXTRA=collection bash scripts/install_deps.sh", proc.stderr)


if __name__ == "__main__":
    unittest.main()
