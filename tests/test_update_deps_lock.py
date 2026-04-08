from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import update_deps_lock


class UpdateDepsLockTests(unittest.TestCase):
    def test_select_preferred_artifact_prefers_linux_cp310_wheel_when_no_sdist_exists(self) -> None:
        artifact = update_deps_lock.select_preferred_artifact(
            [
                {"filename": "torch-2.11.0-cp310-cp310-manylinux_2_28_aarch64.whl", "packagetype": "bdist_wheel"},
                {"filename": "torch-2.11.0-cp310-cp310-macosx_11_0_arm64.whl", "packagetype": "bdist_wheel"},
                {"filename": "torch-2.11.0-cp310-cp310-manylinux_2_28_x86_64.whl", "packagetype": "bdist_wheel"},
                {"filename": "torch-2.11.0-cp311-cp311-manylinux_2_28_x86_64.whl", "packagetype": "bdist_wheel"},
            ]
        )

        self.assertEqual(artifact["filename"], "torch-2.11.0-cp310-cp310-manylinux_2_28_x86_64.whl")

    def test_latest_compatible_release_skips_newer_python_incompatible_version(self) -> None:
        payload = {
            "releases": {
                "2.4.4": [{"filename": "numpy-2.4.4.tar.gz", "packagetype": "sdist", "yanked": False, "requires_python": ">=3.11", "digests": {"sha256": "new"}}],
                "2.2.6": [{"filename": "numpy-2.2.6.tar.gz", "packagetype": "sdist", "yanked": False, "requires_python": ">=3.10", "digests": {"sha256": "old"}}],
            }
        }
        with mock.patch("update_deps_lock.fetch_json", return_value=payload):
            version, _ = update_deps_lock.latest_compatible_release("numpy", update_deps_lock.Version("3.10"))

        self.assertEqual(version, "2.2.6")

    def test_verify_manifest_checks_hashes_against_release_metadata(self) -> None:
        manifest = {
            "python_requires": ">=3.10",
            "packages": [
                {
                    "name": "datasets",
                    "project": "datasets",
                    "version": "4.8.4",
                    "artifact": {
                        "filename": "datasets-4.8.4.tar.gz",
                        "packagetype": "sdist",
                        "sha256": "abc123",
                        "url": "https://files.pythonhosted.org/datasets-4.8.4.tar.gz",
                    },
                }
            ],
        }
        release_payload = {
            "urls": [
                {
                    "filename": "datasets-4.8.4.tar.gz",
                    "packagetype": "sdist",
                    "yanked": False,
                    "digests": {"sha256": "abc123"},
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            lock_file = tmp / "requirements.lock"
            manifest_file = tmp / "requirements.lock.json"
            lock_file.write_text("datasets==4.8.4\n", encoding="utf-8")
            manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

            with mock.patch("update_deps_lock.fetch_json", return_value=release_payload):
                update_deps_lock.verify_manifest(lock_file, manifest_file, require_current=False)


if __name__ == "__main__":
    unittest.main()
