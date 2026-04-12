from __future__ import annotations

import contextlib
import io
from pathlib import Path
import unittest
from unittest import mock

from ai_image_detector import cli


class CliTests(unittest.TestCase):
    def test_missing_dependency_message_points_to_repo_bootstrap(self) -> None:
        missing = ModuleNotFoundError("No module named 'torch'")
        missing.name = "torch"
        stderr = io.StringIO()

        with mock.patch("ai_image_detector.cli.import_module", side_effect=missing):
            with contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as raised:
                    cli.train_main()

        self.assertEqual(raised.exception.code, 2)
        message = stderr.getvalue()
        self.assertIn("missing_dependency=torch", message)
        self.assertIn("hint_extra=training", message)
        self.assertIn("run=(cd ", message)
        self.assertIn("&& env DEPS_EXTRA=training ./local.sh deps)", message)
        self.assertNotIn("run=pip install -e .", message)

    def test_missing_dependency_message_is_actionable_outside_repo(self) -> None:
        missing = ModuleNotFoundError("No module named 'torch'")
        missing.name = "torch"
        stderr = io.StringIO()

        with mock.patch("ai_image_detector.cli._repo_root", return_value=None):
            with mock.patch("ai_image_detector.cli.import_module", side_effect=missing):
                with contextlib.redirect_stderr(stderr):
                    with self.assertRaises(SystemExit) as raised:
                        cli.train_main()

        self.assertEqual(raised.exception.code, 2)
        message = stderr.getvalue()
        self.assertIn('run=python -m pip install --upgrade "ai-image-video-detector[training]"', message)
        self.assertNotIn("install_hint=install_missing_extra", message)

    def test_missing_dependency_repo_hint_quotes_paths_with_spaces(self) -> None:
        missing = ModuleNotFoundError("No module named 'torch'")
        missing.name = "torch"
        stderr = io.StringIO()

        with mock.patch("ai_image_detector.cli._repo_root", return_value=Path("/tmp/path with spaces/repo")):
            with mock.patch("ai_image_detector.cli.import_module", side_effect=missing):
                with contextlib.redirect_stderr(stderr):
                    with self.assertRaises(SystemExit):
                        cli.train_main()

        self.assertIn("run=(cd '/tmp/path with spaces/repo' && env DEPS_EXTRA=training ./local.sh deps)", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
