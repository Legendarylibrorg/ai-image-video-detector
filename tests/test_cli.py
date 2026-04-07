from __future__ import annotations

import contextlib
import io
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
        self.assertIn('&& env DEPS_EXTRA="training" ./local.sh deps)', message)
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
        self.assertIn('run=python -m pip install --upgrade "ai-image-detector[training]"', message)
        self.assertNotIn("install_hint=install_missing_extra", message)


if __name__ == "__main__":
    unittest.main()
