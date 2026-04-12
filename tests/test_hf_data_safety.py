"""Safety helpers in ``scripts/hf_data.py`` (no ``datasets`` / Hub runtime required)."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
import tempfile
import unittest

from tests._support import SCRIPTS


def _load_hf_data():
    name = "hf_data_safety_under_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / "hf_data.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class HFDataCollectionSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.hf = _load_hf_data()

    def test_validate_hf_discovery_query_strips_and_accepts(self) -> None:
        self.assertEqual(self.hf.validate_hf_discovery_query("  cats  "), "cats")

    def test_validate_hf_discovery_query_rejects_newline(self) -> None:
        with self.assertRaises(ValueError):
            self.hf.validate_hf_discovery_query("a\nb")

    def test_validate_hf_discovery_query_rejects_oversized(self) -> None:
        name = "hf_data_safety_max8"
        try:
            os.environ["AID_MAX_HF_DISCOVERY_QUERY_CHARS"] = "8"
            spec = importlib.util.spec_from_file_location(name, SCRIPTS / "hf_data.py")
            assert spec and spec.loader
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            with self.assertRaises(ValueError):
                mod.validate_hf_discovery_query("123456789")
        finally:
            os.environ.pop("AID_MAX_HF_DISCOVERY_QUERY_CHARS", None)
            sys.modules.pop(name, None)

    def test_read_noncomment_lines_rejects_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            target = base / "t.txt"
            target.write_text("a/b\n", encoding="utf-8")
            link = base / "l.txt"
            try:
                link.symlink_to(target)
            except OSError:
                self.skipTest("symlinks not supported")
            with self.assertRaises(ValueError):
                self.hf.read_noncomment_lines(link)

    def test_read_noncomment_lines_skips_comments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "s.txt"
            p.write_text("#x\n  org/name  \n\n", encoding="utf-8")
            self.assertEqual(self.hf.read_noncomment_lines(p), ["org/name"])


if __name__ == "__main__":
    unittest.main()
