from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

from ai_image_detector.collection_paths import (
    collection_workspace_root,
    require_under_collection_workspace,
    validate_collection_io_paths,
)


class CollectionPathsTests(unittest.TestCase):
    def test_require_under_rejects_relative_escape(self) -> None:
        with tempfile.TemporaryDirectory() as parent:
            parent_p = Path(parent).resolve()
            wa = parent_p / "workspace"
            wa.mkdir(parents=True)
            outsider = parent_p / "outside.txt"
            outsider.write_text("x", encoding="utf-8")
            ok = wa / "ok.txt"
            ok.write_text("y", encoding="utf-8")
            old_cwd = os.getcwd()
            try:
                os.chdir(wa)
                require_under_collection_workspace("ok.txt", wa)
                with self.assertRaises(ValueError):
                    require_under_collection_workspace("../outside.txt", wa)
            finally:
                os.chdir(old_cwd)

    def test_require_under_rejects_absolute_outside_cwd(self) -> None:
        old_aid = os.environ.get("AID_WORKSPACE_ROOT")
        try:
            os.environ.pop("AID_WORKSPACE_ROOT", None)
            with tempfile.TemporaryDirectory() as outsider_s:
                outsider = Path(outsider_s).resolve()
                f = outsider / "f.txt"
                f.write_text("x", encoding="utf-8")
                with self.assertRaises(ValueError):
                    require_under_collection_workspace(f)
        finally:
            if old_aid is not None:
                os.environ["AID_WORKSPACE_ROOT"] = old_aid

    def test_require_under_strict_env_rejects_absolute_outside(self) -> None:
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            wa = Path(a).resolve()
            wb = Path(b).resolve()
            secret = wb / "secret"
            secret.write_text("x", encoding="utf-8")
            old = os.environ.get("AID_WORKSPACE_ROOT")
            try:
                os.environ["AID_WORKSPACE_ROOT"] = str(wa)
                with self.assertRaises(ValueError):
                    require_under_collection_workspace(secret)
            finally:
                if old is None:
                    os.environ.pop("AID_WORKSPACE_ROOT", None)
                else:
                    os.environ["AID_WORKSPACE_ROOT"] = old

    def test_workspace_root_follows_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            w = Path(tmp).resolve()
            old = os.environ.get("AID_WORKSPACE_ROOT")
            try:
                os.environ["AID_WORKSPACE_ROOT"] = str(w)
                self.assertEqual(collection_workspace_root(), w)
            finally:
                if old is None:
                    os.environ.pop("AID_WORKSPACE_ROOT", None)
                else:
                    os.environ["AID_WORKSPACE_ROOT"] = old

    def test_validate_collection_io_paths_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            w = Path(tmp).resolve()
            (w / "out").mkdir()
            validate_collection_io_paths(workspace=w, out=w / "out", cache_dir=w / ".cache")

    def test_require_under_rejects_symlink_leaf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            w = Path(tmp).resolve()
            target = w / "real_out"
            target.mkdir()
            link = w / "out_link"
            try:
                link.symlink_to(target)
            except OSError:
                self.skipTest("symlinks not supported")
            with self.assertRaises(ValueError) as ctx:
                require_under_collection_workspace(link, w)
            self.assertIn("symlink_not_allowed", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
