from __future__ import annotations

import unittest

from _support import ROOT

import importlib.util


def _load_install_validate():
    path = ROOT / "scripts" / "lib" / "install_validate.py"
    spec = importlib.util.spec_from_file_location("install_validate", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


iv = _load_install_validate()


class InstallValidateTests(unittest.TestCase):
    def test_official_repo_ok_without_custom_flag(self) -> None:
        iv.validate_repo_for_clone(
            "https://github.com/Legendarylibrorg/ai-image-video-detector.git",
            allow_custom_repo=False,
            host_allowlist_raw="github.com",
            allow_any_https_host=False,
            allow_non_official_github_repo=False,
        )

    def test_non_official_repo_rejects_without_custom_flag(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            iv.validate_repo_for_clone(
                "https://github.com/example/fork.git",
                allow_custom_repo=False,
                host_allowlist_raw="github.com",
                allow_any_https_host=False,
                allow_non_official_github_repo=False,
            )
        self.assertIn("repo_url_not_official", str(ctx.exception))

    def test_github_fork_requires_non_official_flag_when_custom(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            iv.validate_repo_for_clone(
                "https://github.com/example/fork.git",
                allow_custom_repo=True,
                host_allowlist_raw="github.com",
                allow_any_https_host=False,
                allow_non_official_github_repo=False,
            )
        self.assertIn("github_repo_not_official_path", str(ctx.exception))

    def test_gitlab_requires_host_allowlist(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            iv.validate_repo_for_clone(
                "https://gitlab.com/org/project.git",
                allow_custom_repo=True,
                host_allowlist_raw="github.com",
                allow_any_https_host=False,
                allow_non_official_github_repo=False,
            )
        self.assertIn("repo_host_not_allowlisted", str(ctx.exception))

    def test_empty_host_allowlist_rejected(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            iv.validate_repo_for_clone(
                "https://github.com/example/fork.git",
                allow_custom_repo=True,
                host_allowlist_raw=" , , ",
                allow_any_https_host=False,
                allow_non_official_github_repo=True,
            )
        self.assertIn("allowlist_empty", str(ctx.exception))

    def test_install_dir_rejects_etc_prefix(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            iv.validate_install_dir("/etc/foo/bar")
        self.assertIn("forbidden_prefix", str(ctx.exception))

    def test_install_dir_rejects_newline(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            iv.validate_install_dir("/tmp/evil\n")
        self.assertIn("newline", str(ctx.exception))

    def test_repo_url_rejects_embedded_credentials_official(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            iv.validate_repo_for_clone(
                "https://x:y@github.com/Legendarylibrorg/ai-image-video-detector.git",
                allow_custom_repo=False,
                host_allowlist_raw="github.com",
                allow_any_https_host=False,
                allow_non_official_github_repo=False,
            )
        self.assertIn("credentials", str(ctx.exception))

    def test_repo_url_rejects_embedded_credentials_custom(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            iv.validate_repo_for_clone(
                "https://token@github.com/example/fork.git",
                allow_custom_repo=True,
                host_allowlist_raw="github.com",
                allow_any_https_host=False,
                allow_non_official_github_repo=True,
            )
        self.assertIn("credentials", str(ctx.exception))

    def test_repo_url_strips_whitespace(self) -> None:
        iv.validate_repo_for_clone(
            "  https://github.com/Legendarylibrorg/ai-image-video-detector.git  ",
            allow_custom_repo=False,
            host_allowlist_raw="github.com",
            allow_any_https_host=False,
            allow_non_official_github_repo=False,
        )

    def test_repo_url_rejects_excessive_length(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            iv.validate_repo_for_clone(
                "https://github.com/x/y.git" + ("a" * 3000),
                allow_custom_repo=True,
                host_allowlist_raw="github.com",
                allow_any_https_host=False,
                allow_non_official_github_repo=True,
            )
        self.assertIn("too_long", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
