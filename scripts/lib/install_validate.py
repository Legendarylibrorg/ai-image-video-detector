#!/usr/bin/env python3
"""Validate ``install.sh`` clone parameters (install path and git remote URL)."""

from __future__ import annotations

import argparse
import os
import sys
from urllib.parse import urlparse

OFFICIAL_REPO_URLS = frozenset(
    {
        "https://github.com/Legendarylibrorg/ai-image-video-detector.git",
        "https://github.com/Legendarylibrorg/ai-image-video-detector",
    }
)

OFFICIAL_GITHUB_ORG = "legendarylibrorg"
OFFICIAL_GITHUB_REPO = "ai-image-video-detector"

FORBIDDEN_INSTALL_PREFIXES = (
    "/etc/",
    "/bin/",
    "/sbin/",
    "/boot/",
    "/dev/",
    "/proc/",
    "/sys/",
    "/run/",
)

_MAX_INSTALL_DIR_CHARS = 4096
_MAX_REPO_URL_CHARS = 2048


def _reject_embedded_credentials(parsed) -> None:
    if parsed.username or parsed.password:
        raise ValueError(
            "install_fail: repo_url_must_not_embed_credentials "
            "(use git credential helpers or a token in ~/.netrc instead of the clone URL)"
        )


def validate_install_dir(install_dir: str) -> None:
    if len(install_dir) > _MAX_INSTALL_DIR_CHARS:
        raise ValueError("install_fail: install_dir_too_long")
    if "\n" in install_dir or "\r" in install_dir:
        raise ValueError("install_fail: newline_in_install_dir")
    for c in ";|&$`":
        if c in install_dir:
            raise ValueError("install_fail: metachar_in_install_dir")
    expanded = os.path.abspath(os.path.expanduser(install_dir))
    if expanded == os.path.abspath(os.sep):
        raise ValueError("install_fail: install_dir_is_root")
    norm = expanded if expanded.endswith(os.sep) else expanded + os.sep
    for bad in FORBIDDEN_INSTALL_PREFIXES:
        rootish = bad.rstrip("/")
        if expanded == rootish or norm.startswith(bad):
            raise ValueError(f"install_fail: install_dir_forbidden_prefix prefix={bad}")


def _github_org_repo(url: str) -> tuple[str, str] | None:
    parsed = urlparse(url)
    if (parsed.hostname or "").lower() != "github.com":
        return None
    parts = [p for p in parsed.path.strip("/").split("/") if p]
    if len(parts) < 2:
        return None
    org, repo = parts[0].lower(), parts[1].lower()
    if repo.endswith(".git"):
        repo = repo[:-4]
    return org, repo


def validate_repo_for_clone(
    repo_url: str,
    *,
    allow_custom_repo: bool,
    host_allowlist_raw: str,
    allow_any_https_host: bool,
    allow_non_official_github_repo: bool,
) -> None:
    repo_url = repo_url.strip()
    if len(repo_url) > _MAX_REPO_URL_CHARS:
        raise ValueError("install_fail: repo_url_too_long")
    if not repo_url.startswith("https://"):
        if not allow_custom_repo:
            raise ValueError(
                "install_fail: repo_url_not_official set INSTALL_ALLOW_CUSTOM_REPO=1 for forks or mirrors"
            )
        raise ValueError("install_fail: custom_repo_url_must_use_https")
    parsed = urlparse(repo_url)
    _reject_embedded_credentials(parsed)
    if not allow_custom_repo:
        if repo_url not in OFFICIAL_REPO_URLS:
            raise ValueError(
                "install_fail: repo_url_not_official set INSTALL_ALLOW_CUSTOM_REPO=1 for forks or mirrors"
            )
        return
    host = (parsed.hostname or "").lower()
    if not host:
        raise ValueError("install_fail: repo_url_missing_host")
    if not allow_any_https_host:
        allow = {x.strip().lower() for x in host_allowlist_raw.split(",") if x.strip()}
        if not allow:
            raise ValueError(
                "install_fail: repo_host_allowlist_empty set INSTALL_REPO_HOST_ALLOWLIST "
                "to a comma-separated hostname list or INSTALL_ALLOW_ANY_HTTPS_HOST=1"
            )
        if host not in allow:
            raise ValueError(
                "install_fail: repo_host_not_allowlisted host="
                + repr(host)
                + " set INSTALL_REPO_HOST_ALLOWLIST or INSTALL_ALLOW_ANY_HTTPS_HOST=1"
            )

    gh = _github_org_repo(repo_url)
    if gh is None:
        return
    org, repo = gh
    if org == OFFICIAL_GITHUB_ORG and repo == OFFICIAL_GITHUB_REPO:
        return
    if not allow_non_official_github_repo:
        raise ValueError(
            "install_fail: github_repo_not_official_path set INSTALL_ALLOW_NON_OFFICIAL_GITHUB_REPO=1 "
            "when cloning a fork or renamed repo from GitHub"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--install-dir", required=True)
    ap.add_argument("--repo-url", required=True)
    ap.add_argument("--allow-custom-repo", default="0")
    args = ap.parse_args()
    allow_custom = str(args.allow_custom_repo).strip() == "1"
    host_allowlist = os.environ.get("INSTALL_REPO_HOST_ALLOWLIST", "github.com")
    allow_any = os.environ.get("INSTALL_ALLOW_ANY_HTTPS_HOST", "").strip() == "1"
    allow_non_gh = os.environ.get("INSTALL_ALLOW_NON_OFFICIAL_GITHUB_REPO", "").strip() == "1"
    try:
        validate_install_dir(args.install_dir)
        validate_repo_for_clone(
            args.repo_url,
            allow_custom_repo=allow_custom,
            host_allowlist_raw=host_allowlist,
            allow_any_https_host=allow_any,
            allow_non_official_github_repo=allow_non_gh,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
