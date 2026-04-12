from __future__ import annotations

import importlib.util
from pathlib import Path
import re
import subprocess
import sys
import unittest

from _support import ROOT, source_tree_env


def _workflow_text(name: str) -> str:
    return (ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")


def _load_update_deps_lock_module():
    path = ROOT / "scripts" / "update_deps_lock.py"
    spec = importlib.util.spec_from_file_location("update_deps_lock", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _named_step_block(text: str, step_name: str) -> str:
    pattern = re.compile(
        rf"(?ms)^      - name: {re.escape(step_name)}\n(?P<body>(?:        .*\n|          .*\n)+)"
    )
    match = pattern.search(text)
    if match is None:
        raise AssertionError(f"missing workflow step {step_name!r}")
    return match.group("body")


class SmokeResumeEvalWorkflowTests(unittest.TestCase):
    def test_smoke_script_is_shell_valid(self) -> None:
        subprocess.run(["bash", "-n", "scripts/smoke_resume_eval.sh"], cwd=ROOT, check=True)

    def test_smoke_test_module_stays_opt_in_without_workflow_env(self) -> None:
        proc = subprocess.run(
            [sys.executable, "-m", "unittest", "-v", "tests.test_e2e_smoke"],
            cwd=ROOT,
            env=source_tree_env({"AID_E2E_SMOKE": ""}),
            capture_output=True,
            text=True,
            check=True,
        )

        combined = proc.stdout + proc.stderr
        self.assertIn("OK (skipped=1)", combined)
        self.assertIn("set AID_E2E_SMOKE=1", combined)


class SecurityWorkflowTests(unittest.TestCase):
    def test_security_workflow_declares_expected_entrypoints(self) -> None:
        text = _workflow_text("security.yml")

        self.assertIn("schedule:", text)
        self.assertIn('cron: "0 14 * * 1"', text)
        self.assertIn("workflow_dispatch:", text)
        self.assertIn("timeout-minutes: 15", text)
        self.assertIn("uses: ./.github/actions/setup-aid-python", text)
        self.assertIn("${{ github.repository }}-aid-security-", text)

    def test_security_workflow_uses_locked_dependency_audit_step(self) -> None:
        text = _workflow_text("security.yml")
        audit_step = _named_step_block(text, "Audit locked dependencies")

        self.assertIn("python3 -m pip_audit -r requirements.lock", audit_step)
        self.assertNotIn("|| true", audit_step)

    def test_security_workflow_verifies_lock_hashes(self) -> None:
        text = _workflow_text("security.yml")
        verify_step = _named_step_block(text, "Verify lock digests vs PyPI")

        self.assertIn("python3 scripts/update_deps_lock.py verify --require-current", verify_step)

    def test_security_workflow_excludes_lock_manifest_from_secret_scan(self) -> None:
        text = _workflow_text("security.yml")
        secret_step = _named_step_block(text, "Secret scan")

        self.assertIn("detect-secrets-hook", secret_step)
        self.assertIn("--exclude-files", secret_step)
        self.assertIn(r"requirements\.lock\.json", secret_step)


class DependencyUpdateWorkflowTests(unittest.TestCase):
    def test_dependency_update_workflow_exists_and_refreshes_lock(self) -> None:
        text = _workflow_text("deps-update.yml")
        refresh_step = _named_step_block(text, "Refresh lock and verify PyPI digests")
        pr_step = _named_step_block(text, "Open PR if lock changed")

        self.assertIn('cron: "0 13 * * *"', text)
        self.assertIn("workflow_dispatch:", text)
        self.assertIn("uses: ./.github/actions/setup-aid-python", text)
        self.assertIn("${{ github.repository }}-aid-deps-update", text)
        self.assertIn("bash scripts/update_deps_lock.sh", refresh_step)
        self.assertIn("python3 scripts/update_deps_lock.py verify --require-current", refresh_step)
        self.assertNotIn("unittest discover", text)
        self.assertIn("peter-evans/create-pull-request@v8.1.1", pr_step)


class CodeqlWorkflowTests(unittest.TestCase):
    def test_codeql_workflow_is_python_only(self) -> None:
        text = _workflow_text("codeql.yml")

        self.assertIn("name: Code scanning (Python)", text)
        self.assertIn("languages: python", text)
        self.assertIn("build-mode: none", text)
        self.assertRegex(text, r"(?m)^\s*languages:\s*python\s*$")
        self.assertNotIn("github/codeql-action/autobuild", text)
        self.assertIn("github/codeql-action/init@v4.35.1", text)
        self.assertIn("github/codeql-action/analyze@v4.35.1", text)
        self.assertIn("pull_request:", text)
        self.assertIn('branches: ["main"]', text)
        self.assertIn("cancel-in-progress: false", text)
        self.assertIn('cron: "30 6 * * 1"', text)
        self.assertIn('AID_CI_IMPORT_PACKAGE: "ai_image_detector"', text)
        self.assertIn('AID_CI_PYPROJECT_NAME: "ai-image-detector"', text)
        self.assertIn("name: Analyze (${{ env.AID_CI_IMPORT_PACKAGE }})", text)
        self.assertIn("${{ github.repository }}-aid-codeql-", text)


class CiPythonVersionSourceTests(unittest.TestCase):
    def test_ci_python_version_file_matches_manifest_wheel_cap(self) -> None:
        try:
            from packaging.version import Version
        except ModuleNotFoundError:  # pragma: no cover
            from pip._vendor.packaging.version import Version  # type: ignore[no-redef]

        raw = (ROOT / ".github" / "ci-python-version.txt").read_text(encoding="utf-8")
        token = "".join(raw.split())
        self.assertTrue(token, "ci-python-version.txt must be non-empty")
        ci_ver = Version(token)
        mod = _load_update_deps_lock_module()
        self.assertEqual(mod.MANIFEST_MAX_WHEEL_CP, ci_ver)

    def test_setup_aid_python_action_reads_version_file(self) -> None:
        text = (ROOT / ".github" / "actions" / "setup-aid-python" / "action.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("ci-python-version.txt", text)
        self.assertIn("actions/setup-python@v6.2.0", text)


if __name__ == "__main__":
    unittest.main()
