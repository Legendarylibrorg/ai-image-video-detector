from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys
import unittest

from _support import ROOT, source_tree_env


def _workflow_text(name: str) -> str:
    return (ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")


def _named_step_block(text: str, step_name: str) -> str:
    pattern = re.compile(
        rf"(?ms)^      - name: {re.escape(step_name)}\n(?P<body>(?:        .*\n|          .*\n)+)"
    )
    match = pattern.search(text)
    if match is None:
        raise AssertionError(f"missing workflow step {step_name!r}")
    return match.group("body")


class SmokeWorkflowTests(unittest.TestCase):
    def test_smoke_workflow_declares_expected_bootstrap_and_gate(self) -> None:
        text = _workflow_text("smoke.yml")
        install_step = _named_step_block(text, "Install")
        smoke_step = _named_step_block(text, "Unit tests and E2E smoke")

        self.assertIn("run: bash scripts/install_deps.sh", install_step)
        self.assertIn('AID_E2E_SMOKE: "1"', smoke_step)
        self.assertIn("run: .venv/bin/python -m unittest discover -s tests -p 'test_*.py'", smoke_step)

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

    def test_security_workflow_uses_locked_dependency_audit_step(self) -> None:
        text = _workflow_text("security.yml")
        audit_step = _named_step_block(text, "Dependency Vulnerability Audit")

        self.assertIn("pip-audit -r requirements.lock", audit_step)
        self.assertNotIn("|| true", audit_step)

    def test_security_workflow_verifies_lock_hashes(self) -> None:
        text = _workflow_text("security.yml")
        verify_step = _named_step_block(text, "Verify Locked Dependency Hashes")

        self.assertIn("python scripts/update_deps_lock.py verify --require-current", verify_step)


class DependencyUpdateWorkflowTests(unittest.TestCase):
    def test_dependency_update_workflow_exists_and_refreshes_lock(self) -> None:
        text = _workflow_text("deps-update.yml")
        refresh_step = _named_step_block(text, "Refresh dependency lock")
        pr_step = _named_step_block(text, "Create dependency update PR")

        self.assertIn('cron: "0 13 * * 1"', text)
        self.assertIn("workflow_dispatch:", text)
        self.assertIn("bash scripts/update_deps_lock.sh", refresh_step)
        self.assertIn("python scripts/update_deps_lock.py verify --require-current", refresh_step)
        self.assertIn("python -m unittest discover -s tests -p 'test_*.py'", text)
        self.assertIn("peter-evans/create-pull-request@v7", pr_step)


if __name__ == "__main__":
    unittest.main()
