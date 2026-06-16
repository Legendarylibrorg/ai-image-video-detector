from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
import unittest

from tests._support import ROOT, source_tree_env


def _local_ci_text() -> str:
    return (ROOT / "scripts" / "run_ci_local.py").read_text(encoding="utf-8")


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


class LocalCiRunnerTests(unittest.TestCase):
    def test_run_ci_local_help(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "run_ci_local.py"), "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("--job", proc.stdout)
        self.assertIn("--fast", proc.stdout)

    def test_run_ci_local_list_documents_jobs(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "run_ci_local.py"), "--list"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("docs/CI_LOCAL.md", proc.stdout)
        for job in ("test", "security", "e2e-smoke"):
            self.assertIn(job, proc.stdout)

    def test_local_ci_runner_declares_test_job_steps(self) -> None:
        text = _local_ci_text()
        self.assertIn("scripts/install_deps.sh", text)
        self.assertIn('"check", "src/ai_image_detector", "tests"', text)
        self.assertIn("unittest discover", text)

    def test_local_ci_runner_declares_security_steps(self) -> None:
        text = _local_ci_text()
        self.assertIn("scripts/update_deps_lock.py", text)
        self.assertIn("verify", text)
        self.assertIn("--require-current", text)
        self.assertIn("detect-secrets-hook", text)
        self.assertIn("pip_audit", text)
        self.assertIn(r"requirements\.lock\.json", text)

    def test_local_ci_runner_declares_e2e_smoke_opt_in(self) -> None:
        text = _local_ci_text()
        self.assertIn("AID_E2E_SMOKE", text)
        self.assertIn("tests.test_e2e_smoke", text)


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
