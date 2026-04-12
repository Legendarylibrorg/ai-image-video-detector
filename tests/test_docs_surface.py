from __future__ import annotations

import re
import unittest

from tests._support import ROOT


class DocsSurfaceTests(unittest.TestCase):
    def test_container_artifacts_exist(self) -> None:
        for rel_path in [
            ".dockerignore",
            "Dockerfile",
            "Dockerfile.gpu",
            "docker-compose.yml",
            "scripts/docker-entrypoint.sh",
        ]:
            with self.subTest(path=rel_path):
                self.assertTrue((ROOT / rel_path).exists())

    def test_removed_legacy_wrappers_and_stale_scripts_stay_gone(self) -> None:
        for name in ("autocollect.sh", "collect.sh", "continuous.sh", "retrain.sh", "run.sh", "start.sh", "train.sh"):
            self.assertFalse((ROOT / name).exists(), name)
        for name in ("max_quality_4090.sh", "continuous_collect.sh", "incremental_refresh.sh", "weekly_retrain_v3.sh"):
            self.assertFalse((ROOT / "scripts" / name).exists(), name)

    def test_env_example_keeps_hf_token_placeholder(self) -> None:
        text = (ROOT / ".env.example").read_text(encoding="utf-8")
        self.assertIn("HF_TOKEN=''", text)

    def test_readme_emphasizes_research_pipeline_in_title(self) -> None:
        lines = (ROOT / "README.md").read_text(encoding="utf-8").splitlines()
        self.assertTrue(lines, "README must not be empty")
        self.assertIn("Research Pipeline", lines[0])
        body = "\n".join(lines[:45])
        self.assertIn("Research pipeline:", body)
        self.assertIn("data, training, reports", body)
        self.assertIn("not serving", body)

    def test_pyproject_description_matches_readme_tagline(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        tag = "Local AI image & video research pipeline — data, training, reports (not serving)."
        self.assertIn(tag, readme)
        self.assertIn(f'description = "{tag}"', pyproject)

    def test_readme_documentation_links_resolve(self) -> None:
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        linked_docs = sorted(set(re.findall(r"\((docs/[A-Z]+\.md)\)", text)))
        self.assertEqual(linked_docs, ["docs/COMMANDS.md", "docs/REFERENCE.md", "docs/STARTUP.md"])
        for rel_path in linked_docs:
            self.assertTrue((ROOT / rel_path).exists(), rel_path)

    def test_open_source_docs_exist_and_are_linked(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        for rel_path in ["AGENTS.md", "CONTRIBUTING.md", "CODE_OF_CONDUCT.md", "SECURITY.md"]:
            with self.subTest(path=rel_path):
                self.assertTrue((ROOT / rel_path).exists(), rel_path)
                self.assertIn(rel_path, readme)

    def test_commands_doc_mentions_core_local_commands(self) -> None:
        text = (ROOT / "docs" / "COMMANDS.md").read_text(encoding="utf-8")
        for command in [
            "./local.sh collect-status",
            "./local.sh docker-doctor",
            "./local.sh finetune",
            "./local.sh train",
        ]:
            with self.subTest(command=command):
                self.assertIn(command, text)

    def test_reference_documents_operator_layers_and_bootstrap_exceptions(self) -> None:
        text = (ROOT / "docs" / "REFERENCE.md").read_text(encoding="utf-8")
        self.assertIn("## Architecture at a glance", text)
        self.assertIn("| **Bootstrap** |", text)
        self.assertIn("scripts/setup_linux.sh", text)
        self.assertIn("scripts/install_deps.sh", text)
        self.assertIn("run_repo_python", text)
        self.assertIn("| **Operator** |", text)

    def test_agents_md_orients_contributors(self) -> None:
        text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
        self.assertIn("docs/REFERENCE.md", text)
        self.assertIn("./local.sh", text)
        self.assertIn("ruff check src/ai_image_detector tests", text)
        self.assertIn("SECURITY.md", text)

    def test_contributing_links_agents_md(self) -> None:
        text = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
        self.assertIn("[AGENTS.md](AGENTS.md)", text)


if __name__ == "__main__":
    unittest.main()
