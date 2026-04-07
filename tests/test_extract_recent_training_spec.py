from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import time
from unittest import mock
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import extract_recent_training_spec


def write_stub(path: Path, payload: bytes = b"img") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


class ExtractRecentTrainingSpecTests(unittest.TestCase):
    def test_find_latest_target_spec_prefers_most_recent_resolved_spec(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_dir = root / "alpha"
            new_dir = root / "beta"
            (old_dir / "target_spec_resolved.json").parent.mkdir(parents=True, exist_ok=True)
            (new_dir / "target_spec_resolved.json").parent.mkdir(parents=True, exist_ok=True)
            (old_dir / "target_spec_resolved.json").write_text(json.dumps({"target_name": "old target"}), encoding="utf-8")
            time.sleep(0.01)
            (new_dir / "target_spec_resolved.json").write_text(json.dumps({"target_name": "new target"}), encoding="utf-8")
            (new_dir / "target_label_aliases.json").write_text(json.dumps({"positive_dir": "ai"}), encoding="utf-8")
            (new_dir / "target_dataset_build_report.json").write_text(json.dumps({"full_targets_ok": True}), encoding="utf-8")

            result = extract_recent_training_spec.find_latest_target_spec(root)

            self.assertIsNotNone(result)
            self.assertEqual(result["mode"], "resolved_target_spec")
            self.assertEqual(result["spec"]["target_name"], "new target")
            self.assertEqual(result["aliases"]["positive_dir"], "ai")
            self.assertTrue(result["report_summary"]["full_targets_ok"])

    def test_summarize_recent_incremental_data_collects_recent_examples_and_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data_new" / "train"
            ai_path = root / "ai" / "source=model_output__reason=positive_term__ai_001.jpg"
            real_path = root / "real" / "source=review_queue__real_001.jpg"
            write_stub(ai_path)
            write_stub(real_path)
            ai_path.with_suffix(".json").write_text(
                json.dumps({"caption": "ceiling smoke detector", "objects": {"label": ["smoke detector"]}}),
                encoding="utf-8",
            )

            summary = extract_recent_training_spec.summarize_recent_incremental_data(root.parent, recent_count=4)

            self.assertEqual(summary["mode"], "recent_incremental_summary")
            self.assertEqual(summary["totals"]["ai"], 1)
            self.assertEqual(summary["totals"]["real"], 1)
            self.assertEqual(summary["top_sources"][0]["source"], "model_output")
            ai_examples = summary["classes"]["ai"]["recent_examples"]
            self.assertEqual(len(ai_examples), 1)
            self.assertIn("caption", json.dumps(ai_examples[0]["sidecar"]))
            self.assertTrue(summary["latest_addition_utc"])

    def test_build_llm_spec_extraction_prompt_embeds_schema_and_summary(self) -> None:
        summary = {
            "incremental_root": "/tmp/data_new/train",
            "latest_addition_utc": "2026-04-07T02:00:00+00:00",
            "totals": {"ai": 3, "real": 3},
            "top_sources": [{"source": "model_output", "count": 2}],
            "top_filename_tokens": [{"token": "smoke", "count": 3}, {"token": "detector", "count": 3}],
            "top_sidecar_fields": [{"field": "caption", "count": 2}],
            "classes": {"ai": {"recent_examples": []}, "real": {"recent_examples": []}},
        }

        prompt = extract_recent_training_spec.build_llm_spec_extraction_prompt(summary)

        self.assertIn("Return only JSON", prompt)
        self.assertIn('"target_name": ""', prompt)
        self.assertIn('"smoke"', prompt)
        self.assertIn('"incremental_root": "/tmp/data_new/train"', prompt)

    def test_parse_args_defaults_emit_to_json(self) -> None:
        with mock.patch.object(sys, "argv", ["extract_recent_training_spec.py"]):
            args = extract_recent_training_spec.parse_args()

        self.assertEqual(args.emit, "json")


if __name__ == "__main__":
    unittest.main()
