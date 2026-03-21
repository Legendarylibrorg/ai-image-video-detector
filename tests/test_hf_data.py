from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

from datasets import Dataset


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_best_dataset
import hf_data


class HFDataTests(unittest.TestCase):
    def test_normalize_image_dataset_split_maps_and_filters_labels(self) -> None:
        ds = Dataset.from_dict(
            {
                "label": ["0", "1", "9", "real"],
                "image": ["a", "b", "c", "d"],
            }
        )

        normalized = hf_data.normalize_image_dataset_split(
            ds,
            label_field="label",
            resolve_label=build_best_dataset.normalize_label,
        )

        labels = list(normalized["_normalized_label"])
        self.assertEqual(labels, ["real", "ai", "real"])

    def test_source_manifest_helpers_round_trip_latest_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.jsonl"
            hf_data.append_source_manifest_entry(path, {"source": "one", "accepted_total": 0})
            hf_data.append_source_manifest_entry(path, {"source": "one", "accepted_total": 2})
            hf_data.append_source_manifest_entry(path, {"source": "two", "accepted_total": 1})

            latest = hf_data.load_latest_source_manifest(path)

            self.assertEqual(latest["one"]["accepted_total"], 2)
            self.assertEqual(latest["two"]["accepted_total"], 1)

    def test_skip_source_from_manifest_requires_matching_policy(self) -> None:
        policy = {
            "streaming": True,
            "stream_buffer_size": 120,
            "max_samples_per_source": 1000,
            "acceptance_warmup_samples": 100,
            "min_acceptance_rate": 0.01,
        }
        entry = {
            "source": "repo/example",
            "skip_future_runs": True,
            "policy": dict(policy),
        }

        self.assertTrue(build_best_dataset.should_skip_source_from_manifest(entry, policy))
        self.assertFalse(
            build_best_dataset.should_skip_source_from_manifest(
                {**entry, "policy": {**policy, "max_samples_per_source": 500}},
                policy,
            )
        )


if __name__ == "__main__":
    unittest.main()
