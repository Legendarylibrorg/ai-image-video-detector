from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

IMPORT_ERROR: Exception | None = None
try:
    from datasets import Dataset
    import build_best_dataset
    import hf_data
except Exception as exc:  # pragma: no cover - optional dependency path
    Dataset = None  # type: ignore[assignment]
    build_best_dataset = None  # type: ignore[assignment]
    hf_data = None  # type: ignore[assignment]
    IMPORT_ERROR = exc


@unittest.skipUnless(Dataset is not None and build_best_dataset is not None and hf_data is not None, f"optional deps unavailable: {IMPORT_ERROR}")
class HFDataTests(unittest.TestCase):
    def test_download_dataset_file_passes_cache_dir(self) -> None:
        calls: list[dict[str, object]] = []

        def fake_download(**kwargs):
            calls.append(kwargs)
            return "/tmp/file"

        import huggingface_hub

        original = huggingface_hub.hf_hub_download
        huggingface_hub.hf_hub_download = fake_download
        try:
            result = hf_data.download_dataset_file("org/repo", "file.mp4", token="tok", cache_dir="/tmp/hf-cache")
        finally:
            huggingface_hub.hf_hub_download = original

        self.assertEqual(result, "/tmp/file")
        self.assertEqual(calls[0]["repo_id"], "org/repo")
        self.assertEqual(calls[0]["filename"], "file.mp4")
        self.assertEqual(calls[0]["cache_dir"], "/tmp/hf-cache")

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
