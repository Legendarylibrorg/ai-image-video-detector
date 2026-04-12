from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest


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

    def test_load_hf_dataset_disables_hub_remote_code_by_default(self) -> None:
        captured: list[dict[str, object]] = []
        stub = {"train": Dataset.from_dict({"k": [1]})}

        def fake_load(_sid: str, **kw: object) -> dict:
            captured.append(dict(kw))
            return dict(stub)

        original = hf_data.load_dataset
        hf_data.load_dataset = fake_load
        try:
            os.environ.pop("AID_HF_TRUST_REMOTE_CODE", None)
            src = hf_data.load_hf_dataset_source("org/name", token=None, streaming=True, cache_dir=None)
            self.assertEqual(src.split_name, "train")
        finally:
            hf_data.load_dataset = original
        self.assertTrue(captured)
        self.assertIs(captured[0].get("trust_remote_code"), False)

    def test_validate_hf_dataset_source_id_rejects_path_tokens(self) -> None:
        with self.assertRaises(ValueError):
            hf_data.validate_hf_dataset_source_id("../evil/name")
        with self.assertRaises(ValueError):
            hf_data.validate_hf_dataset_source_id("org")

    def test_validate_hf_repo_blob_path_rejects_traversal(self) -> None:
        with self.assertRaises(ValueError):
            hf_data.validate_hf_repo_blob_path("../secret")
        with self.assertRaises(ValueError):
            hf_data.validate_hf_repo_blob_path("a/../b.bin")

    def test_trust_remote_allowlist_restricts_flag(self) -> None:
        captured: list[dict[str, object]] = []
        stub = {"train": Dataset.from_dict({"k": [1]})}

        def fake_load(_sid: str, **kw: object) -> dict:
            captured.append(dict(kw))
            return dict(stub)

        original = hf_data.load_dataset
        hf_data.load_dataset = fake_load
        try:
            os.environ["AID_HF_TRUST_REMOTE_CODE"] = "1"
            os.environ["AID_HF_TRUST_REMOTE_ALLOWLIST"] = "allowed/repo"
            hf_data.load_hf_dataset_source("other/repo", token=None, streaming=True, cache_dir=None)
        finally:
            hf_data.load_dataset = original
            os.environ.pop("AID_HF_TRUST_REMOTE_CODE", None)
            os.environ.pop("AID_HF_TRUST_REMOTE_ALLOWLIST", None)
        self.assertTrue(captured)
        self.assertIs(captured[0].get("trust_remote_code"), False)

        captured.clear()
        hf_data.load_dataset = fake_load
        try:
            os.environ["AID_HF_TRUST_REMOTE_CODE"] = "1"
            os.environ["AID_HF_TRUST_REMOTE_ALLOWLIST"] = "allowed/repo"
            hf_data.load_hf_dataset_source("allowed/repo", token=None, streaming=True, cache_dir=None)
        finally:
            hf_data.load_dataset = original
            os.environ.pop("AID_HF_TRUST_REMOTE_CODE", None)
            os.environ.pop("AID_HF_TRUST_REMOTE_ALLOWLIST", None)
        self.assertTrue(captured)
        self.assertIs(captured[0].get("trust_remote_code"), True)

    def test_trust_remote_without_allowlist_stays_false(self) -> None:
        captured: list[dict[str, object]] = []
        stub = {"train": Dataset.from_dict({"k": [1]})}

        def fake_load(_sid: str, **kw: object) -> dict:
            captured.append(dict(kw))
            return dict(stub)

        original = hf_data.load_dataset
        hf_data.load_dataset = fake_load
        try:
            os.environ["AID_HF_TRUST_REMOTE_CODE"] = "1"
            os.environ.pop("AID_HF_TRUST_REMOTE_ALLOWLIST", None)
            os.environ.pop("AID_HF_TRUST_REMOTE_UNSAFE_GLOBAL", None)
            hf_data.load_hf_dataset_source("org/name", token=None, streaming=True, cache_dir=None)
        finally:
            hf_data.load_dataset = original
            os.environ.pop("AID_HF_TRUST_REMOTE_CODE", None)
        self.assertTrue(captured)
        self.assertIs(captured[0].get("trust_remote_code"), False)

    def test_trust_remote_unsafe_global_restores_legacy_flag(self) -> None:
        captured: list[dict[str, object]] = []
        stub = {"train": Dataset.from_dict({"k": [1]})}

        def fake_load(_sid: str, **kw: object) -> dict:
            captured.append(dict(kw))
            return dict(stub)

        original = hf_data.load_dataset
        hf_data.load_dataset = fake_load
        try:
            os.environ["AID_HF_TRUST_REMOTE_CODE"] = "1"
            os.environ["AID_HF_TRUST_REMOTE_UNSAFE_GLOBAL"] = "1"
            hf_data.load_hf_dataset_source("org/name", token=None, streaming=True, cache_dir=None)
        finally:
            hf_data.load_dataset = original
            os.environ.pop("AID_HF_TRUST_REMOTE_CODE", None)
            os.environ.pop("AID_HF_TRUST_REMOTE_UNSAFE_GLOBAL", None)
        self.assertTrue(captured)
        self.assertIs(captured[0].get("trust_remote_code"), True)


if __name__ == "__main__":
    unittest.main()
