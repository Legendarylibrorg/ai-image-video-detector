from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from _support import ROOT, SCRIPTS

IMPORT_ERROR: Exception | None = None
try:
    import build_target_dataset
    from image_materialize import ImageQualityPolicy
except Exception as exc:  # pragma: no cover - optional dependency path
    build_target_dataset = None  # type: ignore[assignment]
    ImageQualityPolicy = None  # type: ignore[assignment]
    IMPORT_ERROR = exc


def image_stub(seed: int):
    if build_target_dataset is None:  # pragma: no cover - guarded by class skip
        raise RuntimeError("build_target_dataset unavailable")
    image = build_target_dataset.Image.new(
        "RGB",
        (32, 32),
        ((seed * 29) % 255, (seed * 47) % 255, (seed * 61) % 255),
    )
    stripe_x = seed % 16
    accent = (255 - ((seed * 29) % 255), 255 - ((seed * 47) % 255), 255 - ((seed * 61) % 255))
    for x in range(stripe_x, min(32, stripe_x + 8)):
        for y in range(32):
            image.putpixel((x, y), accent)
    return image


class FakeDataset:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows
        self.column_names = sorted({key for row in rows for key in row})

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.rows[index]


@unittest.skipUnless(build_target_dataset is not None and ImageQualityPolicy is not None, f"optional deps unavailable: {IMPORT_ERROR}")
class BuildTargetDatasetTests(unittest.TestCase):
    def test_build_llm_target_spec_prompt_exposes_json_schema(self) -> None:
        spec = build_target_dataset.build_default_target_spec(
            target_name="smoke detector",
            target_description="ceiling mounted smoke alarm",
        )

        prompt = build_target_dataset.build_llm_target_spec_prompt(spec)

        self.assertIn("Return only JSON", prompt)
        self.assertIn('"target_name": "smoke detector"', prompt)
        self.assertIn('"positive_terms"', prompt)
        self.assertIn('"treat_other_labeled_as_negative"', prompt)

    def test_build_match_result_handles_nested_target_labels_and_exclusions(self) -> None:
        spec = build_target_dataset.build_default_target_spec(
            target_name="smoke detector",
            positive_terms=["fire alarm"],
            exclude_terms=["toy"],
            text_fields=["objects.category", "label"],
        )

        positive = build_target_dataset.build_match_result(
            {
                "image": image_stub(1),
                "objects": {"category": ["person", "smoke detector"]},
                "caption": "kitchen ceiling alarm",
            },
            source_id="org/alarms",
            row_index=0,
            image_field="image",
            spec=spec,
        )
        excluded = build_target_dataset.build_match_result(
            {
                "image": image_stub(2),
                "label": "smoke detector toy",
            },
            source_id="org/alarms",
            row_index=1,
            image_field="image",
            spec=spec,
        )
        negative = build_target_dataset.build_match_result(
            {
                "image": image_stub(3),
                "label": "cat",
            },
            source_id="org/alarms",
            row_index=2,
            image_field="image",
            spec=spec,
        )

        self.assertEqual(positive.label, "positive")
        self.assertEqual(positive.reason, "positive_term")
        self.assertIn("smoke detector", positive.positive_hits)
        self.assertIsNone(excluded.label)
        self.assertEqual(excluded.reason, "excluded_terms")
        self.assertEqual(negative.label, "negative")
        self.assertEqual(negative.reason, "other_labeled_example")

    def test_validate_output_class_dirs_rejects_nonstandard_layout_by_default(self) -> None:
        with self.assertRaises(SystemExit):
            build_target_dataset.validate_output_class_dirs("target", "background", allow_nonstandard=False)
        build_target_dataset.validate_output_class_dirs("ai", "real", allow_nonstandard=False)
        build_target_dataset.validate_output_class_dirs("target", "background", allow_nonstandard=True)

    def test_count_output_files_uses_shared_image_extension_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train" / "ai").mkdir(parents=True, exist_ok=True)
            (root / "train" / "real").mkdir(parents=True, exist_ok=True)
            (root / "train" / "ai" / "positive.png").write_bytes(b"png")
            (root / "train" / "real" / "negative.webp").write_bytes(b"webp")

            counts = build_target_dataset._count_output_files(root, positive_dir="ai", negative_dir="real")

            self.assertEqual(counts["train"]["ai"], 1)
            self.assertEqual(counts["train"]["real"], 1)

    def test_build_target_dataset_from_sources_is_balanced_and_deterministic(self) -> None:
        spec = build_target_dataset.build_default_target_spec(
            target_name="smoke detector",
            positive_terms=["fire alarm"],
        )
        ds_one = FakeDataset(
            [
                {"image": image_stub(1), "label": "smoke detector"},
                {"image": image_stub(2), "label": "fire alarm"},
                {"image": image_stub(3), "label": "cat"},
                {"image": image_stub(4), "label": "dog"},
                {"image": image_stub(5), "label": "smoke detector"},
                {"image": image_stub(6), "label": "cat"},
            ]
        )
        ds_two = FakeDataset(
            [
                {"image": image_stub(11), "label": "smoke detector"},
                {"image": image_stub(12), "label": "dog"},
                {"image": image_stub(13), "label": "fire alarm"},
                {"image": image_stub(14), "label": "cat"},
                {"image": image_stub(15), "label": "smoke detector"},
                {"image": image_stub(16), "label": "bird"},
            ]
        )
        loaded_sources = [
            SimpleNamespace(source_id="org/alarms-one", split_name="train", split=ds_one, streaming=False),
            SimpleNamespace(source_id="org/alarms-two", split_name="train", split=ds_two, streaming=False),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_one = root / "out_one"
            out_two = root / "out_two"

            summary_one = build_target_dataset.build_target_dataset_from_sources(
                loaded_sources,
                spec=spec,
                out=out_one,
                positive_dir="ai",
                negative_dir="real",
                train_per_class=2,
                val_per_class=1,
                test_per_class=1,
                quality_policy=ImageQualityPolicy(min_side=8, max_aspect_ratio=8.0, min_entropy=0.0),
                near_hamming=0,
                near_window=0,
                max_per_source_class=4,
                max_per_source_split_class=2,
                max_samples_per_source=20,
                jpeg_quality=90,
            )
            summary_two = build_target_dataset.build_target_dataset_from_sources(
                loaded_sources,
                spec=spec,
                out=out_two,
                positive_dir="ai",
                negative_dir="real",
                train_per_class=2,
                val_per_class=1,
                test_per_class=1,
                quality_policy=ImageQualityPolicy(min_side=8, max_aspect_ratio=8.0, min_entropy=0.0),
                near_hamming=0,
                near_window=0,
                max_per_source_class=4,
                max_per_source_split_class=2,
                max_samples_per_source=20,
                jpeg_quality=90,
            )

            self.assertTrue(summary_one["full_targets_ok"])
            self.assertEqual(summary_one["final_counts"]["train"]["ai"], 2)
            self.assertEqual(summary_one["final_counts"]["train"]["real"], 2)
            self.assertEqual(summary_one["final_counts"]["val"]["ai"], 1)
            self.assertEqual(summary_one["final_counts"]["test"]["real"], 1)
            self.assertEqual(
                sorted(str(path.relative_to(out_one)) for path in out_one.rglob("*.jpg")),
                sorted(str(path.relative_to(out_two)) for path in out_two.rglob("*.jpg")),
            )
            aliases = json.loads((out_one / "target_label_aliases.json").read_text(encoding="utf-8"))
            self.assertEqual(aliases["positive_dir"], "ai")
            self.assertEqual(aliases["positive_label"], "smoke detector")
            self.assertTrue((out_one / "target_dataset_build_report.json").exists())
            self.assertEqual(summary_two["final_counts"], summary_one["final_counts"])


if __name__ == "__main__":
    unittest.main()
