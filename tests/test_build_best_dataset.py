from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from _support import write_rgb_image

IMPORT_ERROR: Exception | None = None
try:
    import build_best_dataset
except Exception as exc:  # pragma: no cover - optional dependency path
    build_best_dataset = None  # type: ignore[assignment]
    IMPORT_ERROR = exc


class _FakeFeature:
    def __init__(self, names):
        self.names = list(names)


class _FakeSplit:
    def __init__(self, names):
        self.features = {"label": _FakeFeature(names)}


@unittest.skipUnless(build_best_dataset is not None, f"optional deps unavailable: {IMPORT_ERROR}")
class BuildBestDatasetTests(unittest.TestCase):
    def test_source_acceptance_path_decodes_each_example_once(self) -> None:
        text = Path(build_best_dataset.__file__).read_text(encoding="utf-8")
        self.assertEqual(text.count("open_example_image(ex, image_field)"), 1)

    def test_label_resolver_uses_feature_names_for_integer_labels(self) -> None:
        resolver = build_best_dataset.build_label_resolver(_FakeSplit(["real", "synthetic"]), "label")
        self.assertEqual(resolver(0), "real")
        self.assertEqual(resolver(1), "ai")

    def test_normalize_label_rejects_ambiguous_integer_ids(self) -> None:
        self.assertIsNone(build_best_dataset.normalize_label(2))
        self.assertIsNone(build_best_dataset.normalize_label("7"))

    def test_count_existing_excludes_hardneg_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "train" / "ai").mkdir(parents=True)
            (out / "train" / "real").mkdir(parents=True)
            (out / "val" / "ai").mkdir(parents=True)
            (out / "val" / "real").mkdir(parents=True)
            (out / "test" / "ai").mkdir(parents=True)
            (out / "test" / "real").mkdir(parents=True)
            (out / "train" / "ai" / "source=foo__train_ai_0000001.jpg").touch()
            (out / "train" / "ai" / "hardneg=blur__foo__hn0000000.jpg").touch()

            raw_counts = build_best_dataset.count_output_files(out, include_hardneg=False)
            total_counts = build_best_dataset.count_output_files(out, include_hardneg=True)

            self.assertEqual(raw_counts["train"]["ai"], 1)
            self.assertEqual(total_counts["train"]["ai"], 2)

    def test_existing_dedupe_state_ignores_hardnegs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            base = out / "train" / "ai" / "source=foo__train_ai_0000001.jpg"
            hardneg = out / "train" / "ai" / "hardneg=blur__foo__hn0000000.jpg"
            write_rgb_image(base)
            write_rgb_image(hardneg)

            deduper = build_best_dataset.ImageDeduper.from_output(
                out,
                splits=build_best_dataset.SPLITS,
                classes=build_best_dataset.CLASSES,
            )

            self.assertEqual(len(deduper.seen_exact), 1)
            self.assertEqual(len(deduper.seen_dhash_by_cls["ai"]), 1)

    def test_generate_hard_negatives_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            write_rgb_image(out / "train" / "ai" / "source=foo__train_ai_0000001.jpg")
            (out / "train" / "real").mkdir(parents=True, exist_ok=True)

            targets = {"train": {"ai": 4, "real": 4}, "val": {"ai": 0, "real": 0}, "test": {"ai": 0, "real": 0}}
            build_best_dataset.generate_hard_negatives(out, targets, hardneg_fraction=0.5, jpeg_quality=92, seed=7)
            first = len(list((out / "train" / "ai").glob("hardneg=*.jpg")))
            build_best_dataset.generate_hard_negatives(out, targets, hardneg_fraction=0.5, jpeg_quality=92, seed=7)
            second = len(list((out / "train" / "ai").glob("hardneg=*.jpg")))

            self.assertEqual(first, 1)
            self.assertEqual(second, 1)

    def test_next_split_for_source_class_respects_per_split_cap(self) -> None:
        have = {
            "train": {"ai": 1, "real": 0},
            "val": {"ai": 0, "real": 0},
            "test": {"ai": 0, "real": 0},
        }
        need = {
            "train": {"ai": 5, "real": 0},
            "val": {"ai": 5, "real": 0},
            "test": {"ai": 5, "real": 0},
        }
        source_split_counts = {
            "train": {"ai": 1, "real": 0},
            "val": {"ai": 0, "real": 0},
            "test": {"ai": 0, "real": 0},
        }

        split = build_best_dataset.next_split_for_source_class(
            have,
            need,
            source_split_counts,
            "ai",
            rng=__import__("random").Random(0),
            max_per_source_split_class=1,
        )

        self.assertIsNone(split)

    def test_next_split_for_source_class_keeps_existing_source_in_one_split(self) -> None:
        have = {
            "train": {"ai": 10, "real": 0},
            "val": {"ai": 10, "real": 0},
            "test": {"ai": 10, "real": 0},
        }
        need = {
            "train": {"ai": 20, "real": 0},
            "val": {"ai": 20, "real": 0},
            "test": {"ai": 20, "real": 0},
        }
        source_split_counts = {
            "train": {"ai": 3, "real": 0},
            "val": {"ai": 0, "real": 0},
            "test": {"ai": 0, "real": 0},
        }

        split = build_best_dataset.next_split_for_source_class(
            have,
            need,
            source_split_counts,
            "ai",
            rng=__import__("random").Random(1),
            max_per_source_split_class=10,
        )

        self.assertEqual(split, "train")

    def test_next_split_for_source_class_chooses_underfilled_split_when_unassigned(self) -> None:
        have = {
            "train": {"ai": 18, "real": 0},
            "val": {"ai": 4, "real": 0},
            "test": {"ai": 4, "real": 0},
        }
        need = {
            "train": {"ai": 20, "real": 0},
            "val": {"ai": 20, "real": 0},
            "test": {"ai": 20, "real": 0},
        }
        source_split_counts = {
            "train": {"ai": 0, "real": 0},
            "val": {"ai": 0, "real": 0},
            "test": {"ai": 0, "real": 0},
        }

        split = build_best_dataset.next_split_for_source_class(
            have,
            need,
            source_split_counts,
            "ai",
            rng=__import__("random").Random(1),
            max_per_source_split_class=10,
        )

        self.assertIn(split, {"val", "test"})

    def test_main_reaches_source_validation_without_nameerror(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch(
                "sys.argv",
                [
                    "build_best_dataset.py",
                    "--out",
                    tmp,
                    "--train-per-class",
                    "0",
                    "--val-per-class",
                    "0",
                    "--test-per-class",
                    "0",
                    "--no-discover-hf",
                    "--no-default-sources",
                ],
            ):
                with self.assertRaises(SystemExit) as ctx:
                    build_best_dataset.main()

        self.assertIn("no_hf_sources_resolved", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
