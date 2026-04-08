from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from _support import ROOT, write_rgb_image  # noqa: F401

IMPORT_ERROR: Exception | None = None
try:
    import torch
    from ai_image_detector.data import unpack_image_batch
    from ai_image_detector.ensemble import metadata_features_from_paths
except Exception as exc:  # pragma: no cover - optional dependency path
    torch = None  # type: ignore[assignment]
    unpack_image_batch = None  # type: ignore[assignment]
    metadata_features_from_paths = None  # type: ignore[assignment]
    IMPORT_ERROR = exc


@unittest.skipUnless(torch is not None, f"optional deps unavailable: {IMPORT_ERROR}")
class MetadataEvalSurfaceTests(unittest.TestCase):
    def test_metadata_helper_builds_batch_tensor_from_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            img1 = Path(tmp) / "one.png"
            img2 = Path(tmp) / "two.jpg"
            write_rgb_image(img1)
            write_rgb_image(img2)

            features = metadata_features_from_paths([str(img1), str(img2)], device=torch.device("cpu"))

        self.assertEqual(features.shape[0], 2)
        self.assertGreater(features.shape[1], 0)

    def test_unpack_image_batch_supports_plain_and_metadata_batches(self) -> None:
        x = torch.zeros(2, 3, 8, 8)
        y = torch.tensor([0, 1])
        metadata = torch.ones(2, 4)

        plain = unpack_image_batch((x, y))
        with_metadata = unpack_image_batch((x, metadata, y))

        self.assertIsNone(plain[1])
        self.assertTrue(torch.equal(plain[0], x))
        self.assertTrue(torch.equal(plain[2], y))
        self.assertTrue(torch.equal(with_metadata[1], metadata))

    def test_eval_and_refinement_scripts_support_metadata_aware_models(self) -> None:
        fit_text = (ROOT / "scripts" / "fit_ensemble.py").read_text(encoding="utf-8")
        eval_text = (ROOT / "scripts" / "eval_test_ensemble.py").read_text(encoding="utf-8")
        domain_text = (ROOT / "scripts" / "fit_domain_thresholds.py").read_text(encoding="utf-8")
        robust_text = (ROOT / "src" / "ai_image_detector" / "robust_eval.py").read_text(encoding="utf-8")
        hard_text = (ROOT / "scripts" / "mine_hard_negatives.py").read_text(encoding="utf-8")
        distill_text = (ROOT / "scripts" / "train_distill.py").read_text(encoding="utf-8")

        self.assertIn("loaded.uses_metadata_features", fit_text)
        self.assertIn("MetadataImageFolder", fit_text)
        self.assertIn("build_loader_kwargs", fit_text)
        self.assertIn("make_eval_transform", fit_text)
        self.assertIn("unpack_image_batch", fit_text)
        self.assertIn("metadata_features=metadata_features", fit_text)

        self.assertIn("loaded.uses_metadata_features", eval_text)
        self.assertIn("MetadataImageFolder", eval_text)
        self.assertIn("DataLoader(", eval_text)
        self.assertIn("build_loader_kwargs", eval_text)
        self.assertIn("make_eval_transform", eval_text)
        self.assertIn("unpack_image_batch", eval_text)
        self.assertIn("metadata_features=metadata_features", eval_text)
        self.assertLess(eval_text.index("loaded = load_models"), eval_text.index("dataset_cls = MetadataImageFolder"))

        self.assertIn("metadata_features_from_paths", domain_text)
        self.assertIn("loaded.uses_metadata_features", domain_text)
        self.assertIn("make_eval_transform", domain_text)

        self.assertIn("metadata_features_from_paths", robust_text)
        self.assertIn("loaded.uses_metadata_features", robust_text)
        self.assertIn("make_eval_transform", robust_text)
        self.assertIn("batch_size = 32", robust_text)
        self.assertIn("torch.stack(variant_batches[name], dim=0)", robust_text)

        self.assertIn("loaded.uses_metadata_features", hard_text)
        self.assertIn("MetadataImageFolder", hard_text)
        self.assertIn("DataLoader(", hard_text)
        self.assertIn("make_eval_transform", hard_text)
        self.assertIn("unpack_image_batch", hard_text)
        self.assertIn("batch_paths = [path for path, _ in ds.samples[offset : offset + x.shape[0]]]", hard_text)

        self.assertIn("loaded.uses_metadata_features", distill_text)
        self.assertIn("MetadataImageFolder", distill_text)
        self.assertIn("make_eval_transform", distill_text)
        self.assertIn("unpack_image_batch", distill_text)
        self.assertIn("teacher(x, metadata_features=metadata_features)", distill_text)


if __name__ == "__main__":
    unittest.main()
