from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from _support import source_tree_env, write_rgb_image


ROOT = Path(__file__).resolve().parents[1]


class MetadataFinetuneSurfaceTests(unittest.TestCase):
    def test_metadata_module_exposes_fixed_feature_vector_helpers(self) -> None:
        text = (ROOT / "src" / "ai_image_detector" / "metadata.py").read_text(encoding="utf-8")
        self.assertIn("METADATA_FEATURE_NAMES = (", text)
        self.assertIn('"provenance_score"', text)
        self.assertIn('"text_score"', text)
        self.assertIn("def extract_metadata_features(image_path: str) -> list[float]:", text)
        self.assertIn("def metadata_feature_dim() -> int:", text)

    def test_data_loader_supports_optional_metadata_features(self) -> None:
        text = (ROOT / "src" / "ai_image_detector" / "data.py").read_text(encoding="utf-8")
        self.assertIn("class MetadataImageFolder", text)
        self.assertIn("extract_metadata_features(path)", text)
        self.assertIn("make_jailed_rgb_loader", text)
        self.assertIn("use_metadata_features: bool = False", text)
        self.assertIn("metadata_feature_dim() if use_metadata_features else 0", text)

    def test_train_and_infer_support_metadata_features(self) -> None:
        train_text = (ROOT / "src" / "ai_image_detector" / "train.py").read_text(encoding="utf-8")
        infer_text = (ROOT / "src" / "ai_image_detector" / "infer.py").read_text(encoding="utf-8")
        ensemble_text = (ROOT / "src" / "ai_image_detector" / "ensemble.py").read_text(encoding="utf-8")
        wrapper_text = (ROOT / "scripts" / "metadata_finetune_4090.sh").read_text(encoding="utf-8")
        self.assertIn("--use-metadata-features", train_text)
        self.assertIn("--init-from", train_text)
        self.assertIn("metadata_feature_dim=metadata_dim", train_text)
        self.assertIn("MetadataImageFolder if bool(best.get(\"use_metadata_features\", False)) else datasets.ImageFolder", train_text)
        self.assertIn("extract_metadata_features", infer_text)
        self.assertIn("metadata_features=metadata_features", infer_text)
        self.assertIn('metadata_dim = int(ckpt.get("metadata_feature_dim", 0))', ensemble_text)
        self.assertIn('declare -a metadata_candidates=("$BASE_CKPT_SEARCH_ROOT"/m*/best.safetensors)', wrapper_text)
        self.assertIn('searched=$BASE_CKPT_SEARCH_ROOT/m*/best.safetensors', wrapper_text)

    def test_metadata_cli_strip_command_runs_after_lazy_import_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "input.jpg"
            output_path = tmp_path / "output.jpg"
            write_rgb_image(input_path, color=(255, 255, 255), size=(8, 8))

            proc = subprocess.run(
                [
                    os.environ.get("PYTHON", "") or os.sys.executable,
                    "-m",
                    "ai_image_detector.metadata",
                    "strip",
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
                env=source_tree_env(),
            )

            self.assertIn("saved stripped image", proc.stdout)
            self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
