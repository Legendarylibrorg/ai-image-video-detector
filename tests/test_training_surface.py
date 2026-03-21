from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TrainingSurfaceTests(unittest.TestCase):
    def test_pyproject_keeps_only_pipeline_entrypoints_and_no_web_dependencies(self) -> None:
        text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertNotIn("aid-serve", text)
        self.assertNotIn("aid-detect", text)
        self.assertNotIn("aid-explain", text)
        self.assertNotIn("aid-metadata", text)
        self.assertNotIn("aid-robust-eval", text)
        self.assertNotIn("aid-video-detect", text)
        self.assertNotIn("aid-video-detect-temporal", text)
        self.assertNotIn("aid-train-advanced", text)
        self.assertIn('aid-train = "ai_image_detector.train:main"', text)
        self.assertIn('aid-video-train = "ai_image_detector.video_temporal:train_main"', text)
        self.assertIn('aid-dataset = "ai_image_detector.dataset_tools:main"', text)
        self.assertNotIn("fastapi", text.lower())
        self.assertNotIn("uvicorn", text.lower())

    def test_api_module_is_removed(self) -> None:
        self.assertFalse((ROOT / "src" / "ai_image_detector" / "api.py").exists())

    def test_local_retrain_defaults_to_train_existing(self) -> None:
        text = (ROOT / "scripts" / "local_retrain_4090.sh").read_text(encoding="utf-8")
        self.assertIn('PIPELINE_CMD="${PIPELINE_CMD:-bash scripts/do.sh train-existing}"', text)
        self.assertNotIn("start-v2", text)
        self.assertIn("--skip-video", text)

    def test_reference_doc_stays_pipeline_focused(self) -> None:
        text = (ROOT / "docs" / "REFERENCE.md").read_text(encoding="utf-8")
        self.assertIn("Pipeline tools", text)
        self.assertIn("RTX 4090", text)
        self.assertNotIn("aid-detect --model", text)
        self.assertNotIn("aid-explain --model", text)
        self.assertNotIn("aid-video-detect-temporal", text)


if __name__ == "__main__":
    unittest.main()
