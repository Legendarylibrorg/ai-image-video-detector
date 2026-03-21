from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TrainingSurfaceTests(unittest.TestCase):
    def test_pyproject_removes_serve_entrypoint_and_web_dependencies(self) -> None:
        text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertNotIn("aid-serve", text)
        self.assertNotIn("fastapi", text.lower())
        self.assertNotIn("uvicorn", text.lower())

    def test_api_module_is_removed(self) -> None:
        self.assertFalse((ROOT / "src" / "ai_image_detector" / "api.py").exists())

    def test_local_retrain_defaults_to_train_existing(self) -> None:
        text = (ROOT / "scripts" / "local_retrain_4090.sh").read_text(encoding="utf-8")
        self.assertIn('PIPELINE_CMD="${PIPELINE_CMD:-bash scripts/do.sh train-existing}"', text)
        self.assertNotIn("start-v2", text)
        self.assertIn("--skip-video", text)


if __name__ == "__main__":
    unittest.main()
