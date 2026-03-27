from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from ai_image_detector.checkpoints import save_safetensors_checkpoint
from ai_image_detector.ensemble import EnsembleDetector, load_models, stack_model_logits
from ai_image_detector.model import build_model
from ai_image_detector.train import BinaryClassificationLoss


class RecordingModel(torch.nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = float(value)
        self.seen_shapes: list[tuple[int, int]] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.seen_shapes.append((int(x.shape[-2]), int(x.shape[-1])))
        return torch.full((x.shape[0],), self.value, dtype=x.dtype, device=x.device)


class EnsembleTests(unittest.TestCase):
    def test_build_model_supports_convnext_tiny(self) -> None:
        model = build_model(backbone="convnext_tiny", pretrained_backbone=False)

        with torch.no_grad():
            out = model(torch.zeros(2, 3, 64, 64))

        self.assertEqual(tuple(out.shape), (2,))

    def test_binary_classification_loss_supports_soft_targets(self) -> None:
        logits = torch.tensor([0.25, -0.5, 0.9], dtype=torch.float32)
        targets = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float32)

        for kind in ("ce", "bce", "focal"):
            loss = BinaryClassificationLoss(kind=kind)(logits, targets)
            self.assertTrue(torch.isfinite(loss))

    def test_stack_model_logits_resizes_each_model_input(self) -> None:
        models = [RecordingModel(1.0), RecordingModel(2.0)]
        x = torch.zeros(2, 3, 80, 80)

        logits = stack_model_logits(models, [32, 48], x)

        self.assertEqual(tuple(logits.shape), (2, 2))
        self.assertEqual(models[0].seen_shapes, [(32, 32)])
        self.assertEqual(models[1].seen_shapes, [(48, 48)])

    def test_load_models_allows_mixed_image_sizes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            model = build_model(backbone="tiny", pretrained_backbone=False)
            state_dict = model.state_dict()

            ckpt1 = tmp_path / "m1.safetensors"
            ckpt2 = tmp_path / "m2.safetensors"
            save_safetensors_checkpoint(
                ckpt1,
                {"state_dict": state_dict, "img_size": 64, "threshold": 0.4, "temperature": 1.0, "backbone": "tiny"},
            )
            save_safetensors_checkpoint(
                ckpt2,
                {"state_dict": state_dict, "img_size": 96, "threshold": 0.6, "temperature": 1.5, "backbone": "tiny"},
            )

            loaded = load_models([str(ckpt1), str(ckpt2)], torch.device("cpu"))

            self.assertEqual(loaded.img_sizes, [64, 96])
            self.assertEqual(loaded.img_size, 96)

            detector = EnsembleDetector(loaded.models, weights=loaded.weights, img_sizes=loaded.img_sizes)
            out = detector(torch.zeros(1, 3, 96, 96))
            self.assertEqual(tuple(out.shape), (1,))


if __name__ == "__main__":
    unittest.main()
