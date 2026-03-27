from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from _support import ROOT  # noqa: F401
from ai_image_detector import video_temporal


class _FakeEfficientNet:
    def __init__(self) -> None:
        self.features = torch.nn.Identity()


class _FakeTemporalModel:
    def __init__(self, pretrained_backbone: bool = True) -> None:
        self.pretrained_backbone = pretrained_backbone
        self.loaded = None

    def to(self, device):
        return self

    def load_state_dict(self, state_dict):
        self.loaded = state_dict

    def eval(self):
        return self

    def __call__(self, x):
        return torch.zeros((x.shape[0],), dtype=torch.float32)


class VideoTemporalTests(unittest.TestCase):
    def test_collect_videos_includes_m4v_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clip = root / "ai" / "clip.m4v"
            clip.parent.mkdir(parents=True)
            clip.write_bytes(b"x")

            samples = video_temporal._collect_videos(root)

            self.assertEqual(samples, [(str(clip), 1)])

    def test_temporal_video_detector_does_not_request_pretrained_weights_by_default(self) -> None:
        captured: list[object] = []

        def fake_effnet(*, weights=None):
            captured.append(weights)
            return _FakeEfficientNet()

        with mock.patch.object(video_temporal.models, "efficientnet_b0", side_effect=fake_effnet):
            video_temporal.TemporalVideoDetector(pretrained_backbone=False)

        self.assertEqual(captured, [None])

    def test_infer_main_uses_non_pretrained_backbone(self) -> None:
        fake_model = _FakeTemporalModel(pretrained_backbone=False)

        with mock.patch("sys.argv", ["prog", "--model", "fake.safetensors", "--video", "fake.mp4"]), \
            mock.patch.object(video_temporal, "load_checkpoint", return_value={"state_dict": {}, "img_size": 16, "frames": 2, "threshold": 0.5}), \
            mock.patch.object(video_temporal, "_sample_frames", return_value=np.zeros((2, 3, 16, 16), dtype=np.float32)), \
            mock.patch.object(video_temporal, "TemporalVideoDetector", side_effect=lambda pretrained_backbone=False: _FakeTemporalModel(pretrained_backbone=pretrained_backbone)) as ctor, \
            mock.patch.object(video_temporal.torch.cuda, "is_available", return_value=False):
            with redirect_stdout(io.StringIO()):
                video_temporal.infer_main()

        self.assertEqual(ctor.call_args.kwargs, {"pretrained_backbone": False})


if __name__ == "__main__":
    unittest.main()
