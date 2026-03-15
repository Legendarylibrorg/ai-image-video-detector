from __future__ import annotations

import argparse

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .decision import decide_label
from .ensemble import EnsembleDetector, load_models


def _iter_frames(path: str, every: int, max_frames: int):
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError("opencv-python-headless is required for video inference") from exc

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"unable to open video: {path}")

    idx = 0
    yielded = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % max(every, 1) == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield Image.fromarray(rgb)
            yielded += 1
            if yielded >= max_frames:
                break
        idx += 1
    cap.release()


def main() -> None:
    ap = argparse.ArgumentParser(description="Video AI-image style detector")
    ap.add_argument("--model", nargs="+", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--sample-every", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=48)
    ap.add_argument("--unknown-margin", type=float, default=0.08)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = load_models(args.model, device)
    model = EnsembleDetector(loaded.models).to(device)
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((loaded.img_size, loaded.img_size)),
        transforms.ToTensor(),
    ])

    probs = []
    for frame in _iter_frames(args.video, args.sample_every, args.max_frames):
        x = tf(frame).unsqueeze(0).to(device)
        with torch.no_grad():
            p = torch.sigmoid(model(x) / loaded.temperature).item()
        probs.append(p)

    if not probs:
        raise RuntimeError("no frames sampled from video")

    mean_p = float(np.mean(probs))
    max_p = float(np.max(probs))
    label = decide_label(mean_p, loaded.threshold, args.unknown_margin, ood_score=0.0)

    print(
        f"label={label} prob_ai_mean={mean_p:.6f} prob_ai_max={max_p:.6f} "
        f"frames_used={len(probs)} threshold={loaded.threshold:.3f}"
    )


if __name__ == "__main__":
    main()
