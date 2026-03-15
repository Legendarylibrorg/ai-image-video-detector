from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .ensemble import EnsembleDetector, load_models


def saliency_map(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    x = x.clone().detach().requires_grad_(True)
    logit = model(x)
    score = logit.squeeze(0)
    score.backward()
    grad = x.grad.detach().abs().max(dim=1).values.squeeze(0)
    grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-8)
    return grad


def overlay_heatmap(image: Image.Image, heat: np.ndarray, alpha: float = 0.45) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    h = np.clip(heat, 0.0, 1.0)
    # Red-yellow map.
    cmap = np.stack([h, np.sqrt(h), np.zeros_like(h)], axis=-1)
    out = (1.0 - alpha) * base + alpha * cmap
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def patch_scores(heat: np.ndarray, grid: int = 8) -> list[tuple[int, int, float]]:
    hh, ww = heat.shape
    y_step = max(hh // grid, 1)
    x_step = max(ww // grid, 1)
    scores: list[tuple[int, int, float]] = []
    for gy in range(grid):
        for gx in range(grid):
            y0, y1 = gy * y_step, min((gy + 1) * y_step, hh)
            x0, x1 = gx * x_step, min((gx + 1) * x_step, ww)
            if y1 <= y0 or x1 <= x0:
                continue
            s = float(np.mean(heat[y0:y1, x0:x1]))
            scores.append((gy, gx, s))
    scores.sort(key=lambda t: t[2], reverse=True)
    return scores


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate saliency heatmap and patch-level risk")
    ap.add_argument("--model", nargs="+", required=True)
    ap.add_argument("--ensemble-config", default="", help="Optional JSON with learned ensemble weights/threshold")
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="./artifacts/explain")
    ap.add_argument("--grid", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=8)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = load_models(args.model, device, ensemble_config=args.ensemble_config)
    model = EnsembleDetector(loaded.models, weights=loaded.weights).to(device)
    model.eval()

    img = Image.open(args.image).convert("RGB")
    tf = transforms.Compose([
        transforms.Resize((loaded.img_size, loaded.img_size)),
        transforms.ToTensor(),
    ])
    x = tf(img).unsqueeze(0).to(device)

    heat = saliency_map(model, x).cpu().numpy()
    viz = overlay_heatmap(img.resize((loaded.img_size, loaded.img_size)), heat)

    heat_path = out_dir / "heatmap_overlay.jpg"
    viz.save(heat_path, quality=95)

    scores = patch_scores(heat, grid=args.grid)[: args.top_k]
    print(f"saved_heatmap={heat_path}")
    for gy, gx, s in scores:
        print(f"patch=(row={gy}, col={gx}) score={s:.4f}")


if __name__ == "__main__":
    main()
