from __future__ import annotations

import argparse
from collections import defaultdict, deque
import io
import json
import logging
from pathlib import Path
import tempfile
import time

import torch
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torchvision import transforms
import uvicorn

from . import __version__
from .decision import combined_risk, decide_label, image_ood_score
from .ensemble import EnsembleDetector, load_models
from .metadata import analyze_metadata
from .provenance import analyze_provenance


app = FastAPI(title="Advanced AI Image Detector")
_state = {}
logger = logging.getLogger("ai_image_detector.api")
WEB_DIR = Path(__file__).resolve().parent / "web"

if WEB_DIR.exists():
    app.mount("/web", StaticFiles(directory=str(WEB_DIR)), name="web")


@app.get("/health")
def health():
    return {
        "ok": True,
        "version": __version__,
        "model_ids": _state.get("model_ids", []),
    }


@app.get("/")
def index():
    if not WEB_DIR.exists():
        raise HTTPException(status_code=404, detail="frontend not found")
    return FileResponse(str(WEB_DIR / "index.html"))


def _enforce_rate_limit(ip: str) -> None:
    now = time.monotonic()
    window_sec = _state["rate_window_sec"]
    limit = _state["rate_limit_per_min"]

    q = _state["rate_buckets"][ip]
    while q and now - q[0] > window_sec:
        q.popleft()
    if len(q) >= limit:
        raise HTTPException(status_code=429, detail="rate limit exceeded")
    q.append(now)


@app.post("/detect")
async def detect(request: Request, image: UploadFile = File(...)):
    client_ip = request.client.host if request.client else "unknown"
    _enforce_rate_limit(client_ip)

    if image.content_type not in _state["allowed_content_types"]:
        raise HTTPException(status_code=415, detail=f"unsupported content type: {image.content_type}")

    content = await image.read()
    if len(content) > _state["max_bytes"]:
        raise HTTPException(status_code=413, detail="file too large")

    try:
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="invalid image payload") from exc

    tf = _state["tf"]
    x = tf(img).unsqueeze(0).to(_state["device"])

    with torch.no_grad():
        logit = _state["model"](x)
        p = torch.sigmoid(logit / max(_state["temperature"], 1e-6)).item()

    with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
        tmp.write(content)
        tmp.flush()
        meta = analyze_metadata(tmp.name)

    prov = analyze_provenance(content)
    ood = image_ood_score(img)

    metadata_score = float(meta["metadata_score"])
    provenance_score = float(prov["provenance_score"])
    c_risk = combined_risk(p, metadata_score, provenance_score)
    label = decide_label(
        p,
        _state["threshold"],
        _state["unknown_margin"],
        float(ood["ood_score"]),
    )

    response = {
        "label": label,
        "prob_ai": p,
        "metadata_score": metadata_score,
        "metadata_flags": meta["metadata_flags"],
        "metadata_fields": meta["metadata_fields"],
        "provenance_score": provenance_score,
        "provenance_flags": prov["provenance_flags"],
        "ood_score": float(ood["ood_score"]),
        "ood_flags": ood["ood_flags"],
        "combined_risk": c_risk,
        "threshold": _state["threshold"],
        "unknown_margin": _state["unknown_margin"],
        "model_ids": _state["model_ids"],
        "model_count": len(_state["model_ids"]),
        "model_version": __version__,
    }

    logger.info(
        json.dumps(
            {
                "event": "detect",
                "ip": client_ip,
                "label": label,
                "prob_ai": round(p, 6),
                "combined_risk": round(c_risk, 6),
                "ood_score": round(float(ood["ood_score"]), 6),
            }
        )
    )
    return response


def load(model_paths: list[str], max_bytes: int, rate_limit_per_min: int, unknown_margin: float):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = load_models(model_paths, device)
    model = EnsembleDetector(loaded.models).to(device)
    model.eval()

    _state["device"] = device
    _state["model"] = model
    _state["threshold"] = loaded.threshold
    _state["temperature"] = loaded.temperature
    _state["model_ids"] = loaded.model_ids
    _state["unknown_margin"] = float(unknown_margin)

    _state["tf"] = transforms.Compose([
        transforms.Resize((loaded.img_size, loaded.img_size)),
        transforms.ToTensor(),
    ])

    _state["max_bytes"] = int(max_bytes)
    _state["allowed_content_types"] = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
    _state["rate_limit_per_min"] = int(rate_limit_per_min)
    _state["rate_window_sec"] = 60
    _state["rate_buckets"] = defaultdict(deque)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-bytes", type=int, default=10 * 1024 * 1024)
    ap.add_argument("--rate-limit-per-min", type=int, default=60)
    ap.add_argument("--unknown-margin", type=float, default=0.08)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)
    load(args.model, args.max_bytes, args.rate_limit_per_min, args.unknown_margin)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
