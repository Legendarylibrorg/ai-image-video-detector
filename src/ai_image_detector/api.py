from __future__ import annotations

import argparse
from collections import defaultdict, deque
import io
import json
import logging
from pathlib import Path
import tempfile
import time
from datetime import datetime, timezone
import hashlib

import torch
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field
from PIL import Image
from torchvision import transforms
import uvicorn

from . import __version__
from .decision import combined_risk, decide_label, image_ood_score
from .domain import classify_domain, load_domain_config, resolve_domain_threshold
from .ensemble import EnsembleDetector, load_models
from .metadata import analyze_metadata
from .multimodal import (
    analyze_audio_bytes,
    analyze_conversation,
    load_fusion_config,
    analyze_pdf_bytes,
    analyze_text_content,
    analyze_url,
    fuse_multimodal_risk,
)
from .provenance import analyze_provenance
from .risk_tools import apply_risk_tools, load_tools_config
from .text_signals import analyze_text_signals


app = FastAPI(title="Advanced AI Image Detector")
_state = {}
logger = logging.getLogger("ai_image_detector.api")


class TextRequest(BaseModel):
    text: str = Field(default="")


class URLRequest(BaseModel):
    url: str = Field(default="")


class MultimodalRequest(BaseModel):
    scores: dict[str, float] = Field(default_factory=dict)


@app.get("/health")
def health():
    return {
        "ok": True,
        "version": __version__,
        "model_ids": _state.get("model_ids", []),
        "ensemble_config": _state.get("ensemble_config") or None,
        "domain_config": _state.get("domain_config") or None,
        "fusion_config": _state.get("fusion_config") or None,
        "tools_config": _state.get("tools_config") or None,
    }


@app.get("/")
def index():
    return {
        "ok": True,
        "service": "ai-image-detector-api",
        "docs_hint": "Use /health and /detect endpoints.",
    }


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


def _log_ip_value(raw_ip: str) -> str:
    mode = str(_state.get("ip_log_mode", "masked")).lower()
    if mode == "none":
        return "redacted"
    if mode == "full":
        return raw_ip
    # masked (default): deterministic hash, not raw IP.
    salt = str(_state.get("ip_log_salt", ""))
    h = hashlib.sha256((salt + raw_ip).encode("utf-8")).hexdigest()
    return f"hash:{h[:12]}"


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
        views = [x]
        tta_views = int(_state.get("tta_views", 1))
        if tta_views >= 2:
            views.append(torch.flip(x, dims=[3]))
        if tta_views >= 3:
            views.append(torch.flip(x, dims=[2]))
        logit = torch.stack([_state["model"](v) for v in views], dim=0).mean(dim=0)
        p = torch.sigmoid(logit / max(_state["temperature"], 1e-6)).item()

    with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
        tmp.write(content)
        tmp.flush()
        meta = analyze_metadata(tmp.name)

    prov = analyze_provenance(content)
    ood = image_ood_score(img)
    text = analyze_text_signals(img)

    metadata_score = float(meta["metadata_score"])
    provenance_score = float(prov["provenance_score"])
    text_score = float(text["text_score"])
    domain = classify_domain(img, text_score=text_score)
    threshold = resolve_domain_threshold(_state["base_threshold"], domain, _state.get("domain_cfg", {}))
    c_risk = combined_risk(p, metadata_score, provenance_score, text_score)
    adjusted = apply_risk_tools(
        prob_ai=p,
        combined_risk=c_risk,
        metadata_flags=meta["metadata_flags"],
        ood_flags=ood["ood_flags"],
        text_flags=text["text_flags"],
        cfg=_state.get("tools_cfg", {}),
    )
    p = float(adjusted["prob_ai"])
    c_risk = float(adjusted["combined_risk"])
    label = decide_label(
        p,
        threshold,
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
        "text_score": text_score,
        "text_flags": text["text_flags"],
        "text_regions": int(text.get("text_regions", 0)),
        "ood_score": float(ood["ood_score"]),
        "ood_flags": ood["ood_flags"],
        "combined_risk": c_risk,
        "threshold": threshold,
        "unknown_margin": _state["unknown_margin"],
        "domain": domain,
        "model_ids": _state["model_ids"],
        "model_count": len(_state["model_ids"]),
        "ensemble_weights": _state.get("ensemble_weights", []),
        "ensemble_config": _state.get("ensemble_config") or None,
        "tool_adjustments": adjusted["tool_adjustments"],
        "tta_views": int(_state.get("tta_views", 1)),
        "model_version": __version__,
    }

    logger.info(
        json.dumps(
            {
                "event": "detect",
                "ip": _log_ip_value(client_ip),
                "label": label,
                "prob_ai": round(p, 6),
                "combined_risk": round(c_risk, 6),
                "ood_score": round(float(ood["ood_score"]), 6),
            }
        )
    )

    # Optional active-learning queue: capture uncertain/high-risk samples for review.
    if _state.get("uncertain_capture", False):
      should_capture = (
          label == "Unknown"
          or abs(float(p) - float(threshold)) <= float(_state.get("uncertain_capture_margin", 0.05))
          or float(c_risk) >= float(_state.get("uncertain_capture_risk", 0.85))
      )
      if should_capture:
          qdir = Path(_state.get("uncertain_dir", "./incoming_review_queue"))
          qdir.mkdir(parents=True, exist_ok=True)
          ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
          stem = f"{ts}_p{p:.3f}_risk{c_risk:.3f}_{label.replace(' ', '_')}"
          img_path = qdir / f"{stem}.jpg"
          meta_path = qdir / f"{stem}.json"
          try:
              img_path.write_bytes(content)
              meta_path.write_text(json.dumps(response, indent=2), encoding="utf-8")
          except Exception:
              pass
    return response


@app.post("/analyze/text")
def analyze_text(req: TextRequest):
    return analyze_text_content(req.text)


@app.post("/analyze/conversation")
def analyze_conversation_text(req: TextRequest):
    return analyze_conversation(req.text)


@app.post("/analyze/url")
def analyze_url_text(req: URLRequest):
    return analyze_url(req.url)


@app.post("/analyze/pdf")
async def analyze_pdf(file: UploadFile = File(...)):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="empty payload")
    return analyze_pdf_bytes(content)


@app.post("/analyze/audio")
async def analyze_audio(file: UploadFile = File(...)):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="empty payload")
    return analyze_audio_bytes(content)


@app.post("/analyze/multimodal")
def analyze_multimodal(req: MultimodalRequest):
    return fuse_multimodal_risk(req.scores, config=_state.get("fusion_cfg", {}))


def load(
    model_paths: list[str],
    max_bytes: int,
    rate_limit_per_min: int,
    unknown_margin: float,
    ensemble_config: str = "",
    domain_config: str = "",
    fusion_config: str = "",
    tools_config: str = "",
    tta_views: int = 1,
    uncertain_capture: bool = False,
    uncertain_dir: str = "./incoming_review_queue",
    uncertain_capture_margin: float = 0.05,
    uncertain_capture_risk: float = 0.85,
    ip_log_mode: str = "masked",
    ip_log_salt: str = "",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    loaded = load_models(model_paths, device, ensemble_config=ensemble_config)
    model = EnsembleDetector(loaded.models, weights=loaded.weights, img_sizes=loaded.img_sizes).to(device)
    model.eval()

    _state["device"] = device
    _state["model"] = model
    _state["base_threshold"] = float(loaded.threshold)
    _state["temperature"] = loaded.temperature
    _state["model_ids"] = loaded.model_ids
    _state["unknown_margin"] = float(unknown_margin)
    _state["ensemble_weights"] = [float(w) for w in loaded.weights]
    _state["ensemble_config"] = ensemble_config
    _state["domain_config"] = domain_config
    _state["domain_cfg"] = load_domain_config(domain_config)
    _state["fusion_config"] = fusion_config
    _state["fusion_cfg"] = load_fusion_config(fusion_config)
    _state["tools_config"] = tools_config
    _state["tools_cfg"] = load_tools_config(tools_config)
    _state["tta_views"] = int(max(1, tta_views))
    _state["uncertain_capture"] = bool(uncertain_capture)
    _state["uncertain_dir"] = str(uncertain_dir)
    _state["uncertain_capture_margin"] = float(uncertain_capture_margin)
    _state["uncertain_capture_risk"] = float(uncertain_capture_risk)
    _state["ip_log_mode"] = str(ip_log_mode)
    _state["ip_log_salt"] = str(ip_log_salt)

    _state["tf"] = transforms.Compose([
        transforms.Resize((loaded.img_size, loaded.img_size)),
        transforms.ToTensor(),
    ])

    _state["max_bytes"] = int(max_bytes)
    _state["allowed_content_types"] = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
    _state["rate_limit_per_min"] = int(rate_limit_per_min)
    _state["rate_window_sec"] = 60
    _state["rate_buckets"] = defaultdict(deque)

    # Warmup pass to reduce first-request latency.
    with torch.no_grad():
        x = torch.zeros(1, 3, loaded.img_size, loaded.img_size, device=device)
        _ = model(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-bytes", type=int, default=10 * 1024 * 1024)
    ap.add_argument("--rate-limit-per-min", type=int, default=60)
    ap.add_argument("--unknown-margin", type=float, default=0.08)
    ap.add_argument("--ensemble-config", default="", help="Optional JSON with learned ensemble weights/threshold")
    ap.add_argument("--domain-config", default="", help="Optional JSON with per-domain thresholds")
    ap.add_argument("--fusion-config", default="", help="Optional JSON with learned multimodal fusion weights")
    ap.add_argument("--tools-config", default="", help="Optional JSON with rule/policy risk adjustments")
    ap.add_argument("--tta-views", type=int, default=1, help="1=none, 2=+hflip, 3=+vflip")
    ap.add_argument("--uncertain-capture", action="store_true", default=False, help="Save uncertain/high-risk samples for review")
    ap.add_argument("--uncertain-dir", default="./incoming_review_queue")
    ap.add_argument("--uncertain-capture-margin", type=float, default=0.05)
    ap.add_argument("--uncertain-capture-risk", type=float, default=0.85)
    ap.add_argument("--ip-log-mode", choices=["none", "masked", "full"], default="masked")
    ap.add_argument("--ip-log-salt", default="", help="Optional salt used when ip-log-mode=masked")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)
    load(
        args.model,
        args.max_bytes,
        args.rate_limit_per_min,
        args.unknown_margin,
        args.ensemble_config,
        args.domain_config,
        args.fusion_config,
        args.tools_config,
        args.tta_views,
        args.uncertain_capture,
        args.uncertain_dir,
        args.uncertain_capture_margin,
        args.uncertain_capture_risk,
        args.ip_log_mode,
        args.ip_log_salt,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
