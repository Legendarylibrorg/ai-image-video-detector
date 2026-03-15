from __future__ import annotations

import math
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlparse
import wave
import numpy as np


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def analyze_text_content(text: str) -> dict[str, Any]:
    t = (text or "").strip()
    if not t:
        return {"risk_score": 0.0, "flags": ["empty_text"], "tokens": 0}

    lower = t.lower()
    flags: list[str] = []
    risk = 0.0

    scam_terms = [
        "urgent",
        "verify account",
        "click here",
        "crypto",
        "wire transfer",
        "gift card",
        "password reset",
        "seed phrase",
        "guaranteed return",
    ]
    hits = sum(1 for s in scam_terms if s in lower)
    if hits > 0:
        flags.append("suspicious_phrasing")
        risk += min(0.45, 0.08 * hits)

    upper_ratio = sum(c.isupper() for c in t) / max(1, sum(c.isalpha() for c in t))
    if upper_ratio > 0.45:
        flags.append("excessive_caps")
        risk += 0.12

    punct_ratio = sum(c in "!?$" for c in t) / max(1, len(t))
    if punct_ratio > 0.06:
        flags.append("aggressive_punctuation")
        risk += 0.1

    urls = re.findall(r"https?://\S+|www\.\S+", lower)
    if urls:
        flags.append("contains_links")
        risk += min(0.2, 0.04 * len(urls))

    return {"risk_score": _clip01(risk), "flags": flags, "tokens": len(t.split())}


def analyze_conversation(text: str) -> dict[str, Any]:
    base = analyze_text_content(text)
    lower = (text or "").lower()
    flags = list(base["flags"])
    risk = float(base["risk_score"])

    grooming_markers = ["keep this secret", "don't tell", "private chat", "send pics", "move to telegram"]
    gm = sum(1 for g in grooming_markers if g in lower)
    if gm:
        flags.append("grooming_or_coercion_markers")
        risk += min(0.35, 0.1 * gm)

    harassment_markers = ["idiot", "kill yourself", "hate you", "worthless"]
    hm = sum(1 for h in harassment_markers if h in lower)
    if hm:
        flags.append("harassment_markers")
        risk += min(0.25, 0.08 * hm)

    return {"risk_score": _clip01(risk), "flags": flags, "tokens": base["tokens"]}


def analyze_url(url: str) -> dict[str, Any]:
    raw = (url or "").strip()
    if not raw:
        return {"risk_score": 0.0, "flags": ["empty_url"], "domain": ""}
    if "://" not in raw:
        raw = "https://" + raw

    flags: list[str] = []
    risk = 0.0
    parsed = urlparse(raw)
    host = (parsed.hostname or "").lower()

    if parsed.scheme not in {"https"}:
        flags.append("non_https")
        risk += 0.2

    if re.match(r"^\d+\.\d+\.\d+\.\d+$", host):
        flags.append("ip_literal_host")
        risk += 0.35

    if host.count("-") >= 3 or len(host) > 45:
        flags.append("suspicious_domain_shape")
        risk += 0.18

    risky_tlds = {".zip", ".top", ".click", ".gq", ".work"}
    if any(host.endswith(tld) for tld in risky_tlds):
        flags.append("risky_tld")
        risk += 0.22

    suspicious_terms = ["secure", "verify", "wallet", "recovery", "auth", "gift", "prize", "login"]
    if any(s in host for s in suspicious_terms):
        flags.append("social_engineering_domain_terms")
        risk += 0.12

    rep = load_url_reputation()
    deny = {str(x).lower() for x in rep.get("deny_domains", [])}
    warn = {str(x).lower() for x in rep.get("warn_domains", [])}
    if host in deny:
        flags.append("domain_reputation_deny")
        risk += 0.55
    elif host in warn:
        flags.append("domain_reputation_warn")
        risk += 0.25

    return {"risk_score": _clip01(risk), "flags": flags, "domain": host}


def analyze_pdf_bytes(pdf_bytes: bytes) -> dict[str, Any]:
    b = pdf_bytes or b""
    flags: list[str] = []
    risk = 0.0

    if not b.startswith(b"%PDF"):
        flags.append("invalid_pdf_header")
        risk += 0.7

    size_mb = len(b) / (1024 * 1024)
    if size_mb > 20:
        flags.append("large_pdf")
        risk += 0.1

    lower = b.lower()
    if b"/javascript" in lower or b"/js" in lower:
        flags.append("embedded_javascript")
        risk += 0.35
    if b"/launch" in lower:
        flags.append("launch_action")
        risk += 0.35
    if b"/openaction" in lower:
        flags.append("open_action")
        risk += 0.2

    return {"risk_score": _clip01(risk), "flags": flags, "size_bytes": len(b)}


def analyze_audio_bytes(audio_bytes: bytes) -> dict[str, Any]:
    flags: list[str] = []
    risk = 0.0
    n = len(audio_bytes or b"")
    if n == 0:
        return {"risk_score": 0.0, "flags": ["empty_audio"], "duration_sec": 0.0}

    duration_sec = 0.0
    fr = 0
    raw: np.ndarray | None = None
    try:
        import io

        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            fr = wf.getframerate()
            nf = wf.getnframes()
            duration_sec = float(nf) / max(1, fr)
            if wf.getnchannels() == 1:
                flags.append("mono_audio")
            if fr < 16000:
                flags.append("low_sample_rate")
                risk += 0.08
            sampwidth = wf.getsampwidth()
            ch = wf.getnchannels()
            frames = wf.readframes(nf)
            if sampwidth == 2:
                arr = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            elif sampwidth == 1:
                arr = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
            else:
                arr = np.array([], dtype=np.float32)
            if ch > 1 and arr.size > 0:
                arr = arr.reshape(-1, ch).mean(axis=1)
            raw = arr if arr.size > 0 else None
    except Exception:
        flags.append("non_wav_or_unreadable")
        risk += 0.15

    if duration_sec > 0 and duration_sec < 0.5:
        flags.append("very_short_clip")
        risk += 0.12
    if duration_sec > 300:
        flags.append("very_long_clip")
        risk += 0.08

    if raw is not None and raw.size > 1024 and fr > 0:
        zcr = float(np.mean(np.abs(np.diff(np.sign(raw)))) / 2.0)
        energy = float(np.mean(raw * raw))
        spec = np.abs(np.fft.rfft(raw[: min(len(raw), fr * 20)]))
        spec_sum = float(np.sum(spec) + 1e-8)
        spec_norm = spec / spec_sum
        flatness = float(np.exp(np.mean(np.log(spec + 1e-8))) / (np.mean(spec) + 1e-8))
        entropy = float(-np.sum(spec_norm * np.log(spec_norm + 1e-12)))

        if zcr < 0.01:
            flags.append("low_variation_waveform")
            risk += 0.12
        if flatness < 0.015:
            flags.append("narrowband_tone_like")
            risk += 0.12
        if entropy < 4.5:
            flags.append("low_spectral_entropy")
            risk += 0.08
        if energy < 1e-5:
            flags.append("near_silence")
            risk += 0.15

    return {"risk_score": _clip01(risk), "flags": flags, "duration_sec": float(duration_sec), "size_bytes": n}


def load_fusion_config(path: str = "") -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def load_url_reputation(path: str = "") -> dict[str, Any]:
    if not path:
        path = "./.local/url_reputation.json"
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def fuse_multimodal_risk(scores: dict[str, float], config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = config or {}
    learned_weights = cfg.get("weights", {})
    bias = float(cfg.get("bias", 0.0))

    weights = {
        "image": 0.24,
        "video": 0.16,
        "text": 0.13,
        "conversation": 0.11,
        "pdf": 0.11,
        "audio": 0.11,
        "url": 0.09,
    }
    if isinstance(learned_weights, dict):
        for k, v in learned_weights.items():
            try:
                weights[str(k)] = float(v)
            except Exception:
                continue

    weighted_sum = 0.0
    used_w = 0.0
    for k, w in weights.items():
        if k in scores:
            v = _clip01(float(scores[k]))
            weighted_sum += w * v
            used_w += w
    if used_w <= 0:
        fused = 0.0
    else:
        lin = (weighted_sum / used_w) + bias
        fused = 1.0 / (1.0 + math.exp(-max(-12.0, min(12.0, lin))))

    if fused >= 0.8:
        label = "high_risk"
    elif fused >= 0.5:
        label = "review"
    else:
        label = "low_risk"

    return {"multimodal_risk": _clip01(fused), "label": label, "inputs_used": sorted(scores.keys()), "fusion_mode": "learned" if cfg else "default"}
