from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from .decision import combined_risk, decide_label, image_ood_score
from .domain import classify_domain, load_domain_config, resolve_domain_threshold
from .ensemble import EnsembleDetector, load_models
from .metadata import analyze_metadata
from .provenance import analyze_provenance
from .text_signals import analyze_text_signals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", required=True, help="One or more model checkpoints for ensembling")
    ap.add_argument("--ensemble-config", default="", help="Optional JSON with learned ensemble weights/threshold")
    ap.add_argument("--image", required=True)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--domain-config", default="", help="Optional JSON with per-domain thresholds")
    ap.add_argument("--unknown-margin", type=float, default=0.08)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = load_models(args.model, device, ensemble_config=args.ensemble_config)
    model = EnsembleDetector(loaded.models, weights=loaded.weights).to(device)
    model.eval()

    base_threshold = float(args.threshold) if args.threshold is not None else float(loaded.threshold)
    domain_cfg = load_domain_config(args.domain_config)

    tf = transforms.Compose([
        transforms.Resize((loaded.img_size, loaded.img_size)),
        transforms.ToTensor(),
    ])

    image_path = Path(args.image)
    image_bytes = image_path.read_bytes()
    img = Image.open(args.image).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = model(x)
        prob_ai = torch.sigmoid(logit / max(loaded.temperature, 1e-6)).item()

    meta = analyze_metadata(args.image)
    prov = analyze_provenance(image_bytes)
    ood = image_ood_score(img)
    text = analyze_text_signals(img)

    metadata_score = float(meta["metadata_score"])
    provenance_score = float(prov["provenance_score"])
    text_score = float(text["text_score"])
    domain = classify_domain(img, text_score=text_score)
    threshold = resolve_domain_threshold(base_threshold, domain, domain_cfg)
    c_risk = combined_risk(prob_ai, metadata_score, provenance_score, text_score)
    label = decide_label(prob_ai, threshold, args.unknown_margin, float(ood["ood_score"]))

    out = {
        "label": label,
        "prob_ai": float(prob_ai),
        "threshold": threshold,
        "unknown_margin": float(args.unknown_margin),
        "combined_risk": c_risk,
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
        "model_ids": loaded.model_ids,
        "model_count": len(loaded.model_ids),
        "temperature": float(loaded.temperature),
        "ensemble_weights": [float(w) for w in loaded.weights],
        "ensemble_config": args.ensemble_config or None,
        "domain": domain,
        "domain_config": args.domain_config or None,
    }

    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print(
            "label={} prob_ai={:.6f} combined_risk={:.6f} threshold={:.3f} ood_score={:.3f}".format(
                out["label"],
                out["prob_ai"],
                out["combined_risk"],
                out["threshold"],
                out["ood_score"],
            )
        )
        print(f"metadata_flags={out['metadata_flags']}")
        print(f"provenance_flags={out['provenance_flags']}")
        print(f"ood_flags={out['ood_flags']}")


if __name__ == "__main__":
    main()
