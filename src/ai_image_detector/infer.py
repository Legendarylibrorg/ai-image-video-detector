from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .data import make_eval_transform
from .io_limits import configure_pil_limits, open_image_rgb, read_bytes_limited, reject_symlink
from .decision import image_ood_score
from .domain import classify_domain, load_domain_config, resolve_domain_threshold
from .ensemble import EnsembleDetector, load_models
from .inference_report import ConfigReport, DecisionOptions, ModelReport, build_inference_report
from .metadata import analyze_metadata, extract_metadata_features
from .provenance import analyze_provenance
from .risk_tools import load_tools_config
from .runtime import training_device
from .text_signals import analyze_text_signals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", required=True, help="One or more model checkpoints for ensembling")
    ap.add_argument("--ensemble-config", default="", help="Optional JSON with learned ensemble weights/threshold")
    ap.add_argument("--image", required=True)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--domain-config", default="", help="Optional JSON with per-domain thresholds")
    ap.add_argument("--tools-config", default="", help="Optional JSON for rule/policy risk adjustments")
    ap.add_argument("--tta-views", type=int, default=1, help="1=none, 2=+hflip, 3=+vflip")
    ap.add_argument("--unknown-margin", type=float, default=0.04)
    ap.add_argument("--unknown-margin-ai", type=float, default=0.03)
    ap.add_argument("--unknown-margin-real", type=float, default=0.05)
    ap.add_argument("--borderline-ood-threshold", type=float, default=0.45)
    ap.add_argument("--hard-ood-threshold", type=float, default=0.80)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    configure_pil_limits()
    device = training_device()
    loaded = load_models(args.model, device, ensemble_config=args.ensemble_config)
    model = EnsembleDetector(loaded.models, weights=loaded.weights, img_sizes=loaded.img_sizes).to(device)
    model.eval()

    base_threshold = float(args.threshold) if args.threshold is not None else float(loaded.threshold)
    domain_cfg = load_domain_config(args.domain_config)
    tools_cfg = load_tools_config(args.tools_config)

    tf = make_eval_transform(loaded.img_size)

    image_path = Path(args.image)
    reject_symlink(image_path)
    image_bytes = read_bytes_limited(image_path)
    img = open_image_rgb(image_path, allow_symlink=False)
    x = tf(img).unsqueeze(0).to(device)
    metadata_features = torch.tensor([extract_metadata_features(args.image)], dtype=x.dtype, device=device)

    with torch.no_grad():
        views = [x]
        if args.tta_views >= 2:
            views.append(torch.flip(x, dims=[3]))
        if args.tta_views >= 3:
            views.append(torch.flip(x, dims=[2]))
        logit = torch.stack([model(v, metadata_features=metadata_features) for v in views], dim=0).mean(dim=0)
        prob_ai = torch.sigmoid(logit / max(loaded.temperature, 1e-6)).item()

    meta = analyze_metadata(args.image)
    prov = analyze_provenance(image_bytes)
    ood = image_ood_score(img)
    text = analyze_text_signals(img)

    domain = classify_domain(img, text_score=float(text["text_score"]))
    threshold = resolve_domain_threshold(base_threshold, domain, domain_cfg)
    out = build_inference_report(
        prob_ai=prob_ai,
        threshold=threshold,
        metadata=meta,
        provenance=prov,
        text=text,
        ood=ood,
        domain=domain,
        decision=DecisionOptions(
            unknown_margin=float(args.unknown_margin),
            unknown_margin_ai=float(args.unknown_margin_ai),
            unknown_margin_real=float(args.unknown_margin_real),
            borderline_ood_threshold=float(args.borderline_ood_threshold),
            hard_ood_threshold=float(args.hard_ood_threshold),
            tta_views=int(args.tta_views),
        ),
        model=ModelReport(
            model_ids=loaded.model_ids,
            weights=[float(w) for w in loaded.weights],
            temperature=float(loaded.temperature),
            ensemble_config=args.ensemble_config,
        ),
        config=ConfigReport(domain_config=args.domain_config, tools_config=args.tools_config),
        tools_cfg=tools_cfg,
    )

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
