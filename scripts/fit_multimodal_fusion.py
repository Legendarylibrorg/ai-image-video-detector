from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


FEATURES = ["image", "video", "text", "conversation", "url", "pdf", "audio"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit learned multimodal fusion weights from CSV outcomes")
    ap.add_argument("--csv", required=True, help="CSV with label column and modality score columns")
    ap.add_argument("--label-col", default="label", help="Binary label column (1=risky, 0=safe)")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--lr", type=float, default=0.08)
    ap.add_argument("--l2", type=float, default=0.001)
    ap.add_argument("--out", default="./artifacts_ens/fusion_config.json")
    args = ap.parse_args()

    rows = []
    with Path(args.csv).open("r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows.append(r)
    if not rows:
        raise RuntimeError("No rows found in CSV")

    X = []
    y = []
    for r in rows:
        try:
            yi = float(r.get(args.label_col, "0"))
        except Exception:
            continue
        feats = []
        for k in FEATURES:
            try:
                feats.append(float(r.get(k, "0") or 0))
            except Exception:
                feats.append(0.0)
        X.append(feats)
        y.append(1.0 if yi >= 0.5 else 0.0)
    Xn = np.asarray(X, dtype=np.float64)
    yn = np.asarray(y, dtype=np.float64)
    if Xn.size == 0:
        raise RuntimeError("No usable rows after parsing")

    w = np.zeros(Xn.shape[1], dtype=np.float64)
    b = 0.0
    n = float(len(yn))

    for _ in range(max(1, args.steps)):
        z = Xn @ w + b
        p = _sigmoid(z)
        err = p - yn
        gw = (Xn.T @ err) / n + args.l2 * w
        gb = float(np.mean(err))
        w -= args.lr * gw
        b -= args.lr * gb

    pred = _sigmoid(Xn @ w + b)
    eps = 1e-8
    loss = float(np.mean(-(yn * np.log(pred + eps) + (1 - yn) * np.log(1 - pred + eps))))
    acc = float(np.mean((pred >= 0.5) == (yn >= 0.5)))

    out = {
        "weights": {k: float(v) for k, v in zip(FEATURES, w.tolist())},
        "bias": float(b),
        "fit": {"loss": loss, "acc": acc, "n": int(len(yn)), "steps": int(args.steps), "lr": float(args.lr), "l2": float(args.l2)},
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()
