from __future__ import annotations

import math
from typing import Any

import numpy as np


def _to_np(x) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        return arr.reshape(-1)
    return arr


def sigmoid(x):
    x = _to_np(x)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def binary_metrics(probs, labels, threshold: float = 0.5) -> dict[str, float]:
    p = _to_np(probs)
    y = _to_np(labels).astype(np.int64)
    pred = (p >= threshold).astype(np.int64)

    tp = int(np.sum((pred == 1) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))

    total = max(len(y), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    acc = (tp + tn) / total
    fpr = fp / max(fp + tn, 1)
    tpr = recall

    return {
        "threshold": float(threshold),
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "tpr": float(tpr),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def brier_score(probs, labels) -> float:
    p = _to_np(probs)
    y = _to_np(labels)
    return float(np.mean((p - y) ** 2))


def ece_score(probs, labels, n_bins: int = 15) -> float:
    p = _to_np(probs)
    y = _to_np(labels)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p >= lo) & (p < hi if i < n_bins - 1 else p <= hi)
        if not np.any(mask):
            continue
        conf = float(np.mean(p[mask]))
        acc = float(np.mean(y[mask]))
        frac = float(np.mean(mask))
        ece += abs(conf - acc) * frac
    return float(ece)


def roc_auc(probs, labels) -> float:
    p = _to_np(probs)
    y = _to_np(labels).astype(np.int64)
    pos = p[y == 1]
    neg = p[y == 0]
    n_pos = len(pos)
    n_neg = len(neg)
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1, dtype=np.float64)
    sum_ranks_pos = float(np.sum(ranks[y == 1]))
    auc = (sum_ranks_pos - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)
    return float(auc)


def _threshold_objective_score(metrics: dict[str, float], objective: str) -> float:
    if objective == "balanced":
        return float(0.5 * (metrics["tpr"] + (1.0 - metrics["fpr"])))
    if objective == "youden":
        return float(metrics["tpr"] - metrics["fpr"])
    return float(metrics["f1"])


def _threshold_is_operable(metrics: dict[str, float]) -> bool:
    return float(metrics.get("tp", 0.0)) > 0.0 and float(metrics.get("tn", 0.0)) > 0.0


def find_best_threshold(probs, labels, objective: str = "f1") -> tuple[float, float, dict[str, float]]:
    p = _to_np(probs)
    y = _to_np(labels)
    fallback_threshold = 0.5
    fallback_metrics = binary_metrics(p, y, threshold=float(fallback_threshold))
    fallback_metrics["operable"] = _threshold_is_operable(fallback_metrics)
    fallback_metrics["search_status"] = "fallback_no_operable_threshold"
    best_th = float(fallback_threshold)
    best_score = -1.0
    best_metrics: dict[str, float] = {}
    best_distance = float("inf")

    for th in np.linspace(0.05, 0.95, 91):
        m = binary_metrics(p, y, threshold=float(th))
        if not _threshold_is_operable(m):
            continue
        score = _threshold_objective_score(m, objective)
        distance = abs(float(th) - 0.5)
        if score > best_score or (abs(score - best_score) <= 1e-12 and distance < best_distance):
            best_score = float(score)
            best_th = float(th)
            best_metrics = m
            best_distance = float(distance)

    if not best_metrics:
        fallback_score = _threshold_objective_score(fallback_metrics, objective)
        return float(fallback_threshold), float(fallback_score), fallback_metrics

    best_metrics["operable"] = True
    best_metrics["search_status"] = "operable"
    return best_th, best_score, best_metrics


def _nll_from_logits(logits, labels, temperature: float) -> float:
    z = _to_np(logits) / max(temperature, 1e-6)
    y = _to_np(labels)
    p = sigmoid(z)
    eps = 1e-8
    nll = -(y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps))
    return float(np.mean(nll))


def fit_temperature(logits, labels) -> tuple[float, float]:
    best_t = 1.0
    best_nll = math.inf
    for t in np.linspace(0.5, 3.0, 101):
        nll = _nll_from_logits(logits, labels, float(t))
        if nll < best_nll:
            best_t = float(t)
            best_nll = float(nll)
    return best_t, best_nll


def full_metric_report(probs, labels, threshold: float) -> dict[str, Any]:
    base = binary_metrics(probs, labels, threshold=threshold)
    base["auc"] = roc_auc(probs, labels)
    base["ece"] = ece_score(probs, labels)
    base["brier"] = brier_score(probs, labels)
    return base
