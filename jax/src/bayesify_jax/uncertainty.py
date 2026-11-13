"""Predictive-uncertainty metrics operating on class probabilities.

The input is a matrix of probabilities, not logits and not an average loss.
Ties in ranking metrics are consumed as complete groups, so an arbitrary input
order cannot make AUROC or AUPRC look better.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Ranking:
    auroc: float
    auprc: float
    fpr95: float
    fpr: FloatArray
    tpr: FloatArray
    precision: FloatArray
    recall: FloatArray


def _probabilities(probability: ArrayLike) -> FloatArray:
    p = np.asarray(probability, dtype=np.float64)
    if p.ndim != 2 or not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError("probabilities must be a finite nonnegative matrix")
    total = p.sum(1, keepdims=True)
    if np.any(total <= 0) or np.max(np.abs(total - 1)) >= 1e-4:
        raise ValueError("probability rows must sum to one")
    return p / total


def reliability(probability: ArrayLike, labels: ArrayLike, bins: int = 15) -> dict[str, FloatArray]:
    """Return count, confidence and accuracy in equal-width confidence bins."""
    p = _probabilities(probability)
    y = np.asarray(labels, dtype=np.int64)
    confidence, prediction = p.max(1), p.argmax(1)
    group = np.minimum((confidence * bins).astype(int), bins - 1)
    count = np.bincount(group, minlength=bins)
    mean_confidence = np.full(bins, np.nan)
    mean_accuracy = np.full(bins, np.nan)
    for i in range(bins):
        take = group == i
        if take.any():
            mean_confidence[i] = confidence[take].mean()
            mean_accuracy[i] = (prediction[take] == y[take]).mean()
    return {"count": count, "confidence": mean_confidence, "accuracy": mean_accuracy}


def ranking_curve(score: ArrayLike, positive: ArrayLike) -> Ranking:
    """Tie-safe binary ranking curve, AUROC, average precision and FPR@95TPR."""
    score = np.asarray(score, dtype=np.float64)
    positive = np.asarray(positive, dtype=bool)
    order = np.argsort(-score, kind="stable")
    score, positive = score[order], positive[order]
    # Evaluate only after a complete tied-score group.
    end = np.r_[np.flatnonzero(np.diff(score) != 0), len(score) - 1]
    cumulative_positive = np.cumsum(positive)[end]
    selected = end + 1
    cumulative_negative = selected - cumulative_positive
    positives, negatives = positive.sum(), (~positive).sum()
    if positives == 0 or negatives == 0:
        nan = np.asarray([np.nan])
        return Ranking(np.nan, np.nan, np.nan, nan, nan, nan, nan)
    recall = cumulative_positive / positives
    fpr = cumulative_negative / negatives
    precision = cumulative_positive / selected
    tpr0, fpr0 = np.r_[0.0, recall], np.r_[0.0, fpr]
    # Trapezoidal integral written directly.  This is the mathematical rule
    # and works in both NumPy 1.x (which lacks ``trapezoid``) and NumPy 2.x
    # (which removed ``trapz``).
    auroc = np.sum(np.diff(fpr0) * (tpr0[:-1] + tpr0[1:]) / 2)
    auprc = np.sum(np.diff(np.r_[0.0, recall]) * precision)
    fpr95 = np.min(fpr0[tpr0 >= 0.95])
    return Ranking(float(auroc), float(auprc), float(fpr95), fpr0, tpr0, precision, recall)


def uncertainty_metrics(probability: ArrayLike, labels: ArrayLike, bins: int = 15) -> dict[str, float]:
    """Accuracy, proper scores, calibration and selective-prediction metrics."""
    p = _probabilities(probability)
    y = np.asarray(labels, dtype=np.int64)
    if y.shape != (len(p),) or np.any(y < 0) or np.any(y >= p.shape[1]):
        raise ValueError("labels and probability matrix are incompatible")
    n = len(y)
    confidence, prediction = p.max(1), p.argmax(1)
    error = prediction != y
    target = np.eye(p.shape[1], dtype=np.float64)[y]
    table = reliability(p, y, bins)
    ece = np.nansum(table["count"] * np.abs(table["confidence"] - table["accuracy"])) / n
    order = np.argsort(-confidence, kind="stable")
    risk = np.cumsum(error[order]) / np.arange(1, n + 1)
    oracle = np.cumsum(np.sort(error)) / np.arange(1, n + 1)
    detection = ranking_curve(1 - confidence, error)
    at = lambda fraction: float(risk[int(np.ceil(fraction * n)) - 1])
    return {
        "n": float(n),
        "accuracy": float((~error).mean()),
        "nll": float(-np.log(np.maximum(p[np.arange(n), y], 1e-30)).mean()),
        "brier": float(np.square(p - target).sum(1).mean()),
        "ece": float(ece),
        "aurc": float(risk.mean()),
        "excess_aurc": float((risk - oracle).mean()),
        "error_auroc": detection.auroc,
        "error_auprc": detection.auprc,
        "risk80": at(0.80),
        "risk90": at(0.90),
        "risk95": at(0.95),
        "entropy": float(-(p * np.log(np.maximum(p, 1e-30))).sum(1).mean()),
    }


def ood_metrics(id_probability: ArrayLike, ood_probability: ArrayLike) -> dict[str, float]:
    """OOD ranking from entropy and one minus maximum softmax probability."""
    p_id, p_ood = _probabilities(id_probability), _probabilities(ood_probability)
    if p_id.shape[1] != p_ood.shape[1]:
        raise ValueError("ID and OOD predictions must have the same classes")
    entropy = lambda p: -(p * np.log(np.maximum(p, 1e-30))).sum(1)
    positive = np.r_[np.zeros(len(p_id), dtype=bool), np.ones(len(p_ood), dtype=bool)]
    ent = ranking_curve(np.r_[entropy(p_id), entropy(p_ood)], positive)
    msp = ranking_curve(np.r_[1 - p_id.max(1), 1 - p_ood.max(1)], positive)
    return {
        "entropy_auroc": ent.auroc,
        "entropy_auprc": ent.auprc,
        "entropy_fpr95": ent.fpr95,
        "msp_auroc": msp.auroc,
        "msp_auprc": msp.auprc,
        "msp_fpr95": msp.fpr95,
        "id_n": float(len(p_id)),
        "ood_n": float(len(p_ood)),
        "ood_prevalence": float(positive.mean()),
    }
