"""Framework-neutral diagnostic plots used by every tutorial benchmark."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import matplotlib.pyplot as plt

from .metrics import reliability, ranking_curve


def _square(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.18)


def agreement_plot(
    reference,
    approximation,
    *,
    xlabel: str = "reference",
    ylabel: str = "approximation",
    title: str = "Moment agreement",
    ax=None,
):
    """All coordinates, common axis limits, and the identity line."""
    reference = np.asarray(reference).reshape(-1)
    approximation = np.asarray(approximation).reshape(-1)
    if reference.shape != approximation.shape:
        raise ValueError("reference and approximation must have the same shape")
    if ax is None:
        _, ax = plt.subplots(figsize=(4.2, 4.2))
    limits = np.array([reference.min(initial=0), reference.max(initial=0),
                       approximation.min(initial=0), approximation.max(initial=0)])
    lo, hi = limits.min(), limits.max()
    margin = 0.04 * max(hi - lo, 1e-12)
    ax.scatter(reference, approximation, s=10, alpha=0.55)
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin], color="black", lw=1)
    ax.set_xlim(lo - margin, hi + margin)
    ax.set_ylim(lo - margin, hi + margin)
    _square(ax, xlabel, ylabel, title)
    return ax


def calibration_plot(probability, labels, bins: int = 15, *, ax=None, label=None):
    """Reliability curve with bin mass shown by marker area."""
    r = reliability(probability, labels, bins)
    if ax is None:
        _, ax = plt.subplots(figsize=(4.4, 4.1))
    ax.plot([0, 1], [0, 1], color="black", lw=1, ls="--")
    valid = r["count"] > 0
    mass = r["count"] / max(r["count"].sum(), 1)
    ax.plot(r["confidence"][valid], r["accuracy"][valid], marker="o", label=label)
    ax.scatter(
        r["confidence"][valid], r["accuracy"][valid], s=30 + 400 * mass[valid], alpha=0.7
    )
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="mean confidence", ylabel="accuracy",
           title="Reliability")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.18)
    if label:
        ax.legend(frameon=False)
    return ax


def risk_coverage_plot(probability, labels, *, ax=None, label=None):
    """Selective risk in descending confidence order, matching metric AURC."""
    probability = np.asarray(probability)
    labels = np.asarray(labels)
    error = probability.argmax(1) != labels
    order = np.argsort(-probability.max(1), kind="stable")
    coverage = np.arange(1, len(error) + 1) / len(error)
    risk = np.cumsum(error[order]) / np.arange(1, len(error) + 1)
    if ax is None:
        _, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.plot(coverage, risk, label=label)
    ax.set(xlabel="coverage", ylabel="selective error", title="Risk-coverage")
    ax.grid(alpha=0.18)
    if label:
        ax.legend(frameon=False)
    return ax


def ood_roc_plot(id_score, ood_score, *, ax=None, label=None):
    """OOD ROC where larger scores denote more uncertainty."""
    score = np.r_[np.asarray(id_score), np.asarray(ood_score)]
    truth = np.r_[np.zeros(len(id_score), dtype=int), np.ones(len(ood_score), dtype=int)]
    curve = ranking_curve(score, truth)
    if ax is None:
        _, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.plot(curve.fpr, curve.tpr, label=label)
    ax.plot([0, 1], [0, 1], color="black", lw=1, ls="--")
    ax.set(xlabel="false-positive rate", ylabel="true-positive rate", title="OOD ROC")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.18)
    if label:
        ax.legend(frameon=False)
    return ax


def uncertainty_panel(
    probability,
    labels,
    *,
    id_score=None,
    ood_score=None,
    method: str = "model",
):
    """The common calibration/selective/OOD figure used in the tutorial."""
    columns = 3 if id_score is not None and ood_score is not None else 2
    figure, axes = plt.subplots(1, columns, figsize=(4.4 * columns, 4.1))
    calibration_plot(probability, labels, ax=axes[0], label=method)
    risk_coverage_plot(probability, labels, ax=axes[1], label=method)
    if columns == 3:
        ood_roc_plot(id_score, ood_score, ax=axes[2], label=method)
    figure.tight_layout()
    return figure, axes


def learning_curves(
    records: Mapping[str, Sequence[float]],
    *,
    x: str = "epoch",
    metrics: tuple[str, str] = ("validation_nll", "validation_accuracy"),
):
    """Two-panel curves without extrapolating stopped runs."""
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.7))
    coordinate = np.asarray(records[x])
    for ax, metric in zip(axes, metrics):
        ax.plot(coordinate, records[metric])
        ax.set(xlabel=x.replace("_", " "), ylabel=metric.replace("_", " "),
               title=metric.replace("_", " ").title())
        ax.grid(alpha=0.18)
    figure.tight_layout()
    return figure, axes


__all__ = [
    "agreement_plot",
    "calibration_plot",
    "risk_coverage_plot",
    "ood_roc_plot",
    "uncertainty_panel",
    "learning_curves",
]
