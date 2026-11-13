"""Regenerate benchmark diagnostics from original native PyTorch predictions.

No training or checkpoint loading occurs here. The same public metric and
plot functions serve all datasets. Historical times retain their device and
measurement scope; no CPU replay is advertised as GPU benchmark training.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .metrics import uncertainty_metrics, ood_metrics, ranking_curve
from .plotting import calibration_plot, risk_coverage_plot, ood_roc_plot


def load_predictions(directory, family):
    """Read authenticated native arrays with configuration/measurement receipts."""
    directory = Path(directory)
    records = []
    for r in json.loads((directory / "manifest.json").read_text()):
        if r["family"] != family:
            continue
        path = directory / r["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != r["sha256"]:
            raise ValueError(f"changed evidence: {path.name}")
        with np.load(path) as a:
            records.append((r, {k: a[k] for k in a.files}))
    if not records:
        raise ValueError(f"no native predictions for {family}")
    return records


def native_image_predictions(directory):
    """Read completed native runs; ambiguous labels and OOD use different units."""
    records=[]
    labels={'ordinary':'Ordinary','moment_mfvi':'Moment MFVI','moment_ep':'Moment EP',
            'sampled_mfvi':'Sampled MFVI','sampled_ep':'Sampled EP'}
    for path in sorted(Path(directory).glob('*/metrics.json')):
        scores=json.loads(path.read_text());cfg=json.loads((path.parent/'config.json').read_text())
        if scores['status']!='complete_fixed_budget':continue
        with np.load(path.parent/'predictions.npz') as a:
            p,y,p_id=a['p'],a['y'],a['p']
            if 'p_ambiguous' in a:
                pa=a['p_ambiguous'];p_id=np.concatenate((p,pa))
                p=np.concatenate((p,np.repeat(pa,10,axis=0)));y=np.concatenate((y,a['ambiguous_labels'].ravel()))
            arrays=dict(probabilities=p,labels=y,ood_id_probabilities=p_id,ood_probabilities=a['p_ood'])
        record=dict(label=labels[cfg['method']],seed=cfg['seed'],ood='Fashion-MNIST',
                    configuration=cfg,measurement=scores)
        records.append((record,arrays))
    return records


def benchmark_metrics(records):
    """Every seed is evaluated separately; do not turn seeds into an ensemble."""
    rows = []
    for r, a in records:
        stats = uncertainty_metrics(a["probabilities"], a["labels"])
        if "ood_probabilities" in a:
            stats.update(ood_metrics(a["ood_id_probabilities"], a["ood_probabilities"]))
        rows.append({"method": r["label"], "seed": r["seed"], **stats})
    return pd.DataFrame(rows)


def benchmark_panel(records, title):
    """Common four-panel diagnostics. Seed 41 is fixed before viewing curves."""
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 8))
    have_ood = all("ood_probabilities" in a for r, a in records if r["seed"] == 41)
    for r, a in records:
        if r["seed"] != 41:
            continue
        p, y, name = a["probabilities"], a["labels"], r["label"]
        calibration_plot(p, y, ax=axes[0, 0], label=name)
        risk_coverage_plot(p, y, ax=axes[0, 1], label=name)
        roc = ranking_curve(1-p.max(1), p.argmax(1) != y)
        axes[1, 0].plot(roc.fpr, roc.tpr, label=f"{name} ({roc.auroc:.3f})")
        if have_ood:
            ood_roc_plot(1-a["ood_id_probabilities"].max(1), 1-a["ood_probabilities"].max(1),
                         ax=axes[1, 1], label=name)
        else:
            axes[1, 1].hist(p.max(1), bins=np.linspace(0, 1, 31), histtype="step",
                            density=True, label=name)
    axes[1, 0].plot([0, 1], [0, 1], "k--", lw=.6)
    axes[1, 0].set(title="Error detection ROC (AUROC)", xlabel="correct cases flagged",
                   ylabel="errors flagged", xlim=(0, 1), ylim=(0, 1), aspect="equal")
    if not have_ood:
        axes[1, 1].set(title="Confidence distribution", xlabel="maximum probability", ylabel="density")
    else:
        axes[1, 1].set_title(f"OOD: {records[0][0]['ood']} (MSP)")
    for ax in axes.flat:
        ax.grid(alpha=.15)
        ax.legend(fontsize=7, framealpha=.9, loc="best")
    fig.suptitle(title+" — seed 41", fontweight="bold")
    fig.tight_layout()
    return fig


def benchmark_learning(directory, family):
    """Saved validation trajectories; average only epochs present for all seeds."""
    table = pd.read_csv(Path(directory)/"learning_curves.csv")
    subset = table[(table.family == family) & (table.cohort.isin(["main", "scratch"]))]
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7))
    for name, group in subset.groupby("label", sort=False):
        color = None
        for ax, metric, x in zip(axes.flat,
                ["validation_nll", "validation_accuracy"]*2,
                ["epoch", "epoch", "training_seconds", "training_seconds"]):
            for _, seed in group.groupby("seed"):
                seed = seed.sort_values("epoch")
                line, = ax.plot(seed[x], seed[metric], alpha=.3, lw=.7, color=color)
                if color is None: color = line.get_color()
            means = group.groupby("epoch")[[x, metric]].mean() if x != "epoch" else group.groupby("epoch")[metric].mean().reset_index()
            # Recorded training time varies by seed; this is mean time at a
            # common epoch, never interpolation beyond a stopped run.
            counts = group.groupby("epoch").seed.nunique()
            keep = counts[counts == group.seed.nunique()].index
            means = means.loc[means.index.isin(keep)] if x != "epoch" else means[means.epoch.isin(keep)]
            ax.plot(means[x], means[metric], label=name, color=color)
            ax.set(xlabel=x.replace("_", " "), ylabel=metric.replace("_", " "))
            ax.legend(fontsize=7, framealpha=.9)
            ax.grid(alpha=.15)
    fig.tight_layout()
    return fig
