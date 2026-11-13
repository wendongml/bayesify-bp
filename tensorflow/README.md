# Bayesify

**Bayesian Backprop as Belief Propagation: Single-Pass Predictive Uncertainty**  
[NeurIPS 2026 paper page](https://neurips.cc/virtual/2026/poster/154383) ·
[Project website](https://wendongml.github.io/bayesify-bp/) ·
[Executable tutorial](README.ipynb) ·
[Tutorial PDF](README.pdf) ·
[v0.1.0 release](https://github.com/wendongml/bayesify-bp/releases/tag/v0.1.0)

> **R + TensorFlow implementation.** Part of the shared Bayesify repository alongside [PyTorch](../pytorch/README.md) and [JAX](../jax/README.md).

Implementation, derivations, and experiments for moment-based Bayesian neural
networks. The examples cover exact Gaussian inference, EP and MFVI, activation
covariance approximations, MNIST and CIFAR-10 classifiers, and transformer adapters.

Start with [README.ipynb](README.ipynb), the [HTML](README.html), or the
[PDF](README.pdf). They follow the same sequence of mathematics,
code, figures, and results. The notebook includes all executable source;
supporting cells are expandable in the notebook and HTML.

## Paper and citation

If this implementation contributes to your work, please cite the paper:

```bibtex
@inproceedings{dong2026bayesian,
  title     = {Bayesian Backprop as Belief Propagation: Single-Pass Predictive Uncertainty},
  author    = {Dong, Wen},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  url       = {https://neurips.cc/virtual/2026/poster/154383}
}
```

The machine-readable citation is in [`CITATION.cff`](../CITATION.cff).

## Run

Open `README.ipynb` and run its cells in order. The default mode recomputes the
small mathematical checks and approximation comparisons, then displays the
labeled recorded benchmark tables and figures. The notebook includes the full
source and can also be opened by itself in an R/IRkernel runtime, including Colab with an R runtime.

**No checkpoints, model weights, full datasets, or bulk prediction draws are included.**
`results/` contains compact metrics, histories and plotting coordinates. The
larger experiment archive is not needed to read the supplement or run its
mathematical checks. Fresh benchmark training regenerates model states and
per-example predictions from public data.

`Rscript reproduce_results.R` runs the same R cells. This requires R packages
`reticulate`, `jsonlite`, `IRdisplay` and `base64enc`, and Python TensorFlow; set
`RETICULATE_PYTHON` if needed. Use the notebook/HTML for formatted recorded outputs.

For fresh training, enable `DOWNLOAD_PUBLIC_DATA` and the appropriate
`RUN_MNIST_TRAINING`, `RUN_CIFAR_TRAINING`, or
`RUN_ORIGINAL_TRANSFORMER_TRAINING` switch. These run full public-data protocols
and report newly generated predictions. Full training requires a GPU and takes
substantially longer than replaying saved results.

## Implementation

The R code uses one shared forward graph and explicit reverse traversal. EP and
MFVI have distinct objectives and finite updates. Handwritten reference formulas
check the shared implementation. Covariance directions change the approximation
represented on that graph.

| Files | Contents |
|---|---|
| `R/gaussian.R`, `R/messages.R` | Gaussian coordinates, EP/MFVI equations, independent references |
| `R/layers.R`, `R/covariance.R` | Forward maps, reverse traversal, covariance approximations |
| `R/models.R` | MLP, LeNet, ResNet, DenseNet and transformer constructors |
| `R/inference.R` | Heads, cavities, optimizer recurrences and prediction |
| `R/experiments.R` | Data preparation, fixed training protocols and evaluation |
| `R/uncertainty.R`, `R/plots.R` | Metrics and diagnostic plots |
| `R/checks.R`, `R/results.R`, `R/setup.R` | Checks, record readers and runtime setup |
| `python/gpt2/`, `python/blob_mirror/` | Original transformer training and prediction implementations |

The image results use R/TensorFlow. Selected transformer fits use the separately
labeled Python/PyTorch sources; their R constructors describe the architecture
and moment operations. The BLoB-inspired example uses stochastic conditional
activation carriers and an AdamW/SGD split update. It is distinguished from
literal EP and the canonical MFVI update.

[SECTIONS.md](SECTIONS.md) lists cell ranges, code sizes and dependencies.
`records/source_map.json` maps complete notebook definitions to their R/Python
files. `EXPERIMENT_PROTOCOLS.md` records data splits, optimization and prediction
settings. Compact metrics, histories and plot coordinates are under `results/`;
small mathematical fixtures are under `advanced_results/`.

## Validation and interpretation

The included checks distinguish inference identities, approximation fidelity,
and predictive performance. Default reading mode does not retrain benchmarks
or remeasure their GPU timings. Training and prediction costs retain their
recorded hardware and protocols. `records/validation.json` describes verification
coverage. The combined release ZIP includes `MANIFEST.json` with current file checksums.
Checksums for the original R supplement are preserved separately in
`records/original_r_supplement.sha256` as source provenance.

Public models, datasets and third-party methods retain their source attribution.
The optional BLoB dependency is [bayesian-peft](https://github.com/Wang-ML-Lab/bayesian-peft),
pinned to commit `7d960082efa3a680b76aac79d2a4f13d7f0f651a`. No private bucket
or author account is required for the included public-data training recipes.
