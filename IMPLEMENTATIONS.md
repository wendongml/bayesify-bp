# One mathematical tutorial, three implementations

R/TensorFlow is the default reading entry point. PyTorch and JAX are equally
first-class native implementations, with their own source, checks, architecture
inspection, runnable notebooks, and numerical evidence.

| Mathematical role | R / TensorFlow | PyTorch | JAX / Flax |
|---|---|---|---|
| Gaussian mean/raw-second coordinates | `tensorflow/R/gaussian.R` | `geometry.py` | `gaussian.py` |
| Forward layers and literal local reverse | `layers.R`, `messages.R` | `layers.py` | `layers.py` |
| Shared-parent accumulation and explicit reverse pass | `layers.R`, `messages.R` | `graph.py` | `messages.py` |
| EP and MFVI objectives / finite updates | `inference.R` | `heads.py`, `inference.py` | `heads.py`, `inference.py` |
| Activation correlation / structured weight beliefs | `covariance.R` | `correlation.py`, `residual.py` | `covariance.py`, `residual.py` |
| Conditional readout mean/variance derivatives | `inference.R` | `heads.py` | `heads.py` |
| MLP, LeNet, ResNet, DenseNet | `models.R` | `models.py`, `architectures.py` | `models.py` |
| Transformers and conditional adapters | `models.R`; labeled original Python sources | `transformer_inspection.py`; original training sources | `transformers.py`, `blob.py`, `gaussian_text.py` |
| Uncertainty metrics and plots | `uncertainty.R`, `plots.R` | `metrics.py`, `plotting.py`, `results.py` | `uncertainty.py`, `plots.py`, `results.py` |
| Independent checks | `checks.R` | `tests/` | `tests/`, independent Gaussian references |

Python module paths are relative to `pytorch/src/bayesify_torch/` and
`jax/src/bayesify_jax/`. R filenames after the first row are relative to
`tensorflow/R/`. The original transformer sources live in
`tensorflow/python/` because they accompany that original supplement; their
PyTorch provenance remains explicit.

## Reading and evidence

The readers share eight topics and the restored original Gaussian-factor, layer,
message-carrier and site-update sections. [TUTORIAL_PARITY.md](TUTORIAL_PARITY.md)
maps the original author notebook and records where the experiments differ.
Shared headings alone are not evidence of matched implementation or training. Each notebook
keeps its executable order and saved outputs; the common web renderer changes
presentation, not scientific results. PDFs retain their separately typeset
mathematics and code. Switching implementations preserves the current topic.

Exact Gaussian controls establish the coordinate/message identities. Nonlinear
Monte Carlo comparisons measure approximation error. Benchmark tables measure
predictive quality, uncertainty, and cost under the stated protocol. Agreement
in one of these tests does not substitute for another.

The large JAX transformer training runs are conditional Gaussian-A/point-B MFVI
LoRA. A separate fixed-belief check transfers the original Gaussian A+B MFVI/EP
adapters and compares native predictions and all adapter likelihood derivatives.
Its full-weight API has smaller implementation tests; those are not presented as
full-weight benchmark results. The original R reader's selected transformer fits
use its explicitly labeled Python sources.
Framework-specific reading maps provide the detailed correspondence:
[R](tensorflow/SECTIONS.md), [PyTorch](pytorch/READER_MAP.md),
[JAX](jax/MIRROR_MAP.md).

## Maintaining the readers

Edit mathematical/source content in the selected implementation and retain its
execution evidence. `python site/build.py` renders all three executed notebooks
with shared CSS, navigation, and topic anchors. `python site/check.py` verifies
the generated site and downloads. The renderer never trains or replaces a
notebook's outputs. Expensive training is an explicit separate step.

## Source-size audit

The core implementation counts are R **2,361**, PyTorch **2,617**, and JAX
**1,979** physical lines. Whole native module directories contain 4,789, 4,429,
and 2,917 lines respectively. See [the counting scope and per-file inventory](TUTORIAL_PARITY.md#comparable-implementation-sizes).

The R reader uses native **IRkernel**, as the original author notebook did.
`python site/check_tutorials.py` checks that contract, source/listing correspondence
and shared restored mathematical content.
