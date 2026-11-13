# Bayesify

**Bayesian Backprop as Belief Propagation: Single-Pass Predictive Uncertainty**  
[NeurIPS paper](https://neurips.cc/virtual/2026/poster/154383) ·
[Read the tutorial](https://wendongml.github.io/bayesify-bp/) ·
[Download](https://github.com/wendongml/bayesify-bp/releases/latest) ·
[Citation](CITATION.cff)

One project, with three native implementations of Gaussian moment propagation,
explicit backward messages, and Bayesian uncertainty experiments. The project
website opens the **R + TensorFlow tutorial**. Its implementation tabs switch to
the corresponding topic in PyTorch or JAX.

| Implementation | Read | Run | Source |
|---|---|---|---|
| **R + TensorFlow** | [HTML](https://wendongml.github.io/bayesify-bp/tensorflow/README.html) · [PDF](tensorflow/README.pdf) | [Notebook](tensorflow/README.ipynb) · [Colab](https://colab.research.google.com/github/wendongml/bayesify-bp/blob/main/tensorflow/README.ipynb) | [R modules](tensorflow/R) |
| **PyTorch** | [HTML](https://wendongml.github.io/bayesify-bp/pytorch/README.html) · [PDF](pytorch/README.pdf) | [Notebook](pytorch/README.ipynb) · [Colab](https://colab.research.google.com/github/wendongml/bayesify-bp/blob/main/pytorch/README.ipynb) | [Python modules](pytorch/src/bayesify_torch) |
| **JAX / Flax** | [HTML](https://wendongml.github.io/bayesify-bp/jax/README.html) · [PDF](jax/README.pdf) | [Notebook](jax/README.ipynb) · [Colab](https://colab.research.google.com/github/wendongml/bayesify-bp/blob/main/jax/README.ipynb) | [Python modules](jax/src/bayesify_jax) |

Each notebook is self-contained, including its computational source and small
numerical evidence. Default reading mode runs mathematical checks and shows
saved benchmark outputs. Full public-data training is opt-in; it takes longer
than replaying the tutorial. No trained checkpoints or datasets are distributed.

## What to read

The tutorials follow **mathematics → code → its output → interpretation**:

1. Gaussian coordinates and shared forward layers.
2. Gaussian, EP, and MFVI message-passing/backpropagation equivalence.
3. Sampling-assisted moment propagation: accuracy versus Monte Carlo and cost.
4. MNIST with MLP and LeNet.
5. CIFAR-10 with ResNet and DenseNet.
6. GPT-2 and LoRA.
7. Conditional moments and RoBERTa / CLINC150.
8. Reproduction and code reading.

[IMPLEMENTATIONS.md](IMPLEMENTATIONS.md) maps equations and modules across all
three frameworks. The source remains idiomatic to each framework, and every
results table identifies its method, prediction mode, and experiment.

[Tutorial correspondence and execution evidence](TUTORIAL_PARITY.md) compares
the three implementations. The [matched approximation study](site/evidence/README.md)
leads with the sampling-assisted nonlinear residual repair. It keeps the
uncorrected 64-direction closure as an ablation and checks mean and variance
messages separately, with numerical data and scripts to regenerate the figures.

## Run locally

Follow the setup instructions inside your chosen directory:
[TensorFlow](tensorflow/README.md), [PyTorch](pytorch/README.md), or
[JAX](jax/README.md). Use separate environments for their pinned dependencies.
The notebooks also run when copied individually to Colab.

The original transformer training sources accompanying the R tutorial are
preserved and explicitly labeled under `tensorflow/python/`. They are PyTorch
sources, not a claim of TensorFlow training. The native PyTorch and JAX readers
explain their own execution and validation scope.

## Development

The code and documentation are under active development. For a reproducible
snapshot, use a [release](https://github.com/wendongml/bayesify-bp/releases);
changes are recorded in [CHANGELOG.md](CHANGELOG.md). Each experiment includes
its data split, optimization settings, prediction mode and measured costs.

## Citation

```bibtex
@inproceedings{dong2026bayesian,
  title     = {Bayesian Backprop as Belief Propagation: Single-Pass Predictive Uncertainty},
  author    = {Dong, Wen},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  url       = {https://neurips.cc/virtual/2026/poster/154383}
}
```

Public datasets, pretrained models, and third-party methods retain their source
attribution. The optional BLoB reference is
[bayesian-peft](https://github.com/Wang-ML-Lab/bayesian-peft), pinned in the
reproduction code. No author account or private bucket is required.
