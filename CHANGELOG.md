# Releases

## v0.1.0 — 2026-11-12

Initial public implementation accompanying **Bayesian Backprop as Belief
Propagation: Single-Pass Predictive Uncertainty**.

- One project with R/TensorFlow, PyTorch, and JAX/Flax implementations.
- Tutorial website with implementation tabs, executable notebooks, and compact
  LaTeX PDFs; R/TensorFlow uses a native R kernel.
- Gaussian, EP, and MFVI equations, proofs, layers, and message-equivalence checks.
- Sampling-assisted moment refinement with measured approximation errors and
  computational costs; retained-direction closures are presented as ablations.
- MNIST, CIFAR-10, GPT-2/LoRA, and RoBERTa uncertainty examples, with protocols,
  numerical evidence, and implementation-specific execution scope.
- Source correspondence, architecture inspection, reproducible figures, and a
  combined release archive with file checksums.

The release contains source and compact evidence, without trained checkpoints
or full datasets. New development will continue separately from published releases.
