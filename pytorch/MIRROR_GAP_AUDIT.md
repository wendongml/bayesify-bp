# PyTorch implementation and evidence map

The reader follows the same mathematical and experimental route as its
R/TensorFlow and JAX counterparts. Each result identifies whether it is an exact
identity, an approximation comparison, a new native fit, or a historical fit.

| Question | Implementation and reader location | Evidence |
|---|---|---|
| How are Gaussian messages encoded? | §1; `geometry.py` | Independent round trips, finite updates and negative controls |
| Do literal messages equal backpropagation? | §2; `gaussian_chain.py`, `heads.py`, `inference.py` | Independent Gaussian conditioning/RTS, EP and MFVI comparisons |
| Does the port preserve trained-model derivatives? | §3; `graph.py`, `layers.py` | Same ResNet-20 beliefs/inputs/directions in three frameworks; 80 comparisons |
| Which approximation improves fidelity? | §3; `correlation.py`, `residual.py` | 524,288-network reference; rank, head-integral and paired-readout ablations |
| Can the compact implementation train? | §4; `benchmark.py`, `run_training.py` | Five full 30-epoch native Dirty-MNIST runs, seed 41; saved curves and test predictions |
| How do uncertainty and cost compare? | §§4–7; `metrics.py`, `plotting.py`, `results.py` | Native prediction records; cohorts, methods, prediction modes and costs retained |
| What are the actual architectures? | §§4–7; `models.py`, `architectures.py`, `transformer_inspection.py` | Native torchinfo summaries; six checked ONNX/Netron models |
| Does the text implementation agree? | §6; original Gaussian A+B adapter code | Same saved GPT-2 beliefs, 872 predictions and all adapter likelihood derivatives at four fixed examples, versus JAX |
| What changes for conditional adapters? | §7; original `python/blob_mirror` | Gaussian-A/point-B MFVI with a source-ordered split update, distinctly labeled |

## The approximation lesson

Identical mathematics on identical ResNet beliefs agrees across R/TensorFlow,
PyTorch and JAX to a maximum absolute error of 1.82e-13. This does not make a
Gaussian closure exact. The independent Monte Carlo reference exposes residual
forward and backward approximation errors. More covariance directions help
forward covariance but need not improve every derivative. Holding the hidden
closure fixed and improving the final integration isolates one source of error;
the paired nonlinear readout correction addresses another.

The public [numerical evidence](../site/evidence/README.md) provides equations,
reference uncertainty, arrays, figures and explicit timing boundaries. The
readout correction is a local fidelity experiment, not a newly trained model or
a general speedup result.

## Training and attribution

The full native Dirty-MNIST runs establish the compact trainer's end-to-end
execution. They do not retrospectively replace the original optimized
MLP/LeNet/CIFAR fits or their training times. Those earlier tables retain their
recorded provenance. Original Python transformer training source is included;
R and JAX are not credited with producing those historical PyTorch fits. The
new Gaussian A+B JAX comparison is a fixed-belief replay, not fresh EP training.
Conditional Gaussian-A/point-B MFVI is a separate family, not a relabeling of EP.

Every Python figure is generated from numerical arrays in Python. A common
reference computed once is explicitly attributed; it is never presented as
three independent Monte Carlo calculations. No checkpoint or dataset is bundled.

## Regression coverage

Independent tests cover likelihood sums, shared-parent accumulation, a single
BatchNorm state commit, reverse-over-JVP normalization derivatives, LeNet
projection order and pooling, Gaussian readout covariance, EP normalizers,
MFVI canonical updates, and prediction without unused backward records.
Finite differences complement autodiff comparisons. Sampling a different
Gaussian or comparing two differently normalized objectives is not a valid port
check.

[MATH_COVERAGE.md](MATH_COVERAGE.md) maps each mathematical R passage to the
native reader. [READER_MAP.md](READER_MAP.md) gives actual cells and visible versus
supporting code. [The repository comparison](../TUTORIAL_PARITY.md) gives native
source counts and experiment scope. Execution receipts are recorded separately
in [COLAB_VALIDATION.md](COLAB_VALIDATION.md).
