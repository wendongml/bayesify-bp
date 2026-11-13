# R/TensorFlow → JAX reading map

The port preserves the equations and shares layers between EP and MFVI. JAX
PyTrees replace R lists, `jax.vjp` replaces a local TensorFlow tape, and Flax NNX
owns the model parameters. Numerical references remain independently implemented.

| Original R file | JAX counterpart | Purpose and executable evidence |
|---|---|---|
| `setup.R` | notebook setup; `pyproject.toml` | Environment, dtype and reproducible execution |
| `gaussian.R` | `gaussian.py`, `gaussian_chain.py`, `gaussian_reference.py` | Full/diagonal Fisher coordinates; finite EP/canonical MFVI; independent conditioning and RTS |
| `layers.R` | `layers.py` | Dense, convolution, ReLU, pooling, normalization, joins; literal and local-VJP reverse |
| `messages.R` | `messages.py`, `heads.py` | One cached forward/reverse traversal; duplicate-parent accumulation; separate EP/MFVI heads |
| `covariance.R` | `covariance.py`, `residual.py` | Shared activation factors, structured weights, Gaussian residual and gradient-moment integration |
| `models.R` | `models.py`, `transformers.py` | MLP, LeNet, ResNet-20, DenseNet-40, GPT-2 and RoBERTa; real NNX summaries and ONNX graphs |
| `inference.R` | `inference.py`, `training.py`, `gaussian_text.py` | Explicit Gaussian EP sites and canonical MFVI; separately named minibatch optimizer variants |
| Original conditional-adapter training | `blob.py`, `text_training.py` | Gaussian-A/point-B MFVI and conditional activation carriers; source split optimizer |
| `checks.R` | `tests/`, `run_validation.py` | Independent inference, derivative and negative-control checks |
| `experiments.R` | `experiments.py`, `run_*`, `learned_fidelity.py` | Native experiments; no imported R predictions |
| `uncertainty.R` | `uncertainty.py` | Accuracy, NLL, Brier, ECE, selective risk, error ranking, OOD ranking |
| `plots.R` | `plots.py`, `experiments.py` | Native Matplotlib figures with equal-scale identity comparisons |
| `results.R` | `results.py` | Regenerate tables and figures from per-example numerical records |

The original Gaussian adapter/full-weight path and conditional Gaussian adapter
path are different model families. Their code and labels remain separate. The
latter is MFVI, not EP. Exact-site EP, the tied EP energy optimized with Adam, and
moment approximations of nonlinear heads are also named separately.

The NNX modules describe inspectable architectures; pure functions receive
immutable array snapshots during numerical inference. This is the usual JAX
separation between model/state ownership and compiled calculations, not a second
independent implementation of the layers.

For a line-by-line mathematical comparison, start with these functions:

| Operation in the R reader | JAX function | Independent check |
|---|---|---|
| `encode_diagonal`, `decode_diagonal` | `gaussian.encode`, `gaussian.decode` | Nonzero-mean round trips and deterministic-input tests in `test_core.py` |
| `encode_full`, `decode_full` | `gaussian.encode_full`, `gaussian.decode_full` | Full-covariance round trips in `test_core.py` |
| `tilted_update`, `canonical_mfvi` | `gaussian.ep_moment_update`, `gaussian.mfvi_update` | Scalar negative control, missing-quadratic-term control, and coordinate VI |
| `moment_dense`, `mfvi_dense_backward` | `layers.dense`, `layers.dense_backward` | Literal versus local VJP and Euclidean finite differences |
| `moment_conv2d`, `mfvi_conv2d_backward` | `layers.convolution`, `layers.convolution_backward` | Convolution input, weight, bias, mean and variance adjoints |
| ReLU, pooling and normalization | `layers.relu`, `max_pool`, `avg_pool`, `normalization` and their backward functions | Separate literal/local-VJP layer tests |
| `moment_graph_forward`, `moment_graph_reverse` | `messages.forward`, `messages.backward` | Complete nonlinear MLP and repeated-parent/parameter accumulation |
| Literal custom-gradient wrapper | `layers.literal_autodiff` | Custom VJP returns decoded Euclidean cotangents, checked against ordinary autodiff |
| `moment_forward`, `moment_reverse` with shared factors | `covariance.shared_records`, `covariance.shared_likelihood` | Residual cross terms and explicit reverse versus global autodiff |
| `network_residual_backward` | `residual.network_residual_backward` | Paired nonlinear-minus-Gaussian integration; same-belief MC plots |
| `gaussian_mlp_condition`, `gaussian_mlp_smooth` | `gaussian_reference.condition`, `gaussian_reference.chain_reference` | Dense conditioning and RTS compared with `gaussian_chain.candidate`; reference does not call candidate |
| Gaussian and categorical observation heads | `heads.gaussian_head`, `heads.categorical_head` | Distinct EP/MFVI objectives and current-state ELBO–KL certificate |

The literal convolution reverse uses JAX's convolution transpose/VJP primitive
for the linear convolution operation; its Gaussian mean/variance chain rule is
written explicitly. It does not obtain its whole moment-layer reverse by
differentiating the projected implementation. Likewise, shared-factor boundaries
carry ordinary factor adjoints: a finite-rank factor is not inverted as though
it were a positive-definite full Gaussian covariance.

The covariance extensions offer shared-factor activation propagation and a
`D + LLᵀ` weight representation. Benchmark training here uses diagonal weight
beliefs. Availability of a structured representation is not evidence of a
trained structured-posterior result.

`shared_forward` is the fully first-order covariance ablation, including its
readout. `shared_records` uses the same directional kernels with a Gaussian
readout that analytically adds conditional readout-weight noise. The small
synthetic rank study uses the former; CNN training and the learned-ResNet
comparison use the latter. This readout choice is explicit in the tutorial.

`run_gaussian_text.py --full-weight` exposes full-weight transformer training;
its update path is tested on a small transformer. The large text tables instead
use the explicitly labeled LoRA protocols. They must not be described as
full-weight RoBERTa results.
