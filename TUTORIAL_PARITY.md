# Tutorial correspondence and source-size audit

The comparison uses the author's original **Bayesian_Backprop_as_Belief_Propagation.ipynb**
(71 cells; native R kernel), not just the eight navigation headings added for the
website. Original SHA-256:
`da2d16b21a5d58417a7d254989aabb2bfc4a38a3a9dd52bfce157a169caec8cb`.
The original file is preserved unchanged.

## Approximation reading path

Section 3 now leads with sampling-assisted nonlinear residual integration. The
uncorrected 64-direction closure is an ablation. The earlier R/TensorFlow repair
validates selected-coordinate mean/raw-second improvement; the shared PyTorch
readout study separately validates centered-variance accuracy. Small-fixture
native correction checks and matched closure checks do not establish three
independently executed all-layer corrected training runs. The saved benchmark
tables are unchanged. Only figures rebuilt from existing data and reader text
were refreshed for this revision; no new training is represented.

## Reading and execution contract

Each reader follows the same mathematical route: Gaussian coordinates and shared
layers; exact, EP and MFVI message checks; approximation versus Monte Carlo;
MNIST; CIFAR-10; Gaussian transformer adapters; conditional moment adapters;
and reproduction. A common heading is not sufficient: the tables below identify
the actual computations and their sources.

- The R notebook uses native **IRkernel (`ir`)**, without `%%R` or an rpy2 bridge.
  The original Python transformer training programs remain explicitly labeled.
- The local Gaussian-factor proof, explicit-message/autodiff correspondence,
  layer equations and cavity/site/update-schedule semantics are shared in
  `site/tutorial_content.json`. Native implementation listings and outputs follow
  those derivations in each reader.
- The standalone notebooks carry their native sources and numerical evidence,
  not checkpoints. Download/training switches are separate from reading and
  numerical validation. Historical training time is not notebook execution time.
- `python site/check_tutorials.py --write` checks native kernels, execution of
  every code cell, source/listing correspondence, embedded source identity,
  mathematical content and actual source sizes. Its receipt is
  [site/tutorial_validation.json](site/tutorial_validation.json).

## Original topic → corresponding reading section

All three readers now follow the same conceptual route. The second column gives
one-based cells in the immutable original; the third identifies material in
**each** current reader, not merely a renamed chapter heading.

| Original topic | Original cells | Corresponding current material |
|---|---:|---|
| Local affine Gaussian factor; uncertain activation and weight | 4–6 | §1 “Local Gaussian factor” and Gaussian coordinate bridge |
| Explicit conditional messages versus ordinary autodiff | 7–12 | §1 two-pipeline derivation; §2 literal/projected carrier contract and exact-Gaussian plots |
| Weight-belief messages versus optimizer gradients | 14 | §1 reconstruction; §2 message contract; §4 separate finite optimizers |
| Nonlinear activations and local approximation | 15 | §1 layer equations; §3 nonlinear closure, covariance and MC fidelity |
| MNIST-based Gaussian/Keras pipeline demonstration | 18–25 | §2 Gaussian activation/weight and EP/MFVI equivalence controls; §4 MNIST models and uncertainty. The current exact fixtures are not a rerun of that original trained-MNIST demonstration. |
| Exponential-family tilted-normalizer identity | 27–29 | §2 EP identity/proof, Gaussian Fisher conversion and cavity/site section |
| Literal and projected layers | 30–52 | §1 full shared layers, explicit affine reverse and native source; §2 independent literal/VJP checks |
| Dense, convolution, nonlinearities, pooling, normalization, merge | 32–52 | §1 layer equations and catalogue; §§4–5 model builders; §§6–7 transformer maps |
| Architecture construction and inspection | 53–64 | §§4–7 native architecture code and summaries. Current image benchmarks are ResNet-20 and DenseNet-40, not the original DenseNet121/WRN28-8 catalogue. |
| Projected/literal model validation | 65–66 | §2 method-specific equivalence; §3 approximation/MC checks; independent native tests |
| Uncertainty statistics and diagnostic drawing | 67–68 | §4 common metric definitions and plotting source; reused in §§5–7 |
| Training, calibration and reproduction | 69–71 | §§4–7 method-specific protocols/results and §8 reproduction |
| MFVI counterpart (later extension) | Not a separate original chapter | §2 MFVI objective/canonical proof and literal/VJP checks; §§4–7 labeled MFVI experiments |
| Correlated activations/weights (later extension) | Not a separate original chapter | §3 shared factors, structured Gaussian algebra and quality/cost comparison |

A common topic is not a claim that every API, approximation or trained model is
identical. In particular, a custom-gradient transport API and an explicit reverse
traversal can carry the same moment message without being the same program.

## Scientific demonstrations and their evidence

| Demonstration | R / TensorFlow | PyTorch | JAX |
|---|---|---|---|
| Gaussian, EP and MFVI message equivalence | Native reference/message checks and plots | Native reference/message checks and plots | Native reference/message checks and plots |
| Same learned ResNet beliefs and mathematics | Native forward moments and all weight derivatives | Native forward moments and all weight derivatives | Native forward moments and all weight derivatives |
| Approximation versus high-precision MC | Native closure values and plotting; explicitly shared MC reference | Native closure values and plotting; MC/correction reproduction source | Native closure values and plotting; explicitly shared MC reference |
| MLP/LeNet and ResNet/DenseNet | Recorded protocols, native training source and attributed original optimized fits | Original optimized fits plus full native compact Dirty-MNIST validation | Native image training records, including Dirty-MNIST |
| Gaussian A+B GPT-2 adapters | Original Python fits explicitly labeled | Original trained beliefs and native likelihood derivatives | Same transferred trained beliefs and native likelihood derivatives |
| Conditional Gaussian-A/point-B GPT-2 and RoBERTa | Original Python fits explicitly labeled | Original native PyTorch training and serving records | Native JAX training and serving records |
| Architecture inspection | Native Keras summaries and downloadable graphs | Native torchinfo summaries and ONNX/Netron graphs, including transformers | Native Flax NNX summaries and downloadable graphs |
| Calibration and error/OOD detection | Native metric/plot code | Native metric/plot code | Native metric/plot code |

The Gaussian A+B fixed-belief transfer check does not constitute new JAX EP
training. Conditional Gaussian-A/point-B is a different variational family,
not an interchangeable name for that EP model. R constructors are not credited
with original Python transformer fits. These distinctions let readers reproduce
and compare the intended algorithm rather than accidentally compare different
families under one label.

## What the mismatches taught us

The matched learned-ResNet comparison uses identical beliefs, inputs, directions,
objectives and float64 arithmetic. All **80 comparisons pass**, with maximum
absolute error **1.82 × 10⁻¹³**. Consequently, none of the three frameworks is a
better approximation when it computes this same mathematical rule.

The common rule can nevertheless differ from the **524,288-network Monte Carlo
reference**. Retaining activation correlations markedly improves forward
covariance; increasing the number of retained directions does not guarantee a
better backward signal. A separate head-integral ablation keeps the hidden
approximation fixed. At 64 directions, improving that integral reduces MFVI's
variance-message relative error from **211.49% to 38.78%**, and EP's from
**41.83% to 31.63%**. A paired nonlinear readout correction with 16,384
evaluations reduces those errors to **5.85% and 0.97%**. These are local message
measurements on one predeclared input, not claims of improved test accuracy.

A separate float32 GPT-2 replay transfers the same saved Gaussian A+B beliefs
between the original PyTorch path and native JAX. All eight checks pass: the
maximum probability difference on 872 inputs is **9.00 × 10⁻⁶**, and the largest
relative discrepancy in the compared mean/variance likelihood derivatives is
**9.92 × 10⁻⁶**. Both compute the derivative of the logit Jacobian, at the same
four fixed examples. This tests the fully differentiated approximation before
an optimizer is applied. [The comparison script](site/compare_text_beliefs.py)
rebuilds the receipt from the saved derivative and prediction arrays.

Native 30-epoch Dirty-MNIST runs also exercise all five training methods in
PyTorch and JAX. These are independent one-seed fits, not identical RNG streams
or a framework ranking. Moment MFVI improves predictive NLL from 0.50594 to
0.48366 in PyTorch and from 0.49937 to 0.48312 in JAX; OOD improvements are not
uniform. The eager PyTorch traversal and compiled JAX path have markedly
different execution costs. Each reader retains its measured times and states
its compilation/timing boundaries instead of substituting another run's costs.

The full equations, MC uncertainty, data and cost boundaries are in
[the matched approximation study](site/evidence/README.md). Its figures rebuild
from numerical arrays using `site/compare_approximations.py`. Native closure
values, shared reference calculations and correction calculations are identified
separately. A shared reference is never relabeled as an independently executed
backend result.

The port comparison also exposed concrete implementation errors: likelihood
sum/mean scaling, BatchNorm state accounting, a reverse-over-JVP normalization
path, and nonlinear pooling/projection order. Independent finite differences
and regression tests now cover those cases. Agreement of forward means alone
would not have found the backward normalization defect.

## Comparable implementation sizes

Physical lines below include comments and blanks. Notebook copies of source,
HTML, PDFs, compressed payloads, model files and prediction records are excluded.
The complete per-file counts and category assignments are in
[site/source_lines.csv](site/source_lines.csv).

| Scope | R / TensorFlow | PyTorch | JAX |
|---|---:|---:|---:|
| Inference/model/training core¹ | 2,361 | 2,617 | 1,979 |
| Entire native module directory | 4,789 | 4,429 | 2,917 |
| Native module files | 12 | 21 | 22 |
| Nonblank module lines | 4,652 | 3,758 | 2,503 |
| Standalone test lines² | Included in module count | 1,230 | 597 |

¹ Core counts cover the Gaussian representation, layers, message traversal,
covariance, models and inference/update code. They identify the named files in
`site/check_tutorials.py`; they are an approximate role comparison, not an equal
feature count. The largest core is about **1.32×** the smallest; the largest
whole module tree is **1.64×** the smallest. R's `experiments.R`, `plots.R`, and
`results.R` alone account for 2,157 lines; framework organization also differs.

² R checks are already included in its 12-module total. PyTorch/JAX tests live
outside their native package directories. The R command-line replay/display driver is counted separately.
The explicitly labeled original Python transformer/support sources add **2,016
lines in 23 files**; they are used by the R/PyTorch readers and are not counted as
R algorithms. Build/export scripts are also separate from model implementation.
No code was padded or removed to make these counts look similar.

For actual cell ranges and visible versus supporting code, see
[R sections](tensorflow/SECTIONS.md), [PyTorch reader map](pytorch/READER_MAP.md),
and [JAX reader map](jax/READER_MAP.md).
