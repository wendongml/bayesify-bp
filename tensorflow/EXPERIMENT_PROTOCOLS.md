# Recorded experiment protocols and implementation provenance

The selected examples use two explicitly separated implementations. MNIST and
CIFAR use the released R/TensorFlow training recipes. The selected transformer
results use the original Python/PyTorch sources now included under `python/`.
Their opt-in notebook cells execute those sources; the R transformer constructors
explain the moment computations and are not substituted for the original runs.
This document retains the earlier R full-weight transformer contracts as provenance,
not as the configuration of the later selected RoBERTa-LoRA result.

Mathematical checks, update checks, and result replay are documented in
`records/validation.json`; full training follows the schedules below.

This package contains executable R/TensorFlow image experiments and the original
Python/PyTorch transformer sources. Result replay and new training are separate
execution modes.

The reading edition also includes saved PyTorch GPT-2 and RoBERTa predictors,
restored from their original probabilities and run receipts. Their tables use
the same format as the CIFAR models and are not new R training results.
The 92.32% EP checkpoint and the three-seed four-direction serving ablation are documented in
`advanced_results/gpt2_selected_campaign/PROVENANCE.json`. The R reproduction
results and the contracts below remain unchanged. Every displayed metric
is recomputed by the R replay; no
metric or timing is transferred between the two implementations.

## Stronger compact-VI transformer comparisons

The main CLINC result uses the later RoBERTa-base
rank-8 query/value LoRA experiment, alongside the stronger GPT-2 rank-4 VI
result. These are saved PyTorch fits, recomputed in R from their probability
arrays. They do not relabel the full-weight R reproduction described below.
The learned distribution is diagonal Gaussian adapter A, point B and point
classifier, with frozen base weights and embeddings. Training uses the
BLoB-inspired likelihood-AdamW/prior-SGD split and conditional-moment stochastic
activation carriers. It is neither EP nor Vadam. The source protocol is copied
in `advanced_results/compact_transformers/FIXED_TRANSFERS.md`.

SST-2: seed 41, 67,349 fitting examples, three fixed epochs, 872 official
validation examples; nonlinear K1 prediction. CLINC: seed 41, 13,500 fitting
and 1,500 training-stopping examples, selected epoch 6, 4,500 official ID test
examples; tangent K2 prediction. Both average probabilities before NLL.
Rank/serving choices are exploratory saved ablations, not new independent fits.
The complete recorded training totals remain distinct from optimized A100
restored-step measurements. Inference uses five complete-cohort timings and
reports setup separately. No new benchmark training, model prediction run or
OOD collection was performed for this supplement revision.

## CIFAR: registered first reproduction

ResNet-20 and DenseNet-40 (growth 12, no bottleneck/compression), 45,000 fitting
images, fixed 5,000 stopping images, 100 epochs, batch 125, seed 41. The official
10,000-image test set is evaluated only after checkpoint selection. The same
initial means and minibatch order are used across methods within the new port.
TensorFlow and PyTorch random-number generators are not assumed to generate
identical initial tensors from the same integer seed.

The selected archived seeds are 41 for both CIFAR architectures and GPT-2,
and 42 for RoBERTa/CLINC. The default notebook roster uses these per-benchmark
seeds. The first CLINC reproduction used 41; those complete runs are retained
as a separate comparison rather than relabelled as seed 42. Seed provenance
is recorded in `archived_seed_contracts.json`. Reusing a source seed does not
constitute a new, independently selected best-seed search.

| Method | Existing update being reproduced | Rate | Field/KL strength | Training draws |
|---|---|---:|---:|---:|
| Ordinary | AdamW, decay 0.005 | 0.003 | — | 1 point network |
| Sampled MFVI | Historical Vadam recurrence | 0.003 | 0.001 | 1 |
| Moment MFVI | Same Vadam recurrence with moment gradient statistics | 0.003 | 0.001 | 0 |
| Sampled power-EP | Adam on mean and log variance | 0.003 / 0.003 | 0.001 | 4 |
| Moment power-EP | Adam on mean and log variance | 0.003 / 0.003 | 0.01 | 0 |

The moment closure uses one shared covariance direction, Gaussian first-Hermite
ReLU moments, and a conditionally integrated Gaussian readout. The classification
head uses 18 cubature points in nine logit-contrast coordinates. These are not
18 sampled networks. EP includes its cavity and external-field terms at every
minibatch. MFVI uses the reference likelihood-gradient second-moment statistic.
All rates use the same 100-epoch cosine schedule. This is a reproduction of named
historical configurations, not a claim of equal tuning budgets or a final
multi-seed benchmark.

Independent PyTorch/R checks use identical saved arrays, directions, noise and
batches. Separate scalar recurrence checks use float64. Float32 ReLU/Adam equality
failures are preserved; strict float64 full-update checks distinguish arithmetic
sensitivity from a changed update equation. No test tolerance is silently raised.

## Transformer ports: fixed source-matched equations and full-data protocols

The distinct historical protocols below now have separate implementations.
Independent strict float64 full-model checks pass for GPT-2 EP/MFVI and
RoBERTa EP/MFVI, including actual learned-belief and optimizer-state updates.
Implementation checks establish the update equations. Full-training execution receipts and per-example predictions separately establish predictive performance; neither can substitute for the other.

GPT-2-small rank-4 Q/V LoRA uses scale 8, zero prior mean, precision 2500,
initial variance 1e-6, A initialization standard deviation 0.02, zero B,
dropout disabled, and negative/positive token classification. The historical
prompt contains two demonstrations. Tokenization preserves that prefix and the
decision suffix while right-truncating only the new review body. The EP winner
uses exact first-order scalar variance contracted by module and 20-point
Gauss-Hermite probability integration, with full variance sensitivities.
It is not a generic one-direction/two-point approximation.

The historical SST-2 winner trained on all 67,349 training rows and selected
on the public 872-row validation set. An independent stopping-set study is a
different protocol and must not be silently presented as its exact reproduction.

RoBERTa-base/CLINC uses uncertain full encoder weights, pretrained prior centers,
initial variance 1e-8 and precision 1e6, a 150-class head initialized with
standard deviation 0.02, dropout disabled, and a duplicate-grouped training-only
stopping split. Its public validation, test and out-of-scope cohorts must remain
distinct. The selected moment MFVI uses likelihood-only EMA normalization,
frozen-Jacobian gradient curvature and 128 antithetic Sobol pairs. These differ
from the CIFAR update/closure; their independent tests are kept separate.

Source evidence: the saved run sources under
`moment_vadam_mnist/transformer_campaign/winner_uncertainty_20261026/recovered/`,
the CLINC selection and training source, and the immutable prediction/configuration
records already preserved in `evidence/`. They are read-only comparison material.

## DenseNet execution without changing the update

The local-tape implementation retains boundary moments and reconstructs each
module during its reverse pass. Its MFVI curvature calculation reverses the
point-network directional derivative with terminal signals `(H_loss Jd, grad_loss)`.
This is exactly the original Hessian-vector product, not a new covariance
approximation or optimizer. Full DenseNet float64 checks compare every belief,
normalization and optimizer tensor with the earlier implementation: MFVI
maximum absolute difference 1.78e-15, two EP updates exactly equal. A separate
compiled-update check also passes. The GPU gate retains batch125 and tests the
actual eager/compiled update before permitting full training.

If a full-batch float32 arithmetic comparison fails, its discrepancy is retained.
The independent strict check uses every model parameter and two real examples
in float64 on CPU for DenseNet and transformers. Its tolerance remains
1e-10 + 1e-8 times the reference scale. This checks the equations without
duplicating an already resident full-batch GPU graph. It is not a claim that
float32 and float64 optimization trajectories are identical.

## Additional frozen text evaluations

The five SST-2-trained beliefs are also evaluated without retraining on a fixed
1,000-example IMDb sentiment subset and a fixed1,000-example AG News subset
for out-of-task rejection. Inputs remain capped at128tokens. Their identities,
class balance and dataset revisions were registered before prediction. IMDb
truncation is substantial and is reported. Results are retained for all five
methods, including cases where the ordinary model performs better.

CLINC uses its official4,500-example in-domain test and1,000 out-of-scope
examples only after checkpoint selection on its training-only stopping set.
The public development results are reported separately. No test cohort is
used to choose a favorable checkpoint or regularization setting.
