# Faster moment inference with the existing Bayesian advantage

Full reporting-set prediction takes 2-3x the ordinary-model cost on both transfer tasks. It uses one learned Gaussian adapter belief, no ensemble of independently trained networks, and no new training fit. Ordinary prediction receives the same applicable adapter merging, fused attention and CUDA graph optimizations.

## Measured quality and steady-state prediction time

Consumer Colab A100 40 GB; strict float32, TF32 disabled; batch size 32; five timed complete prediction passes. The step column is the restored training-step measurement detailed below, not total training time. SST-2 uses 872 reporting examples and CLINC 4,500. Accuracy is higher-is-better; NLL, ECE and time are lower-is-better. ECE uses 15 equal-width confidence bins.

| Task / predictor | Accuracy ↑ | NLL ↓ | ECE ↓ | Step ms (×) ↓ | Predict s (×) ↓ |
|---|---:|---:|---:|---:|---:|
| GPT-2 / SST-2 / ordinary | 91.40% | 0.26608 | 5.12% | 60.94 (1.00) | 0.958 (1.00) |
| GPT-2 / SST-2 / nonlinear K1 | 91.86% | 0.21489 | 2.02% | 171.42 (2.81) | 2.103 (2.19) |
| RoBERTa / CLINC / ordinary | 96.22% | 0.14990 | 0.99% | 29.79 (1.00) | 1.257 (1.00) |
| RoBERTa / CLINC / tangent K2 | 96.60% | 0.14172 | 0.72% | 91.04 (3.06) | 3.281 (2.61) |

These are predictions from the unchanged epoch-3 SST-2 and stopping-selected epoch-6 CLINC fits. The compact variants improve all three point estimates relative to their matched ordinary checkpoints. The smaller moment representation is an approximation change and was evaluated explicitly; the subsequent matrix/kernel/graph changes preserve those predictions. The final acceleration changes no decisions and has at most 1.1e-05 absolute probability discrepancy against its earlier arithmetic implementation.

One-time serving setup (including graph construction, shape warmup and affine caches) is reported separately: GPT-2 / SST-2 ordinary 5.73s, moments 15.08s; RoBERTa / CLINC ordinary 5.66s, moments 15.16s. These steady-state savings apply to repeated use; setup is not free. Data are already tokenized and model weights loaded. Timings include minibatch handling and GPU-to-CPU probability transfers.

For SST-2, this retains the previously specified fixed epoch-3 comparison. The earlier ordinary epoch-1 checkpoint had NLL 0.21775, so the new moment NLL of 0.21489 is only slightly lower than that earlier checkpoint. The strong epoch-3 NLL difference should not be described as beating every possible stopping policy; both original checkpoints remain in the historical tables.

## How firmly is the predictive improvement established?

Paired 95% intervals below resample reporting examples 2,000 times. They condition on these existing single-seed fits. The approximation ablations are exploratory, so these are descriptive intervals, not an independent confirmation after selection.

| Task | Quantity: moments minus ordinary | Difference | Paired 95% interval |
|---|---|---:|---:|
| GPT-2 / SST-2 | accuracy | +0.45872 pp | [-0.80275, +1.83486] pp |
| GPT-2 / SST-2 | nll | -0.05119 | [-0.08207, -0.02284] |
| GPT-2 / SST-2 | ece15 | -3.09534 pp | [-4.42312, -1.05149] pp |
| RoBERTa / CLINC | accuracy | +0.37778 pp | [+0.06667, +0.71111] pp |
| RoBERTa / CLINC | nll | -0.00818 | [-0.01554, -0.00047] |
| RoBERTa / CLINC | ece15 | -0.26548 pp | [-0.74226, +0.27816] pp |

NLL improves with intervals excluding zero on both tasks. SST-2 also resolves an ECE improvement; CLINC resolves an accuracy improvement. SST-2 accuracy and CLINC ECE have favorable point estimates but their intervals include zero. This supports a useful predictive gain at compact inference cost, not a claim that every metric is independently significant or that the result is robust across training seeds.

## Training: exact arithmetic savings, checked through actual updates

The optimized training routine shares identical affine work across antithetic activation nodes; GPT-2 additionally moves its final deterministic affine projection after the carrier combination. The optimizer, loss, regularization, integration-node count and checkpoints are unchanged. Compilation is applied to ordinary training too. Each timed update restores the same saved weights, optimizer and RNG. All final weights are restored again.

| Task | Ordinary ms/step | Original moments ms/step | Optimized moments ms/step | New / ordinary | Original / new |
|---|---:|---:|---:|---:|---:|
| GPT-2 / SST-2 | 60.94 | 304.23 | 171.42 | 2.81x | 1.77x |
| RoBERTa / CLINC | 29.79 | 143.08 | 91.04 | 3.06x | 1.57x |

Each entry is the mean of five-repeat median timings on fixed training minibatches 0, 1 and 2. The retained CSV contains each shape, all per-batch medians, memory and update comparisons. These are measured forward/backward/optimizer steps, **not a newly measured end-to-end fit**. Historical complete fits still took 1,967.6s vs 500.7s ordinary on SST-2 (A100), and 604.4s vs 100.2s on CLINC (L4). Replacing those historical totals with a step-based extrapolation would misstate the evidence.

Maximum norm-relative difference in actual parameter increments across the three batches is 0.034% on SST-2 and 0.217% on CLINC; mean-parameter differences are smaller. These ratios concern the small update vectors, not the full weight norms. The appendix proves the arithmetic rearrangement; floating-point differences are measured rather than assumed absent. CLINC's stored ordinary optimizer has zero final learning rate; its forward, backward and optimizer kernels still execute, but that control is a cost measurement, not a nonzero ordinary-update agreement test.

## What was removed, and what is still approximated?

The implementation first merges fixed adapter means into the backbone affine maps and batches mean/direction multiplication. It fuses elementwise moment operations, uses fused attention where mathematically applicable, and replays fixed-shape GPU graphs. For training it avoids recomputing the same Q/K/V means and antithetic perturbations; GPT-2 also saves three of four final MLP projections. These savings follow exact identities.

The final SST-2 predictor uses one covariance direction with nonlinear Gaussian/cubature propagation. CLINC uses two correlated first-order directions. They are deterministic moment approximations at prediction time, not one or two whole sampled networks. Lower rank does not have a general guarantee of retaining posterior predictions. Every measured alternative, including the unfavorable first-order SST-2 and nonlinear CLINC variants, is saved in `all_ablations.csv`.

## What the ensemble comparison establishes

The current comparison establishes an economical single-belief improvement over the matched ordinary model. A three-member ensemble is a relevant competing method at this cost; speed alone does not establish superiority over it. No new ensemble was trained because the task preserves the no-new-training constraint.

A real ensemble control was possible on the source MRPC task: three already-trained, independent ordinary seeds were combined by averaging probabilities. It obtains 87.25% accuracy, 0.76211 NLL and 10.17% ECE. The previously measured revised moment method averaged 87.91%, 0.33299 and 5.76% across its three source fits. This shows that probability averaging improves the source ordinary models without reproducing their Bayesian NLL gain. However, the source ordinary NLL remains worse than the published baseline, and this is not a matched ensemble test on SST-2 or CLINC. We do not infer transfer-task ensemble superiority from it.

## Reproduction and evidence

`compact_moments.py` contains the serving computation; `efficient_antithetic_transfer.py` and `efficient_antithetic_roberta.py` contain the optimized conditional-moment training blocks. `fused_carrier.py` expresses the exact carrier/affine identity. `benchmark_updates.py` restores state for each update; `final_prediction.py` checks frozen prediction parity and checkpoint equality. `summarize.py` regenerates these tables and figures from saved numerical results. The accompanying mathematics gives the complete affine-commutation proof and states the covariance approximations explicitly.

