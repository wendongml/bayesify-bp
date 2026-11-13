# Fixed-checkpoint efficiency evaluation

Goal: reduce the revised method's actual training-step and prediction cost
while retaining its Bayesian predictive benefit. No new fits, hyperparameter
search, OOD dataset, or checkpoint selection is part of this experiment.

Use the existing ordinary and revised SST-2 epoch-3 and CLINC selected epoch-6
checkpoints. Restore the released software stack and immutable artifact
generations. All comparisons use the same GPU, precision and input batch;
ordinary inference receives the same applicable kernel optimizations.

1. Reproduce baseline predictions and measure the current cost. Profile a
   representative batch. Test merged affine computations and GPU-friendly
   moment operators with numerical parity against the current implementation.
2. Evaluate compact Gaussian covariance propagation with one, two and four
   directions, including an explicitly first-order ablation. These change the
   approximation, so retain every outcome and compare probabilities, NLL,
   accuracy and ECE to the unchanged K8/K32 predictions. Use development or
   stopping examples for method selection; reserve reporting sets for the
   frozen finalists. Do not tune a calibration temperature on reporting labels.
3. If needed, test a conditional-moment activation carrier with a small fixed
   quadrature budget. It is not an ensemble of separately trained networks.
   Label stochastic and deterministic constructions accurately.
4. Time exact training-step replays only after inference screening. Restore
   identical checkpoint/optimizer/RNG state for each replay and compare actual
   parameter updates. A step speedup is not a measured full-fit time. No
   optimized replay becomes a new trained checkpoint.
5. Compute an ensemble comparator only from existing independently trained,
   compatible ordinary checkpoints. Do not manufacture an ensemble from
   duplicated weights, different epochs, or incompatible training protocols.

Primary cost target: at most 3x the matched ordinary implementation. Primary
quality target: retain the existing predictive improvement without degrading
accuracy, NLL or ECE materially. Report each metric rather than declaring a
win from a selected metric. Paired intervals condition on the existing fits.

Final timing confirmation: after exact affine-sharing and carrier-commutation
checks, replay the fixed first three training minibatches (indices 0, 1, 2)
from the same restored states. Reuse the same antithetic affine savings for
RoBERTa. This broadens input-shape coverage without tuning a model, changing
the learned belief, or continuing training. Keep all per-batch costs and
actual-update discrepancies; do not extrapolate them into measured fit times.
