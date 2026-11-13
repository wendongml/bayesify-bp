# Numerical results

These files contain the reported metrics, training histories, and the numerical
coordinates used in the selected plots. They contain no trained parameters,
optimizer state, datasets, or complete prediction-draw tensors.

- `displayed_tables/`: verbatim table values from the notebook, with a cell index.
- `mnist/`, `gpt2_selected_campaign/`, `compact_transformers/`: metric summaries,
  per-seed comparisons, settings and timing data.
- `plot_data/`: exact reliability-bin, selective-risk, ROC and empirical-CDF
  counts/coordinates, computed with the released uncertainty helpers; no curve smoothing.
- `approximation_cost/`, `message_equivalence/`, `shared_layers/`: approximation
  summaries and independent inference-identity checks.
- Top-level CSVs: the selected CIFAR and transformer comparisons and histories.

The notebook displays the stored benchmark outputs in compact mode. Full training
switches regenerate per-example predictions and run the original metric/plot
functions. Removing bulk prediction arrays changes the distribution contents,
not the experimental results or training implementation.

To recover curve coordinates without rounding: within each method's selective-risk
file, let `k <- seq_len(nrow(d))`; coverage is `k / length(k)` and risk is
`d$cumulative_errors / k`. ROC coordinates are `false_positives / max(false_positives)`
and `true_positives / max(true_positives)`. These integer counts reproduce the
original plotted coordinates exactly while avoiding repeated decimal strings.
