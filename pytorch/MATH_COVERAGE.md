# Mathematical correspondence with the R/TensorFlow reader

Generated from the current notebooks and the inspected topic map in `site/math_topics.json`. Every R Markdown passage containing mathematics is classified. This inventory checks content coverage; numerical identities, approximation fidelity and benchmark outcomes have their own evidence.

| R cell | Topic | Native reader cell(s) | Scope |
|---:|---|---:|---|
| 14 | Local Gaussian factor: the two pipelines in the original tutorial | 7 | derivation |
| 15 | The Gaussian coordinate bridge | 8 | derivation |
| 17 | Layer equations and their literal reverse messages | 12 | derivation |
| 18 | Shared forward layers | 13 | derivation |
| 20 | One explicit reverse traversal | 16 | implementation |
| 24 | Literal messages, projected messages, and custom-gradient carriers | 18 | derivation |
| 25 | Exact Gaussian MLP: activation and weight messages at every layer | 24 | derivation |
| 31 | EP: a normalizer derivative is a tilted-moment change | 19 | derivation |
| 32 | Cavity, site replacement, and the MFVI update schedule | 20 | derivation |
| 33 | MFVI: the same bridge, a different objective and finite update | 19 | derivation |
| 40 | A literal layer calculation, checked against the shared VJP | 13 | derivation |
| 41 | The nonlinear check uses one **64 → 12 → 8 → 10 ReLU moment MLP**, 16 fixed | 30 | experiment |
| 47 | Diagonal activations and shared covariance directions | 33 | derivation |
| 52 | Sampling-assisted moment propagation: integrate the nonlinear remainder | 35 | derivation |
| 55 | Readout variance messages: conditioning before approximation | 46 | derivation |
| 58 | What the R/TensorFlow repair established | 43 | experiment |
| 66 | What the forward pass computes | 50 | derivation |
| 68 | Retaining the correlations that reach the classifier | 53 | derivation |
| 70 | Optimization updates | 57 | derivation |
| 76 | Predictive averaging and uncertainty | 64 | derivation |
| 77 | Metric definitions shared by every benchmark | 64 | derivation |
| 79 | Full training protocol | 72 | protocol |
| 87 | Dirty-MNIST asks a different question | 64 | derivation |
| 92 | EP: explicit sites and the scalable tied energy | 57 | derivation |
| 93 | Native Keras summaries and complete architecture downloads | 82 | inspection |
| 111 | Learning curves: confidence and elapsed training time | 86 | experiment |
| 116 | Conditional Gaussian carriers and probability averaging | 101 | derivation |
| 122 | GPT-2 with LoRA: SST-2 sentiment classification | 96 | experiment |
| 157 | **Results.** Moment EP reaches **92.32% accuracy / 0.22336 NLL**, | 96 | experiment |
| 161 | RoBERTa-base with LoRA: CLINC150 intent classification | 109 | protocol |
| 158 | Independent replay of the same Gaussian adapters | 97 | experiment |

See [the full correspondence and evidence](../TUTORIAL_PARITY.md) for method families, matched-belief checks and computational scope.
