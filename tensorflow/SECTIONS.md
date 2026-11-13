# Reader cell and source map

Generated from the actual notebook. Physical code lines include comments and blanks. Compressed binary payload lines are excluded. Expanded mathematical pseudocode/source excerpts are counted separately from executable cells. This is a reading map, not a claim of matched benchmark protocols.

| Section | Cells | Visible executable lines | Visible excerpt lines | Supporting executable lines | Figure outputs |
|---|---:|---:|---:|---:|---:|
| 1. Overview and Gaussian computations | 1–22 | 233 | 0 | 495 | 0 |
| 2. Gaussian, EP and MFVI: message passing equals backpropagation | 23–45 | 613 | 0 | 183 | 3 |
| 3. Approximation quality versus Monte Carlo and computation | 46–61 | 379 | 0 | 69 | 3 |
| 4. MNIST: MLP and LeNet, with common uncertainty diagnostics | 62–89 | 626 | 0 | 387 | 7 |
| 5. CIFAR-10: shared features in ResNet and DenseNet | 90–114 | 181 | 5 | 1093 | 3 |
| 6. GPT-2 with LoRA: SST-2 | 115–159 | 304 | 0 | 2142 | 1 |
| 7. RoBERTa with LoRA: CLINC150 | 160–166 | 136 | 0 | 0 | 1 |
| 8. Results and reproduction | 167–168 | 0 | 0 | 0 | 0 |

Dependencies: Gaussian coordinates → layers and explicit reverse traversal → EP/MFVI heads and finite updates → covariance extensions → architectures, training and shared uncertainty evaluation.

See [the cross-framework audit](../TUTORIAL_PARITY.md) for original-section correspondence, full source counts, and differences in experimental evidence. All supporting implementation code remains available; visible-line counts do not replace total source counts.
