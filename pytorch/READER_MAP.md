# Reader cell and source map

Generated from the actual notebook. Physical code lines include comments and blanks. Compressed binary payload lines are excluded. Expanded mathematical pseudocode/source excerpts are counted separately from executable cells. This is a reading map, not a claim of matched benchmark protocols.

| Section | Cells | Visible executable lines | Visible excerpt lines | Supporting executable lines | Figure outputs |
|---|---:|---:|---:|---:|---:|
| 1. Gaussian coordinates and shared layer maps | 6–16 | 284 | 0 | 0 | 0 |
| 2. Gaussian, EP and MFVI: message passing equals backpropagation | 17–31 | 56 | 65 | 14 | 3 |
| 3. Approximation quality versus Monte Carlo and computation | 32–48 | 87 | 43 | 25 | 3 |
| 4. MNIST: MLP and LeNet with common uncertainty diagnostics | 49–79 | 193 | 222 | 898 | 7 |
| 5. CIFAR-10: ResNet-20 and DenseNet-40 | 80–87 | 112 | 0 | 0 | 4 |
| 6. GPT-2 with rank-4 Q/V LoRA on SST-2 | 88–99 | 24 | 63 | 8 | 1 |
| 7. Conditional moment MFVI: GPT-2 and RoBERTa LoRA | 100–111 | 16 | 88 | 5 | 2 |
| 8. Validation, reproduction and reading order | 112–114 | 0 | 0 | 11 | 0 |

Dependencies: Gaussian coordinates → layers and explicit reverse traversal → EP/MFVI heads and finite updates → covariance extensions → architectures, training and shared uncertainty evaluation.

See [the cross-framework audit](../TUTORIAL_PARITY.md) for original-section correspondence, full source counts, and differences in experimental evidence. All supporting implementation code remains available; visible-line counts do not replace total source counts.
