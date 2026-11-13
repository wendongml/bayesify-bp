# Reader cell and source map

Generated from the actual notebook. Physical code lines include comments and blanks. Compressed binary payload lines are excluded. Expanded mathematical pseudocode/source excerpts are counted separately from executable cells. This is a reading map, not a claim of matched benchmark protocols.

| Section | Cells | Visible executable lines | Visible excerpt lines | Supporting executable lines | Figure outputs |
|---|---:|---:|---:|---:|---:|
| 1. Gaussian coordinates and shared layers | 4–16 | 0 | 179 | 0 | 0 |
| 2. Gaussian, EP and MFVI message equivalence | 17–29 | 11 | 42 | 0 | 3 |
| 3. Correlation, Monte Carlo fidelity, and computation | 30–46 | 18 | 129 | 0 | 3 |
| 4. MNIST: MLP and LeNet | 47–65 | 22 | 158 | 7 | 5 |
| 5. CIFAR-10: residual and dense connections | 66–71 | 22 | 23 | 0 | 4 |
| 6. GPT-2 LoRA: attention and Gaussian message boundaries | 72–83 | 14 | 76 | 5 | 0 |
| 7. Conditional moment MFVI: GPT-2 and RoBERTa LoRA | 84–90 | 23 | 47 | 0 | 4 |
| 8. Reproduction and code reading | 91–93 | 0 | 0 | 11 | 0 |

Dependencies: Gaussian coordinates → layers and explicit reverse traversal → EP/MFVI heads and finite updates → covariance extensions → architectures, training and shared uncertainty evaluation.

See [the cross-framework audit](../TUTORIAL_PARITY.md) for original-section correspondence, full source counts, and differences in experimental evidence. All supporting implementation code remains available; visible-line counts do not replace total source counts.
