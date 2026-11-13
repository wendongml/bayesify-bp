# Execution validation

The current PyTorch reader was executed from an empty directory in **Consumer
Colab L4 on 12 November 2026**. It reconstructed its embedded native sources,
ran its numerical checks, and regenerated tables and figures from numerical
records. The executed notebook is the source of the HTML and separately typeset
PDF. No checkpoint is included in the reader.

| Check | Result |
|---|---|
| Notebook cells | 114 |
| Executable cells | 36/36 completed; no error outputs |
| Numerical suite | 70 passed; one expected release-sidecar check skipped (71/71 pass locally) |
| Mathematical counterparts | All 31 R mathematical passages mapped in `MATH_COVERAGE.md` |
| Native source correspondence | Embedded modules equal the released files byte for byte |
| Complete new image fits | Five Dirty-MNIST methods, 30 epochs each, seed 41 |
| Learned ResNet comparison | 80 same-math checks across R/TensorFlow, PyTorch and JAX; maximum absolute difference 1.82e-13 |
| Gaussian GPT-2 comparison | Eight PyTorch/JAX prediction and likelihood-derivative checks pass |

The notebook's default switches run numerical validation and regenerate
recorded evidence; full training and public downloads are separate explicit
switches. Replaying figures is not counted as training. The ten native
Dirty-MNIST runs were completed before the final standalone executions and
retain their configurations, learning curves, predictions and measured costs.
Their independent random streams and single seed do not establish a framework
quality ranking.

`validation_receipt.json` records the exact notebook hash, versions, counts and
scope. `../site/tutorial_validation.json` checks all three readers together.
The same-belief approximation study in [the evidence directory](../site/evidence/README.md)
separates code agreement from forward/backward approximation error against a
524,288-network reference. Historical image/transformer fits retain their
original implementation attribution; the fixed-belief text check is not new
text training.

The skipped Colab test requires the repository-side ONNX catalog, which is absent
from a single-notebook upload. The notebook instead exports and validates the
actual models. That catalog check also passes in the complete local repository.
