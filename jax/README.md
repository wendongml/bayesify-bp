# Bayesian backpropagation in JAX

Companion to *Bayesian Backprop as Belief Propagation: Single-Pass Predictive Uncertainty*,
Wen Dong, NeurIPS 2026. See the [paper page](https://neurips.cc/virtual/2026/poster/154383)
and [citation](../CITATION.cff).

Read the [notebook](README.ipynb), [HTML](README.html), or [compact PDF](README.pdf). The notebook carries its own source and small numerical results; it does not contain checkpoints or datasets.

See the [Colab execution record](COLAB_VALIDATION.md),
[mathematical correspondence](MATH_COVERAGE.md), and
[reader cell map](READER_MAP.md) for the tested source and reading order.

The shared implementation uses JAX mathematical functions, Flax NNX modules and explicit Gaussian moment messages. EP and MFVI differ in their observation objectives and finite updates. Their forward layers are shared. Literal local message formulas and local VJPs are independently checked.

Run the commands below from this `jax/` directory. On a CUDA 12 Colab GPU,
first install `python -m pip install 'jax[cuda12]==0.4.38'`; the standalone
notebook performs this device setup automatically. CPU execution remains
sufficient for the small correctness checks.

```sh
python -m pip install -e '.[tutorial,data]'
PYTHONPATH=src python run_validation.py --figures
PYTHONPATH=src python run_image_suite.py --architectures mlp lenet
PYTHONPATH=src python run_image_suite.py --architectures resnet20 densenet40
PYTHONPATH=src python run_text_suite.py --tasks clinc sst2
```

Full training downloads public assets. A completed checkpoint contains model beliefs, optimizer state, random state and update count. The saved configurations distinguish full schedules from subset execution checks. Test labels do not select CLINC or image checkpoints; SST-2 uses its declared fixed final epoch.

Read `gaussian.py` → `layers.py` → `messages.py` for the common mathematical core. `inference.py` defines objectives and updates. `covariance.py` adds shared activation directions and structured weight beliefs. `models.py` defines MLP, LeNet, ResNet and DenseNet. `transformers.py` defines GPT-2 and RoBERTa, including Gaussian embeddings in the full-weight case. `gaussian_text.py` contains the original Gaussian EP/MFVI update path. `blob.py` and `text_training.py` contain the separately labeled conditional-Gaussian MFVI variant and source-ordered split optimizer.

The numerical conditioning/RTS/EP/CAVI references are independent NumPy/SciPy implementations. Architecture summaries come from the actual Flax modules. ONNX exports describe the mean network; Gaussian state and message updates remain in JAX.

No R figures or PyTorch predictions are used as JAX results. The optional pretrained-model validation script uses Transformers/PyTorch only as an independent backbone reference. It is not a runtime dependency of the JAX inference engine.

The supplied ONNX files open directly in Netron. To regenerate them, use a
**separate export environment**, leaving the pinned training environment intact:

```sh
python3.13 -m venv .venv-export
.venv-export/bin/pip install jax==0.12.0.1 flax==0.12.12 jax2onnx==0.19.0 onnx==1.23.2 onnxruntime==1.31.1
PYTHONPATH=src .venv-export/bin/python export_models.py
PYTHONPATH=src .venv-export/bin/python export_transformers.py
```

These are the versions used for the saved exports, recorded in
`assets/architectures/onnx/export_environment.json`. Each exporter compares its
ONNX output with the executable JAX mean network. Weights are graph inputs;
the architecture downloads contain no trained checkpoint. The notebook itself
does not need the export environment to run its mathematical checks or training.
