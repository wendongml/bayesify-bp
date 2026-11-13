# ONNX and Netron architecture views

`../../export_architectures.py` constructs the four models, writes one
`torchinfo` summary format, exports their executable mean networks through
`torch.onnx.export`, removes display-only identity nodes, checks each ONNX
model, and compares its output with the live PyTorch model through ONNX
Runtime.

Open a graph interactively with either the desktop/CLI application or the
browser application:

```bash
netron onnx/cifar_resnet20.onnx
```

or drag the `.onnx` file into <https://netron.app>. The checked ONNX model is
the primary artifact. The PNG and SVG files are Netron 9.3.5 exports with
weights hidden and horizontal layout; their hashes and dimensions are recorded
in `netron_manifest.json`.

The ONNX graph represents the deterministic Gaussian-mean forward pass.
Gaussian log variances, activation moments, and backward messages are native
PyTorch state and are intentionally not misrepresented as ONNX dataflow nodes.
