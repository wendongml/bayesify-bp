"""Consistent model summaries and standards-based architecture exports.

``torchinfo`` provides the PyTorch counterpart of ``keras.Model.summary``.
ONNX stores the executable mean-network graph, and Netron renders the actual
dependencies (including residual and dense edges).  Neither path participates
in inference; they inspect the same modules used by the deterministic forward
pass.
"""

from __future__ import annotations

from pathlib import Path
from collections import Counter
from collections.abc import Mapping
import hashlib
import inspect
import json

import torch
from torch import Tensor, nn

from .layers import GaussianParameter


def fx_graph(model: nn.Module) -> torch.fx.GraphModule:
    """Return PyTorch's dependency-free, auditable module graph."""
    return torch.fx.symbolic_trace(model)


def model_summary(model: nn.Module, example: Tensor | Mapping[str, Tensor]) -> str:
    """Return one Keras-like ``torchinfo`` summary format for every model.

    ``Param #`` counts all trainable Bayesian scalars.  A diagonal Gaussian
    weight therefore contributes one mean and one log-variance scalar.  Use
    :func:`parameter_summary` to display that split explicitly.
    """
    try:
        from torchinfo import summary
    except ImportError as error:
        raise RuntimeError("Install bayesify-torch[inspect] for model summaries") from error
    text = str(
        summary(
            model,
            input_data=example,
            depth=3,
            col_names=("output_size", "num_params"),
            row_settings=("var_names",),
            verbose=0,
        )
    )
    # torchinfo cannot infer MAC or memory accounting for our custom moment
    # modules and reports zeros for those optional estimates.  Keras summary
    # does not print misleading unavailable estimates, so stop after the real
    # trainable/non-trainable parameter totals.
    lines = text.splitlines()
    kept = []
    for line in lines:
        if line.startswith("Total mult-adds"):
            break
        kept.append(line)
    if kept and kept[-1] != lines[0]:
        kept.append(lines[0])
    return "\n".join(kept)


def keras_like_summary(model: nn.Module, example: Tensor | Mapping[str, Tensor]) -> str:
    """Backward-compatible alias for :func:`model_summary`."""

    return model_summary(model, example)


def parameter_summary(model: nn.Module) -> dict[str, int]:
    """Split trainable scalars into Gaussian means, log variances, and others."""

    beliefs = [module for module in model.modules() if isinstance(module, GaussianParameter)]
    means = sum(module.mean.numel() for module in beliefs)
    log_variances = sum(module.log_variance.numel() for module in beliefs)
    total = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "Gaussian mean parameters": means,
        "Gaussian log-variance parameters": log_variances,
        "other trainable parameters": total - means - log_variances,
        "total trainable scalars": total,
    }


def export_onnx(
    model: nn.Module,
    example: Tensor,
    path: str | Path,
    *,
    optimize: bool = True,
    opset: int = 18,
) -> dict[str, object]:
    """Export, check, and describe the executable mean network as ONNX.

    Current PyTorch releases use the recommended dynamo exporter.  The legacy
    branch keeps the tutorial runnable on PyTorch 2.1, whose ONNX exporter
    predates the ``dynamo`` argument.  ``onnxoptimizer`` is used only to remove
    display-only identity nodes; it cannot alter inference or training.
    """

    try:
        import onnx
    except ImportError as error:
        raise RuntimeError("Install bayesify-torch[inspect] for ONNX export") from error

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    signature = inspect.signature(torch.onnx.export)
    modern = "dynamo" in signature.parameters
    # PyTorch 2.1's legacy exporter emits ReduceMean.axes as an attribute.
    # ONNX moved axes to an input in opset 18; the old graph is valid at 17.
    # Current dynamo exports retain the requested opset.
    if not modern:
        opset = min(opset, 17)
    common = {
        "input_names": ["input"],
        "output_names": ["logits"],
        "opset_version": opset,
    }
    with torch.no_grad():
        if modern:
            torch.onnx.export(
                model,
                (example,),
                str(path),
                dynamo=True,
                external_data=False,
                **common,
            )
        else:
            torch.onnx.export(
                model,
                example,
                str(path),
                dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                do_constant_folding=True,
                **common,
            )

    graph = onnx.load(path)
    if optimize:
        try:
            import onnxoptimizer
        except ImportError:
            pass
        else:
            available = set(onnxoptimizer.get_available_passes())
            passes = [
                name
                for name in (
                    "eliminate_identity",
                    "eliminate_nop_dropout",
                    "eliminate_nop_flatten",
                    "eliminate_nop_pad",
                    "eliminate_nop_transpose",
                    "eliminate_unused_initializer",
                )
                if name in available
            ]
            graph = onnxoptimizer.optimize(graph, passes)
            onnx.save(graph, path)
    onnx.checker.check_model(graph, full_check=True)
    operators = Counter(node.op_type for node in graph.graph.node)
    runtime_error = None
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        pass
    else:
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        with torch.no_grad():
            expected = model(example).detach().cpu().numpy()
        actual = session.run(None, {session.get_inputs()[0].name: example.detach().cpu().numpy()})[0]
        runtime_error = float(np.max(np.abs(expected - actual)))
    receipt = {
        "path": path.as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
        "exporter": "torch.onnx dynamo" if modern else "torch.onnx legacy compatibility",
        "opset": opset,
        "nodes": len(graph.graph.node),
        "operators": dict(sorted(operators.items())),
        "onnxruntime_max_abs_error": runtime_error,
        **parameter_summary(model),
    }
    return receipt


def write_architecture_catalog(
    models: Mapping[str, tuple[nn.Module, Tensor]], directory: str | Path
) -> Path:
    """Write ONNX files, uniform summaries, and a machine-readable receipt."""

    directory = Path(directory)
    onnx_directory = directory / "onnx"
    summary_directory = directory / "summaries"
    onnx_directory.mkdir(parents=True, exist_ok=True)
    summary_directory.mkdir(parents=True, exist_ok=True)
    records = []
    for slug, (model, example) in models.items():
        summary_path = summary_directory / f"{slug}.txt"
        summary_path.write_text(model_summary(model, example) + "\n")
        onnx_path = onnx_directory / f"{slug}.onnx"
        record = export_onnx(model, example, onnx_path)
        record["path"] = onnx_path.relative_to(directory).as_posix()
        record["name"] = slug
        record["summary"] = summary_path.relative_to(directory).as_posix()
        records.append(record)
    manifest = directory / "architecture_manifest.json"
    manifest.write_text(json.dumps(records, indent=2) + "\n")
    return manifest
