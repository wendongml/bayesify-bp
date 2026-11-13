"""Build the full standalone PyTorch tutorial from tested source and evidence."""

from __future__ import annotations

import ast
import base64
import io
import json
from pathlib import Path
import textwrap
import zipfile


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
SOURCE = ROOT / "src" / "bayesify_torch"


def derivation(name):
    """Insert the reviewed mathematics immediately before its implementation."""
    text = (ROOT / "tutorial_derivations.md").read_text()
    return text.split(f"<!-- {name} -->", 1)[1].split("<!--", 1)[0].strip()


def markdown(text: str, cell_id: str | None = None) -> dict:
    cell = {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(text).strip() + "\n"}
    if cell_id:
        cell["id"] = cell_id
    return cell


def code(
    text: str,
    *,
    hidden: bool = False,
    collapsed: bool = False,
    outputs: list | None = None,
    cell_id: str | None = None,
    label: str = "Python / PyTorch",
) -> dict:
    metadata = {"code_label": label}
    if hidden:
        metadata["tags"] = ["hide-input"]
    elif collapsed:
        metadata["tags"] = ["collapse-input"]
    if hidden or collapsed:
        metadata["jupyter"] = {"source_hidden": True}
        metadata["cellView"] = "form"
        text = "#@title " + label + " (expand to inspect)\n" + textwrap.dedent(text).strip()
    cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": metadata,
        "outputs": list(outputs or []),
        "source": textwrap.dedent(text).strip() + "\n",
    }
    if cell_id:
        cell["id"] = cell_id
    return cell


def source_of(path: str, names: tuple[str, ...]) -> str:
    text = (SOURCE / path).read_text()
    tree = ast.parse(text)
    found = {
        node.name: ast.get_source_segment(text, node)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    }
    missing = set(names) - set(found)
    if missing:
        raise RuntimeError(f"missing from {path}: {sorted(missing)}")
    return "\n\n".join(found[name] for name in names)


def source_details(title: str, text: str, cell_id: str | None = None, *, key=False) -> dict:
    """A readable, collapsed source excerpt that is not executed twice."""

    if key:
        cell = markdown(f"**{title}**\n\n```python\n{text}\n```", cell_id)
    else:
        cell = markdown(f"<details><summary><strong>{title}</strong></summary>\n\n"
                        f"```python\n{text}\n```\n\n</details>", cell_id)
    cell["metadata"] = {"tags": ["key-source" if key else "support-source"],
                        "source_label": title}
    return cell


def file_excerpt(relative, names):
    """Exact function/method bodies from the implementation, without rewriting."""
    text = (REPO / relative).read_text()
    tree = ast.parse(text)
    found = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            found[node.name] = node
            if isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, ast.FunctionDef):
                        found[node.name + "." + child.name] = child
    snippets = []
    lines = text.splitlines()
    for name in names:
        node = found[name]
        first = min([node.lineno] + [d.lineno for d in node.decorator_list])
        snippets.append(textwrap.dedent("\n".join(lines[first-1:node.end_lineno])))
    return "\n\n".join(snippets)


def embedded_payload() -> str:
    memory = io.BytesIO()
    with zipfile.ZipFile(memory, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(SOURCE.rglob("*")):
            if path.is_file() and (path.suffix in (".py", ".npz")) and "__pycache__" not in path.parts:
                archive.writestr("src/bayesify_torch/"+path.relative_to(SOURCE).as_posix(), path.read_bytes())
        for name in ("run_training.py", "learned_fidelity.py", "head_ablation.py", "readout_correction.py"):
            archive.writestr(name, (ROOT/name).read_bytes())
        for folder in (REPO / "tensorflow" / "python" / "gpt2", REPO / "tensorflow" / "python" / "blob_mirror"):
            for path in sorted(folder.glob("*")):
                if path.is_file():
                    archive.writestr(path.relative_to(REPO / "tensorflow").as_posix(), path.read_bytes())
        for path in sorted((ROOT / "tests").glob("*.py")):
            archive.writestr(f"pytorch/tests/{path.name}", path.read_bytes())
        for path in sorted((ROOT / "evidence").rglob("*")):
            if path.is_file() and "matched" not in path.relative_to(ROOT / "evidence").parts:
                archive.writestr("evidence/" + path.relative_to(ROOT / "evidence").as_posix(), path.read_bytes())
        archive.writestr("evidence/matched/compare_approximations.py", (REPO/"site/compare_approximations.py").read_bytes())
        archive.writestr("evidence/matched/compare_text_beliefs.py", (REPO/"site/compare_text_beliefs.py").read_bytes())
        for path in sorted((REPO/"site/evidence").glob("*")):
            if path.suffix in (".npz", ".json", ".csv"):
                archive.writestr("evidence/matched/evidence/"+path.name,path.read_bytes())
        # Small Netron previews make the single copied notebook readable.  The
        # ONNX models themselves are regenerated from the live PyTorch modules.
        for path in sorted((ROOT / "assets" / "architectures" / "netron").glob("*.png")):
            archive.writestr(f"assets/architectures/netron/{path.name}", path.read_bytes())
    return base64.b64encode(memory.getvalue()).decode()


payload = embedded_payload()

geometry = source_of(
    "geometry.py",
    ("encode_diagonal", "decode_diagonal", "encode_full", "decode_full", "tilted_update", "canonical_mfvi_update"),
)
layers = source_of(
    "layers.py",
    ("dense_moments", "dense_literal_backward", "conv2d_moments", "relu_terms", "relu_literal_backward"),
)
graph = source_of("graph.py", ("GraphPass", "MomentDAG"))
heads = source_of("heads.py", ("categorical_quadrature_head",))
correlation = source_of(
    "correlation.py", ("SharedMoments", "StructuredGaussian", "relu_shared", "gaussian_readout")
)
optimizers = source_of("optimizers.py", ("power_cavity", "Vadam", "EPAdam"))
training = source_of(
    "training.py",
    (
        "moment_likelihood_gradients",
        "sampled_likelihood_gradients",
        "moment_ep_gradients",
        "sampled_ep_gradients",
        "moment_vadam_second",
        "sampled_predict",
        "moment_predict",
    ),
)
benchmark = source_of("benchmark.py", ("TrainingConfig", "GaussianTrainer", "evaluate_batches"))
models = source_of("models.py", ("TinyResidualMLP", "MomentMLP", "MomentLeNet"))
pooling = source_of("layers.py", ("gaussian_max_pair", "_max_pair_adjoint", "max_pool_moments",
                                  "max_pool_literal_backward", "MomentMaxPool2d"))
architectures = source_of(
    "architectures.py", ("_conv", "MomentCifarNet", "resnet20", "densenet40")
)
metrics = source_of(
    "metrics.py",
    ("Ranking", "_probabilities", "reliability", "ranking_curve", "uncertainty_metrics", "ood_metrics"),
)
plotting = source_of(
    "plotting.py",
    ("_square", "agreement_plot", "calibration_plot", "risk_coverage_plot", "ood_roc_plot", "uncertainty_panel"),
)
gaussian_chain = source_of(
    "gaussian_chain.py",
    (
        "forward_gaussian_chain",
        "smooth_gaussian_chain",
        "condition_gaussian_chain",
        "posterior_signal",
        "backprop_gaussian_chain",
        "install_full",
    ),
)
tutorial_evidence = source_of(
    "tutorial_evidence.py",
    ("exact_gaussian_equivalence", "nonlinear_message_equivalence", "approximation_benchmark"),
)
inspection = source_of(
    "inspect.py", ("model_summary", "parameter_summary", "export_onnx")
)

cells: list[dict] = []
add = cells.append

add(markdown(r"""
# Bayesian backpropagation in PyTorch

This is a native PyTorch implementation and executable tutorial. The semantic
and approximation experiments run in this notebook. Benchmark tables and
figures are recomputed in Python from saved per-example predictions, with
configurations and timing receipts. The native Dirty-MNIST chapter includes new
full fits; other benchmark chapters retain their original run provenance. No R-generated plot is presented as a PyTorch result. The notebook
follows one scientific path throughout:

1. encode Gaussian Euclidean adjoints as moment-space messages;
2. propagate moments through ordinary `nn.Module` layers;
3. transport a method-specific EP or MFVI signal backward on a frozen snapshot;
4. update the persistent Gaussian beliefs only after the reverse traversal;
5. evaluate Bayesian predictions by averaging probabilities.

The exact Gaussian example supplies an analytic positive control. EP and MFVI
then use different objectives on the same nonlinear graph. A fixed-belief
experiment finally measures approximation error and computation against a
large reparameterized Monte Carlo calculation. The benchmark chapters apply
the same uncertainty metrics to MNIST, CIFAR-10, SST-2 and CLINC.

| Section | Question answered | Main code |
|---|---|---|
| 1 | How do ordinary derivatives encode Gaussian messages? | `geometry`, `layers`, `graph` |
| 2 | Do exact, EP and MFVI message calculations agree with backprop? | `gaussian_chain`, `heads`, `inference` |
| 3 | What is lost by diagonal moments or a small covariance rank? | `correlation`, `tutorial_evidence` |
| 4 | How are beliefs trained and uncertainty evaluated on MNIST? | `optimizers`, `training`, `metrics` |
| 5 | What changes for residual/dense CNNs on CIFAR-10? | `architectures` |
| 6 | How do Gaussian Q/V adapters work for GPT-2/SST-2? | `python/gpt2` |
| 7 | How do conditional moment carriers work for RoBERTa/CLINC? | `python/blob_mirror` |
| 8 | Which claims can be reproduced, and from which evidence? | tests, manifest, reproduction map |

**Reading the code.** The expanded cells show the mathematical operations,
model builders and short experiment calls. Cells labelled “support” can be
expanded to inspect installation, plotting and complete run routines; their
outputs remain beside the relevant explanation. Exact source excerpts are
labelled as excerpts, and the complete executable source stays embedded in
this single notebook. The PDF prints the key code and all mathematics/results;
support remains accessible in the notebook, HTML and source files.
""", "overview"))

add(code(f"""
# Standalone support. Copy only this notebook to Colab and run it.
import ast, base64, contextlib, copy, html, importlib.util, io, json, math, os, pathlib, subprocess, sys, tempfile, time, zipfile, warnings
from dataclasses import dataclass
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
_here = pathlib.Path.cwd()
if (_here / "pytorch" / "src" / "bayesify_torch").exists():
    _repo = _here
    sys.path.insert(0, str(_here / "pytorch" / "src"))
elif (_here / "src" / "bayesify_torch").exists():
    _repo = _here.parent
    sys.path.insert(0, str(_here / "src"))
else:
    _repo = pathlib.Path(tempfile.mkdtemp(prefix="bayesify_pytorch_"))
    zipfile.ZipFile(io.BytesIO(base64.b64decode("{payload}"))).extractall(_repo)
    sys.path.insert(0, str(_repo / "src"))

import numpy as np, pandas as pd, matplotlib.pyplot as plt, torch
import torch.nn.functional as F
from torch import Tensor, nn
from IPython.display import HTML, Image, display
from bayesify_torch import *
from bayesify_torch.geometry import add_signals
from bayesify_torch.geometry import _require_nonnegative, _require_positive
from bayesify_torch.graph import GraphMessages, MomentDAG, Node, RecordedNode
from bayesify_torch.layers import *
from bayesify_torch.layers import LayerMessages, LocalRecord
from bayesify_torch.models import *
from bayesify_torch.architectures import *
from bayesify_torch.plotting import *
from bayesify_torch.optimizers import PowerCavity
from bayesify_torch.heads import HeadResult
from bayesify_torch.correlation import RootMoments, FactorMessages
from bayesify_torch.training import (normalization_observations, average_observations,
    EPGradients, MomentGradients, SampledGradients,
    _MomentPass, _cavity_replacements)
torch.set_default_dtype(torch.float64)
torch.manual_seed(41)
torch.set_num_threads(2)
warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive.*")
_reference_repo = _repo / "tensorflow" if (_repo / "tensorflow/python").exists() else _repo
print("PyTorch", torch.__version__, "| native standalone sources loaded")
""", hidden=True, cell_id="standalone-setup"))

add(code(r"""
# The summary/export packages are inspection tools, not inference dependencies.
required = {"torchinfo": "torchinfo>=1.8", "onnx": "onnx>=1.14",
            "onnxoptimizer": "onnxoptimizer>=0.3.13", "onnxruntime": "onnxruntime>=1.16",
            "transformers": "transformers==4.48.3",
            "peft": "peft==0.14.0", "accelerate": "accelerate>=0.33"}
import inspect
if "dynamo" in inspect.signature(torch.onnx.export).parameters:
    required["onnxscript"] = "onnxscript>=0.1"
missing = [package for module, package in required.items() if importlib.util.find_spec(module) is None]
# Pin the inspected Transformers/PEFT pair: newer masking primitives can
# require a different ONNX exporter even when native forward inference works.
from importlib.metadata import version
for module in ("transformers","peft"):
    if importlib.util.find_spec(module) is not None and version(module) != required[module].split("==")[1]:
        missing.append(required[module])
if missing:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)
# Some Colab images combine new PEFT with an obsolete optional torchao.
# PEFT checks it even for unquantized LoRA. Repair that dependency only when
# present and incompatible; this does not alter the model or inference math.
from importlib.metadata import version, PackageNotFoundError
from packaging.version import Version
try:
    if Version(version("peft")) >= Version("0.21") and Version(version("torchao")) < Version("0.16"):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "--no-deps", "torchao>=0.16"], check=True)
except PackageNotFoundError:
    pass

_architecture_work = pathlib.Path(tempfile.mkdtemp(prefix="bayesify_onnx_"))
_architecture_assets = ((_repo / "pytorch" / "assets" / "architectures")
                        if (_repo / "pytorch" / "assets").exists()
                        else (_repo / "assets" / "architectures"))

def architecture_link(slug, generated):
    # Sidecar links for the website; self-contained download for one copied notebook.
    packaged = _architecture_assets / "onnx" / f"{slug}.onnx"
    href = (f"assets/architectures/onnx/{slug}.onnx" if packaged.exists() else
            "data:application/octet-stream;base64," + base64.b64encode(pathlib.Path(generated).read_bytes()).decode())
    return f"<a download='{slug}.onnx' href='{html.escape(href)}'>Download complete ONNX graph for Netron</a>"

def show_architecture(name, slug, model, example, *, inline=True):
    '''One summary/export protocol for every self-contained image model.'''
    # Float32 is the deployment/export dtype even when a strict numerical
    # teaching fixture elsewhere in the notebook uses float64.
    inspected = copy.deepcopy(model).float().eval()
    inspected_example = example.float()
    # The current exporter prints a multi-line progress trace even when
    # ``verbose=False``.  Keep the tutorial output focused on the checked
    # receipt; an export exception still escapes this context unchanged.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        receipt = export_onnx(inspected, inspected_example, _architecture_work / f"{slug}.onnx")
    counts = parameter_summary(inspected)
    display(pd.DataFrame([{ "model": name, "input": tuple(example.shape),
        "output": tuple(inspected(inspected_example).shape), "mean parameters": counts["Gaussian mean parameters"],
        "log-variance parameters": counts["Gaussian log-variance parameters"],
        "ONNX nodes": receipt["nodes"], "ONNX max |error|": receipt["onnxruntime_max_abs_error"] }]))
    text = model_summary(inspected, inspected_example)
    display(HTML("<details><summary><b>Complete torchinfo layer summary</b></summary><pre>"
                 + html.escape(text) + "</pre></details>"))
    preview = _architecture_assets / "netron" / f"{slug}.png"
    if inline:
        display(Image(filename=str(preview)))
    else:
        encoded = base64.b64encode(preview.read_bytes()).decode()
        display(HTML("<details><summary><b>Scrollable full Netron graph</b></summary>"
          "<div style='overflow-x:auto'><img style='max-width:none;height:180px' src='data:image/png;base64,"
          + encoded + "'></div></details>"))
    display(HTML(architecture_link(slug, receipt["path"])))
    return receipt
""", hidden=True, cell_id="architecture-inspection-setup"))

add(markdown(r"""
## Code map

The reusable implementation is ordinary Python modules under
`src/bayesify_torch`. The table is computed from those source files in this
standalone notebook payload.
"""))
add(code(r"""
rows = []
source_root = (_repo / "src" / "bayesify_torch") if (_repo / "src").exists() else (_repo / "pytorch" / "src" / "bayesify_torch")
for path in sorted(source_root.glob("*.py")):
    tree = ast.parse(path.read_text())
    definitions = [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    rows.append({"module": path.name, "lines": len(path.read_text().splitlines()),
                 "top-level definitions": len(definitions), "examples": ", ".join(definitions[:4])})
display(pd.DataFrame(rows))
""", cell_id="native-code-map"))

add(markdown(r"""
## 1. Gaussian coordinates and shared layer maps

For $q(u)=\mathcal N(m,C)$, $M=C+mm^\top$, and a maximized scalar objective
$\Phi$, let $g_m=\partial_m\Phi|_C$ and symmetric
$G_C=\partial_C\Phi|_m$. The Gaussian Fisher map is

$$d_m=Cg_m,\qquad d_M=md_m^\top+d_mm^\top+2CG_CC.$$

For diagonal $C=\operatorname{Diag}v$, this becomes
$d_m=v\odot g_m$ and
$d_s=2m\odot d_m+2v^{\odot2}\odot g_v$. Cholesky solves implement the inverse
full-covariance map; deterministic coordinates carry ordinary adjoints.
"""))
add(markdown(derivation("gaussian"), "gaussian-derivation"))
add(code("from bayesify_torch.geometry import *\n\n" + geometry, cell_id="gaussian-geometry"))
add(code(r"""
q = DiagonalMoments(torch.tensor([-.3, .4]), torch.tensor([.2, 1.1]))
g = EuclideanAdjoint(torch.tensor([.7, -.5]), torch.tensor([.2, -.1]))
recovered = decode_diagonal(q, encode_diagonal(q, g))
print("maximum diagonal round-trip error:",
      max((recovered.mean-g.mean).abs().max().item(),
          (recovered.variance-g.variance).abs().max().item()))
"""))

add(markdown(r"""
For $Y=XA^\top+b$, independence of the input and weight beliefs gives

$$E[Y]=mA^\top+m_b,\qquad
\operatorname{Var}(Y)=v(A^2)^\top+(m^2+v)S^\top+v_b.$$

These product-moment identities do not assume that the input distribution is
Gaussian. The handwritten reverse equations below are checked independently
against a local `torch.func.vjp` of this same map.
"""))
add(markdown(derivation("literal_dense"), "literal-dense-derivation"))
add(code("from bayesify_torch.layers import *\n\n" + layers, cell_id="shared-layers"))
add(markdown(r"""
The graph object only caches local maps and sums branch contributions. It does
not choose EP, MFVI, an optimizer, or a covariance approximation.
"""))
add(code("from bayesify_torch.graph import *\n\n" + graph, cell_id="explicit-graph"))

add(markdown(r"""
# 2. Gaussian, EP and MFVI: message passing equals backpropagation

EP differentiates a tilted normalizer. MFVI differentiates an expected log
factor. They can use the same layer maps, but their heads and finite updates
must not be interchanged.

For EP, $m^+=m+\rho d_m$, $M^+=M+\rho d_M$ and
$C^+=M^+-m^+(m^+)^\top$. For MFVI, the diagonal canonical update is

$$\tau^+=(1-\rho)\tau+\rho(\tau_0-2g_v),\quad
h^+=(1-\rho)h+\rho(h_0+g_m-2m\odot g_v).$$
"""))
add(markdown(derivation("ep"), "ep-mfvi-derivation"))
add(markdown(derivation("gaussian_heads"), "gaussian-head-derivation"))
add(code("from bayesify_torch.heads import categorical_quadrature_head\n\n" + heads,
         cell_id="mfvi-ep-heads"))

add(markdown(r"""
## Exact Gaussian positive control

For the affine chain $h_{\ell+1}=A_{\ell+1}h_\ell+B_{\ell+1}w_{\ell+1}+c_{\ell+1}$,
the notebook computes the posterior in three independent ways: one dense
conditioning calculation over all weights, a local smoothing recursion, and
autodiff of the exact terminal log normalizer followed by the explicit Fisher
encoder. The reference signal is the finite raw-moment correction
$d_m=m^+-m$ and $d_M=(C^++m^+m^{+\top})-(C+mm^\top)$.
"""))
add(markdown(derivation("gaussian_smoothing"), "gaussian-smoothing-proof"))
add(source_details("Key code: exact Fisher backpropagation (gaussian_chain.py)",
    source_of("gaussian_chain.py", ("backprop_gaussian_chain",)), "gaussian-backprop-key-source", key=True))
add(source_details("PyTorch source: conditioning, smoothing, and Fisher backprop",
                   gaussian_chain, "exact-gaussian-source"))
add(code(r"""
def message_panels(points, title):
    figure, axes = plt.subplots(2, 2, figsize=(8.2, 7.7))
    for axis, (name, pair) in zip(axes.flat, points.items()):
        reference, computed = (value.detach().cpu().numpy() for value in pair)
        agreement_plot(reference, computed, xlabel="independent reference",
                       ylabel="PyTorch Fisher backprop", title=name, ax=axis)
        axis.text(.03, .96, f"max |error| = {np.max(np.abs(reference-computed)):.2e}",
                  transform=axis.transAxes, va="top", fontsize=8)
    figure.suptitle(title, fontweight="bold")
    figure.tight_layout(); plt.show()

message_panels(exact_gaussian_equivalence(),
               "Exact Gaussian chain: conditioning = smoothing = backpropagation")
""", cell_id="exact-gaussian-equivalence"))

add(markdown(r"""
## EP and MFVI on the same nonlinear graph

The next two figures compare a handwritten literal message calculation with a
local `torch.func.vjp` of the same moment map. EP seeds the reverse traversal
with a log tilted normalizer; MFVI seeds it with an expected log likelihood.
Both figures retain activation and weight signals separately, and retain the
raw-second signal instead of silently treating it as a variance increment.
"""))
add(source_details("PyTorch source: native equivalence and approximation experiments",
                   tutorial_evidence, "native-evidence-source"))
add(code(r"""
for method in ("ep", "mfvi"):
    message_panels(nonlinear_message_equivalence(method),
                   f"{method.upper()}: literal messages = autodiff backpropagation")

# The scalar counterexample distinguishes a finite EP moment update from the
# canonical MFVI update even when both happen to end at variance 1/2.
q = DiagonalMoments(torch.zeros(1), torch.ones(1))
ep = ep_gaussian_head(q, torch.zeros(1), noise_variance=1.)
vi = mfvi_gaussian_head(q, torch.zeros(1), noise_variance=1.)
display(pd.DataFrame([{
    "EP finite update": tilted_update(q, ep.signal).variance.item(),
    "MFVI canonical update": canonical_mfvi_update(q, vi.signal, q).variance.item(),
    "incorrect MFVI raw increment": (q.variance + vi.signal.delta_raw_second).item(),
}]))
""", cell_id="ep-mfvi-message-equivalence"))
add(markdown(r"""
The diagonal identities establish implementation equivalence for each declared
closure. They do not claim that a nonlinear network posterior is exactly a
diagonal Gaussian; that approximation question is measured next.
"""))

add(markdown(r"""
# 3. Approximation quality versus Monte Carlo and computation

The main accuracy refinement in this section is **sampling-assisted moment
propagation**: integrate a frozen Gaussian surrogate, then use paired network
draws to estimate its nonlinear remainder. The weight belief stays one Gaussian.
The correction is stochastic; it is not the uncorrected 64-direction closure.

Read the residual identity and native implementation first, then the corrected
messages against Monte Carlo. The retained-direction plots are **ablations**:
they explain why covariance retention alone is insufficient. The exact Gaussian,
EP and MFVI message-equivalence checks in Section 2 retain their separate role.

Diagonal activations retain marginal variances. A shared-factor state
$H=m+U^\top\epsilon/\sqrt R$ retains covariance $U^\top U/R$ across examples,
channels and graph branches. A structured weight belief
$W=m+L\xi+\sqrt d\odot\zeta$ has covariance
$LL^\top+\operatorname{Diag}d$.
"""))
add(markdown(derivation("covariance"), "covariance-derivation"))
add(code("from bayesify_torch.correlation import *\n\n" + correlation,
         cell_id="correlated-moments"))
add(markdown(derivation("residual_integration"), "residual-integration-proof"))
add(source_details("PyTorch source: correct the nonlinear residual before encoding the message",
    source_of("residual.py", ("residual_backward",)), "residual-backward-source", key=True))
add(source_details("PyTorch source: frozen Gaussian surrogate and paired network integration",
    source_of("residual.py", ("network_residual_backward",)), "residual-network-source"))
add(source_details("One graph, optional shared activation covariance",
    source_of("correlation.py", ("shared_layer", "shared_forward", "SharedPass", "shared_likelihood")),
    "shared-graph-source"))
add(markdown(r"""
`residual_backward` implements the displayed expectation identity for both
objectives. Its inputs are unnormalized likelihood derivatives for EP and
expected-log-likelihood derivatives for MFVI. The corrected derivatives pass
through the same `encode_diagonal` function as the layer messages. Finite belief
updates remain the separate EP raw-moment and MFVI canonical operations.

The independent tests in `tests/test_residual.py` check linear Gaussian-head
integration against Gauss–Hermite quadrature, nonlinear paired integration,
the EP normalizer, and invalid-estimate rejection. The integration correction
is stochastic; it does not turn the nonlinear network into an exact Gaussian.
""", "residual-implementation-contract"))
add(markdown(derivation("mc_definitions"), "mc-error-definitions"))
add(markdown(r"""
For a Gaussian ReLU marginal, the mean is exact. The first-Hermite factor map
uses $E[\phi'(Z)]=\Phi(m/\sigma)$; it is a declared covariance closure, not an
exact nonlinear joint distribution. The executable experiment below creates
one native PyTorch Gaussian belief and compares diagonal moments, shared
directions, and finite Monte Carlo budgets with a 16,384-draw reparameterized
reference, repeated independently to check reference stability. MFVI uses
the sum of expected log factors; EP uses the sum of per-example log tilted
normalizers at a fixed cavity. These are different objectives. The latter
checks local messages, not a complete multi-site EP iteration.
"""))
add(code(r"""
studies = {method: approximation_benchmark(reference_samples=16384, method=method)
           for method in ("mfvi", "ep")}
for method, (records, values) in studies.items():
    display(HTML("<b>Small nonlinear fixture: "+method.upper()+"</b>"))
    display(pd.DataFrame(records)[["method", "seconds", "variance_error",
         "mean_message_error", "centered_message_error"]].round(5))
    display(pd.DataFrame([values["reference_repeat_errors"]], index=["Independent MC repeat"]))
""", collapsed=True, cell_id="native-approximation-cost"))
add(markdown(r"""
### What the R/TensorFlow repair established

The earlier R/TensorFlow experiment applied this residual correction to 236
selected parameter coordinates spanning the ResNet-20 blocks. On the first of
four predeclared CIFAR-10 inputs, 2,048 correction evaluations were compared with
an independent 8,192-network reference. Relative errors were:

| Method | Mean signal: baseline → corrected ↓ | Raw-second signal: baseline → corrected ↓ |
|---|---:|---:|
| MFVI | 68.30% → 4.64% | 68.09% → 4.34% |
| EP | 50.29% → 9.09% | 52.13% → 9.14% |

Mean and raw-second errors decreased on all four inputs. This is the successful
sampling-assisted repair; it is not evidence that a 64-direction closure alone
is accurate. The raw-second signal is $d_s=2m\odot d_m+c$, whereas the centered
variance signal is $c=2v^2\odot g_v$. A large mean contribution can hide errors
in $c$. In this earlier experiment the corrected centered-signal error was about
133% for both methods, with very large MC uncertainty. The readout experiment
below therefore checks mean and centered variance separately with a larger,
independent reference. The two studies have different coordinates and references;
their percentages are not a common learning curve.

The source CSV, including every input and its MC standard error, is
`r_residual_confirmation.csv` in the supplied matched evidence. These are recorded
R/TensorFlow results, not newly executed full-network experiments in every backend.

## Sampling-assisted readout messages against Monte Carlo

Keep the learned ResNet-20 MFVI or EP belief and CIFAR-10 test input 0 fixed.
An independent **524,288-network** reference measures mean and centered-variance
signals, $d_m=v\odot g_m$ and $c=2v^2\odot g_v$. These are backward sensitivities,
not posterior variances. Both plot axes use the same units and scale; the diagonal
is ideal agreement.

The first figure shows the paired residual correction at 4,096 and 16,384
evaluations. The next two figures are closure ablations: forward variance against
MC, then forward/backward error versus retained directions. They do not represent
the repaired method. Antithetic-group reference standard errors are about 0.25%
for logit variance, 0.88%/0.75% for MFVI mean/variance signals and 2.31%/1.39% for EP.

The residual readout values and complete-network reference were computed in
PyTorch and are shared numerical evidence across the readers. Each framework
renders its own plots. The closure ablations use independently computed native
backend values: 80 checks agree within $1.82\times10^{-13}$ in float64.
Implementation agreement does not imply approximation accuracy. The selected-
coordinate R study above and this readout-only study establish different scopes;
neither establishes a matched all-layer corrected training run in all three backends.
""", "learned-resnet-study"))
add(code(r"""
import runpy
_comparison_root = _repo / "site" if (_repo / "site/compare_approximations.py").exists() else _repo / "evidence/matched"
matched = runpy.run_path(str(_comparison_root / "compare_approximations.py"))
error_rows = matched["compare"]()
matched["figures"](error_rows)
for name in ("forward_agreement", "readout_agreement", "covariance_and_message_error"):
    display(Image(filename=str(_comparison_root / "evidence" / (name+".png"))))
reference_rows=[]
for method in ("mfvi","ep"):
    receipt=json.loads((_comparison_root / "evidence" / (method+"_reference.json")).read_text())
    reference_rows.append({"Method":method.upper(),"Network draws":receipt["networks"],
        **{k+" relative MCSE":v for k,v in receipt["MC_relative_standard_error"].items()}})
display(pd.DataFrame(reference_rows).round(5))
cost=pd.DataFrame(json.loads((_comparison_root/"evidence"/"readout_correction_cost.json").read_text()))
display(cost[["method","networks","mean_error","variance_error","total_seconds"]].round(5))
""", collapsed=True, cell_id="learned-resnet-results"))
add(markdown(r"""
### Accuracy and measured work

| Readout calculation | MFVI mean / variance error ↓ | EP mean / variance error ↓ |
|---|---:|---:|
| Uncorrected 64-direction closure (ablation) | 60.30% / 211.49% | 44.37% / 41.83% |
| Paired correction: 4,096 evaluations | 10.1% / 17.2% | 2.73% / 2.09% |
| Paired correction: 16,384 evaluations | 2.79% / 5.85% | 0.98% / 0.97% |
| Paired correction: 65,536 evaluations | 1.45% / 2.92% | 1.29% / 0.70% |

The correction substantially improves readout agreement for both objectives.
Errors near the reference's MC standard error should not be ranked: the last
two EP rows do not establish that fewer evaluations are intrinsically better.

| Correction evaluations | MFVI seconds ↓ | EP seconds ↓ |
|---:|---:|---:|
| 4,096 | 3.75 | 2.06 |
| 16,384 | 9.34 | 7.69 |
| 65,536 | 31.80 | 30.15 |

These saved L4 timings include Jacobian/surrogate preparation and readout
correction. MFVI ran first and paid more setup overhead. They measure one-input
diagnostics, not training or serving. The correction uses float64 and returns
readout derivatives; the original reverse pass returns all weight derivatives.
They therefore do not establish a 2–3× training-cost claim. No benchmark below
has been retrained or credited with this correction.

**Why the closure remains an ablation.** With 64 directions, forward covariance
errors are still 23.3% (MFVI) and 28.2% (EP). Refining only the Gaussian head from
18 nodes to 262,144 Sobol nodes changes mean/variance errors to 49.2%/38.8% and
42.7%/31.6%. Thus neither adding directions nor integrating that approximate
head more accurately removes the missing nonlinear response. The paired residual
directly measures this missing contribution. It is the supported refinement for
this example; its budget and variance accuracy must be measured, not assumed.
""", "learned-resnet-interpretation"))

add(markdown("## Conditional readout: implementation"))
add(source_details("Readout derivatives: conditional Gaussian identity",
    source_of("heads.py", ("sampled_readout_adjoint",)), "price-readout-source", key=True))

add(markdown(r"""
# 4. MNIST: MLP and LeNet with common uncertainty diagnostics

The MLP is 784-400-400-10. LeNet uses two 5x5 convolution/max-pooling blocks
and three dense layers. Deterministic pooling is ordinary max pooling;
moment pooling uses the Gaussian-pair approximation derived below.
Point, sampled MFVI/EP and moment MFVI/EP in the compact implementation all
share the same Gaussian parameter modules. The prediction is
$\bar p(y|x)=E_q[p(y|x,W)]$: probabilities are averaged before NLL.
"""))
add(markdown(r"""
### Max pooling: explicit projection and explicit reverse messages

For independent $A\sim N(m_a,v_a)$ and $B\sim N(m_b,v_b)$, put
$s^2=v_a+v_b$, $t=(m_a-m_b)/s$, $P=\Phi(t)$, $p=\phi(t)$.
Splitting the expectation on $A>B$ and integrating the Gaussian difference gives
$E[\max(A,B)]=m_aP+m_b(1-P)+sp$ and
$E[\max(A,B)^2]=(v_a+m_a^2)P+(v_b+m_b^2)(1-P)+(m_a+m_b)sp$.
Subtract the squared mean to obtain the variance. Differentiating the mean
gives $\partial_{m_a}m=P$, $\partial_{m_b}m=1-P$ and
$\partial_{v_a}m=\partial_{v_b}m=p/(2s)$.
The raw-second derivatives in `_max_pair_adjoint` complete the literal reverse
calculation; the projected path independently differentiates the forward map.

A 2×2 pool applies this pair operation twice. Its intermediate maximum is
projected back to a Gaussian, so the four-way result is an approximation even
for independent Gaussian inputs. For zero input variance the layer returns
the exact deterministic maximum. Neither a hidden epsilon variance nor a
silent replacement by average pooling is used.
""", "maxpool-math"))
add(code("from bayesify_torch.layers import *\n\n" + pooling, cell_id="maxpool-implementation"))
add(code("from bayesify_torch.models import *\n\n" + models, cell_id="mnist-models"))
add(markdown(derivation("classifier"), "classifier-covariance-proof"))
add(markdown(r"""
## Complete minibatch semantics

The sampled path returns $E[g]$ and $E[g^2]$ from complete network draws. The
moment MFVI path uses one curvature direction
$E[g(W)^2]\approx E[g(W)]^2+(H\sqrt v\epsilon)^2$. Tied power EP constructs
$\eta_c=(1-\alpha/N)\eta_q+(\alpha/N)\eta_0$ and differentiates both the
tilted term and the Gaussian external field back to the persistent posterior.
"""))
add(markdown(derivation("training"), "training-derivation"))
add(markdown(derivation("power_notation"), "power-notation"))
add(markdown(derivation("optimizer_complete"), "optimizer-state-derivation"))
add(source_details("Key code: power cavity and optimizer steps (optimizers.py)",
    file_excerpt("pytorch/src/bayesify_torch/optimizers.py", ("power_cavity", "Vadam.step", "EPAdam.step")),
    "optimizer-key-source", key=True))
add(code("from bayesify_torch.optimizers import *\n\n" + optimizers,
         cell_id="gaussian-optimizers", collapsed=True, label="Support: complete optimizer classes"))
add(source_details("Key code: moment likelihood messages and Bayesian prediction (training.py)",
    source_of("training.py", ("moment_likelihood_gradients", "sampled_predict", "moment_predict")),
    "training-key-source", key=True))
add(code("from bayesify_torch.training import *\n\n" + training,
         cell_id="sampled-moment-training", collapsed=True, label="Support: full sampled/moment training routines"))
add(code("from bayesify_torch.benchmark import *\n\n" + benchmark,
         cell_id="benchmark-loop", collapsed=True, label="Support: minibatch orchestration"))

add(markdown(r"""
## Metrics used without changing definitions between benchmarks

ECE is computed from equal-width confidence bins. AURC integrates selective
risk. Ranking functions consume complete tied-score groups so input order
cannot improve AUROC/AUPRC. OOD detection reports entropy and one-minus-MSP.
"""))
add(markdown(derivation("metrics"), "metric-definitions"))
add(source_details("Key code: metrics computed from averaged probabilities (metrics.py)",
    source_of("metrics.py", ("uncertainty_metrics", "ood_metrics")), "metrics-key-source", key=True))
add(code("from bayesify_torch.metrics import *\n\n" + metrics + "\n\n" + plotting,
         cell_id="uncertainty-metrics", collapsed=True, label="Support: ranking, metrics and plot formatting"))
add(source_details("PyTorch source: summaries and ONNX export",
                   inspection, "native-inspection-source"))
add(code(r"""
mnist_models = {"MLP 784-400-400-10": MomentMLP().double(),
                "LeNet": MomentLeNet().double()}
mnist_example = torch.zeros(1, 1, 28, 28, dtype=torch.float64)
mnist_architectures = {
    "MLP 784-400-400-10": show_architecture(
        "MLP 784-400-400-10", "mnist_mlp", mnist_models["MLP 784-400-400-10"], mnist_example),
    "LeNet": show_architecture(
        "LeNet", "mnist_lenet", mnist_models["LeNet"], mnist_example),
}
""", cell_id="native-mnist-architectures"))
add(markdown(r"""
The layer tables are produced by `torchinfo`, PyTorch's maintained
`Model.summary`-style utility. `Param #` counts both the Gaussian mean and its
log variance; the compact row makes that distinction explicit. The ONNX
runtime check compares the exported mean network with the live PyTorch output.
Open the downloaded file in [Netron](https://netron.app) to inspect every edge.
"""))
add(code(r"""
# One complete native PyTorch forward/reverse semantic pass for each model.
torch.manual_seed(41)
image = torch.randn(2, 1, 28, 28)
labels = torch.tensor([2, 7])
smoke = []
for name, model in mnist_models.items():
    start = time.perf_counter()
    passed = model.forward_moments(DiagonalMoments(image, torch.zeros_like(image)))
    head = categorical_quadrature_head(passed.output, labels, "mfvi")
    messages = passed.backward(head.signal)
    smoke.append({"architecture": name, "output shape": tuple(passed.output.mean.shape),
                  "weight message blocks": len(messages.parameters),
                  "finite": bool(torch.isfinite(passed.output.mean).all()
                                 and torch.isfinite(passed.output.variance).all()),
                  "forward + backward seconds": time.perf_counter()-start})
display(pd.DataFrame(smoke))
""", cell_id="native-mnist-semantic-pass"))
add(markdown(r"""
These cells establish executable architecture and message semantics. They do
not substitute a two-example smoke pass for a trained MNIST benchmark. A full
benchmark must be produced by the PyTorch training path and exported from its
own raw predictions; outputs from other frameworks are not embedded in this
PyTorch artifact.
"""))
add(markdown(r"""
## Trained MNIST results and uncertainty diagnostics

The following evidence comes from the original **PyTorch** MLP/LeNet runs,
not the later R/TensorFlow MNIST rerun. All five methods use categorical
classification; the MLP has two 400-unit hidden layers, and the LeNet has two
convolution/max-pooling blocks. Training, stopping and official test sets were
separated; the archived configuration and selected epoch are retained for each
run. The displayed curves use seed 41, fixed before plotting. The table uses
each available training seed separately, with mean ± standard deviation.
It never averages networks across seeds to turn a method into a deep ensemble.

The short semantic checks above and the historical training below answer
different questions. `evidence/manifest.json` identifies the exact original
implementation and checkpoint hash for each saved prediction. The new compact
trainer has **not** been shown to reproduce every original optimized run.
The uncertainty calculations below do reproduce the saved predictions' metrics.
""", "mnist-results-protocol"))
add(source_details("Shared Python result processing (used by every benchmark)",
                   (SOURCE / "results.py").read_text(), "native-result-source"))
add(code(r"""
from bayesify_torch.results import load_predictions, benchmark_metrics, benchmark_panel, benchmark_learning
evidence = _repo / "pytorch" / "evidence" if (_repo / "pytorch/evidence").exists() else _repo / "evidence"

def show_benchmark(family, title, *, learning=False):
    records = load_predictions(evidence, family)
    measured = benchmark_metrics(records)
    fields = {"accuracy": "Accuracy ↑", "nll": "NLL ↓", "brier": "Brier ↓",
              "ece": "ECE ↓", "aurc": "AURC ↓", "error_auroc": "Error AUROC ↑"}
    display(HTML(f"<h3>{title}</h3>"))
    summary = measured.groupby("method", sort=False)[list(fields)].agg(["mean", "std"])
    reader = pd.DataFrame(index=summary.index)
    for key, label in fields.items():
        reader[label] = [f"{m:.5f}" + (f" ± {s:.5f}" if np.isfinite(s) else "")
                         for m, s in zip(summary[key]["mean"], summary[key]["std"])]
    display(reader)
    if "msp_auroc" in measured:
        display(measured.groupby("method", sort=False)[["msp_auroc", "msp_auprc", "msp_fpr95"]]
                .mean().rename(columns={"msp_auroc":"OOD AUROC ↑", "msp_auprc":"OOD AP ↑",
                                        "msp_fpr95":"OOD FPR95 ↓"}).round(5))
    costs = [{"method": r["label"], "seed": r["seed"], **r["costs"]}
             for r, _ in records if "costs" in r]
    if costs:
        times = pd.DataFrame(costs)
        for key in ("training_seconds", "inference_seconds"):
            times[key] = pd.to_numeric(times[key])
        timing_table = times.groupby("method", sort=False).agg(
            training_seconds=("training_seconds", "mean"), inference_seconds=("inference_seconds", "mean"),
            device=("hardware", "first"), prediction_cases=("inference_n", "first"))
        ordinary = timing_table[timing_table.index.str.startswith("Ordinary")].iloc[0]
        if timing_table.device.nunique() == 1 and timing_table.prediction_cases.nunique() == 1:
            timing_table["Train × ↓"] = timing_table.training_seconds/ordinary.training_seconds
            timing_table["Predict × ↓"] = timing_table.inference_seconds/ordinary.inference_seconds
        display(timing_table.rename(columns={"training_seconds":"Train seconds ↓",
                "inference_seconds":"Predict seconds ↓"}).round(3))
        configs = [{"method": r["label"], "seed": r["seed"], **r.get("configuration", {})}
                   for r, _ in records]
        configs = pd.DataFrame(configs)
        keys = [k for k in ["method", "seed", "lr", "tau", "epochs", "best_epoch"] if k in configs]
        display(configs[keys])
        print("Full settings, objective, checkpoint selection and timing provenance: evidence/manifest.json")
    benchmark_panel(records, title); plt.show()
    if learning:
        benchmark_learning(evidence, family); plt.show()
    return measured

""", cell_id="benchmark-report-support", collapsed=True, label="Support: shared result-table formatting"))
add(code(r"""
mlp_results = show_benchmark("mlp", "MLP / MNIST", learning=True)
lenet_results = show_benchmark("lenet", "LeNet / MNIST", learning=True)
""", cell_id="native-mnist-results"))
add(markdown(r"""
The combined diagnostics distinguish calibration, error ranking and OOD
ranking. Fashion-MNIST is the recorded OOD control.
Training and inference times above are historical GPU measurements, not the
time taken to rebuild this notebook. Compilation/setup, stopping policy,
prediction sample count and device must be considered before comparing costs.
The full configuration remains expandable immediately beside its result.
""", "mnist-results-reading"))
add(markdown(r"""
## Dirty-MNIST: uncertainty about ambiguous digits

The native LeNet comparison below trains all five methods for 30 epochs on
110,000 observations, with a fixed 10,000-observation validation set. Repeated
labels of an ambiguous image remain in the same split. Proper scores count all
ten labels; OOD ranking counts each image once. Fashion-MNIST is the OOD control,
while ambiguous digits remain in distribution. No new hyperparameter search is
used in this port check. Each method is reported separately at seed 41.

The source is `run_training.py --architecture lenet --dataset dirty --method …`.
The eager implementation executes the explicit local reverse traversal at every
update. Its timings are measured here; they are not replaced by the historical
optimized implementation's timings.
"""))
add(code(r"""
from bayesify_torch.results import native_image_predictions
dirty = native_image_predictions(evidence / "native_dirty")
assert len(dirty)==5, "The complete native Dirty-MNIST evidence must be bundled."
table = benchmark_metrics(dirty)
for name,key in [("Train s ↓","training_seconds"),("Infer s ↓","inference_seconds")]:
    table[name]=[r["measurement"][key] for r,a in dirty]
ordinary=table[table.method=="Ordinary"].iloc[0]
table["Train × ↓"]=table["Train s ↓"]/ordinary["Train s ↓"]
table["Infer × ↓"]=table["Infer s ↓"]/ordinary["Infer s ↓"]
display(table[["method","accuracy","nll","ece","brier","aurc","error_auroc","msp_auroc",
               "Train s ↓","Train × ↓","Infer s ↓","Infer × ↓"]].rename(columns={
    "accuracy":"Accuracy ↑","nll":"NLL ↓","ece":"ECE ↓","brier":"Brier ↓",
    "aurc":"AURC ↓","error_auroc":"Error AUROC ↑","msp_auroc":"OOD AUROC ↑"}).round(5))
benchmark_panel(dirty,"LeNet / Dirty-MNIST");plt.show()
""",cell_id="native-dirty-results"))
add(markdown(r"""
Inference seconds use the common 10,000-image clean-MNIST cohort, rather than
mixing computation for repeated ambiguous labels into the timing. Moment models
use one moment pass; sampled models use 20 networks. These are the original
warmed cohort measurements from each complete run. The separate values-only
prediction optimization is checked against a live backward-record traversal
and does not replace those historical run times. The reported probability
scores include the ambiguous labels. These one-seed results validate the full
native path and do not establish a universal Bayesian advantage.
"""))

add(markdown(r"""
# 5. CIFAR-10: ResNet-20 and DenseNet-40

ResNet-20 contains three stages of three residual blocks and the option-A
stride-two shortcut. DenseNet-40 has three 12-layer dense blocks with growth
12 and no bottleneck/compression. Addition and concatenation are ordinary DAG
nodes, so all reverse contributions meet before their shared parent is visited.
"""))
add(code("from bayesify_torch.architectures import *\n\n" + architectures,
         cell_id="cifar-architectures"))
add(code(r"""
native_cifar = {"ResNet-20": resnet20().float(), "DenseNet-40 (growth 12)": densenet40().float()}
cifar_example = torch.zeros(1, 3, 32, 32)
cifar_architectures = {
    "ResNet-20": show_architecture(
        "ResNet-20", "cifar_resnet20", native_cifar["ResNet-20"], cifar_example, inline=False),
    "DenseNet-40 (growth 12)": show_architecture(
        "DenseNet-40 (growth 12)", "cifar_densenet40",
        native_cifar["DenseNet-40 (growth 12)"], cifar_example, inline=False),
}
""", cell_id="native-cifar-summary"))
add(markdown(r"""
The complete ResNet and DenseNet graphs are deliberately scrollable instead
of being shrunk into an unreadable page thumbnail. Netron shows the actual
residual additions and dense concatenations. The ONNX files are executable
mean networks; Gaussian variances and reverse messages remain in the PyTorch
modules and are documented by the parameter split and inference sections.
"""))
add(code(r"""
torch.manual_seed(9)
cifar_input, cifar_label = torch.randn(1,3,32,32, dtype=torch.float32), torch.tensor([4])
cifar_smoke = []
for name, model in native_cifar.items():
    model.eval()
    start = time.perf_counter()
    passed = model.forward_moments(DiagonalMoments(cifar_input, torch.zeros_like(cifar_input)))
    child = categorical_quadrature_head(passed.output, cifar_label, "ep", power=.5).signal
    messages = passed.backward(child)
    cifar_smoke.append({"architecture": name, "DAG nodes": len(model.network.nodes),
                        "weight message blocks": len(messages.parameters),
                        "finite": all(torch.isfinite(value.delta_mean).all()
                                      for value in messages.parameters.values()),
                        "forward + backward seconds": time.perf_counter()-start})
display(pd.DataFrame(cifar_smoke))
""", cell_id="native-cifar-semantic-pass"))
add(markdown(r"""
The native smoke pass verifies that graph semantics survive loops in the
undirected inference view: skip and dense connections accumulate one message
per edge, while each persistent weight belief receives one accumulated update.
It is not a replacement for a full CIFAR-10 training run, so no result from a
different implementation is shown here.
"""))
add(markdown(r"""
## Trained residual and dense networks

These original PyTorch runs trained from scratch for 100 epochs, with a fixed
5,000-example validation cohort used for checkpoint selection. This is the
archived validation comparison, **not** the later R/TensorFlow official-test
table. The OOD control is SVHN (1,000 predeclared ID and 1,000 OOD cases).
ResNet and DenseNet share the definitions
of NLL, calibration, selective risk and error/OOD ROC used for MNIST. They
retain separate timing receipts; a speed ratio across different devices would
not be meaningful.

The skip-connection test establishes correct accumulation on a computation
DAG. Covariance approximation and the choice of finite optimizer remain
separate questions; cycles in an undirected factor graph do not turn a
feed-forward residual network into a recurrent computation.
""", "cifar-results-protocol"))
add(code(r"""
resnet_results = show_benchmark("resnet20", "ResNet-20 / CIFAR-10", learning=True)
densenet_results = show_benchmark("densenet40", "DenseNet-40 / CIFAR-10", learning=True)
""", cell_id="native-cifar-results"))

add(markdown(r"""
# 6. GPT-2 with rank-4 Q/V LoRA on SST-2

Only the adapters are uncertain. A complete draw is shared across all tokens
and examples. The moment path retains shared covariance carriers through the
transformer and averages class probabilities at the verbalizer. The original
training sources are already PyTorch modules and are included unchanged.
"""))
add(markdown(derivation("transformer_calculus"), "transformer-directional-proof"))
add(source_details("Key code: local moment push, explicit reverse and probability integration (gpt2/moments.py)",
    file_excerpt("tensorflow/python/gpt2/moments.py", ("TransformerMoments.push", "TransformerMoments.reverse", "binary_probability")),
    "gpt2-key-source", key=True))
add(code(r"""
from pathlib import Path
for path in ["python/gpt2/model.py", "python/gpt2/moments.py",
             "python/gpt2/optimizers.py", "python/gpt2/moment_training.py"]:
    text = (_reference_repo / path).read_text()
    tree = ast.parse(text)
    names = [node.name for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))]
    print(f"{path}: {', '.join(names)}")
""", cell_id="gpt2-code-map"))
add(source_details("Native PyTorch source: GPT-2 LoRA model",
                   (REPO / "tensorflow" / "python" / "gpt2" / "model.py").read_text(),
                   "gpt2-model-source"))
add(source_details("Native PyTorch source: transformer moment carriers",
                   (REPO / "tensorflow" / "python" / "gpt2" / "moments.py").read_text(),
                   "gpt2-moment-source"))
add(code(r"""
from bayesify_torch.transformer_inspection import transformer_summary, export_transformer
text, counts = transformer_summary(_reference_repo, "sst2")
display(pd.DataFrame([counts]))
display(HTML("<details><summary><b>Complete torchinfo layer summary</b></summary><pre>"+html.escape(text)+"</pre></details>"))
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    receipt = export_transformer(_reference_repo, "sst2", _architecture_work / "sst2.onnx")
display(pd.DataFrame([receipt]))
display(HTML(architecture_link("sst2", _architecture_work / "sst2.onnx")))
print("Native training entry point: python/gpt2/run_selected.py")
""", cell_id="gpt2-native-architecture"))
add(markdown(r"""
## Trained adapters: accuracy, calibration and error detection

The original comparison uses GPT-2-small, rank-4 Gaussian Q/V adapters and two
fixed sentiment verbalizer rows. It is binary classification, not autoregressive
generation of a sentence. The official SST-2 validation split contains 872
examples. The displayed selected checkpoints are exploratory single-seed
results; they are not a fresh independent test after choosing among settings.
The original mean/variance learning rates, temperatures, prediction sample
counts, epoch and costs are bundled with the raw probabilities.
"""))
add(code(r"""
sst2_results = show_benchmark("sst2", "GPT-2 / SST-2: selected EP and MFVI settings")
""", cell_id="native-sst2-results"))
add(markdown(r"""
## The same Gaussian adapters in two independent implementations

This check transfers the selected MFVI and EP **beliefs**, including both Gaussian
LoRA factors, without training again. The two implementations use the same frozen
GPT-2 weights, 872 sentiment inputs and 20-node scalar Gaussian integral.
For the logit contrast $z(w)$, the delta approximation is $m_z=z(m)$ and
$v_z=\sum_j(\partial_jz(m))^2v_j$. Differentiating this approximation also
differentiates its Jacobian; freezing that Jacobian would define a different update.
This check uses the fully differentiated map; a frozen-Jacobian training variant
is a separate approximation.

The table compares native PyTorch and JAX predictions and likelihood derivatives
at the same four fixed examples. It tests the implementation of this approximation,
not its accuracy against nonlinear posterior samples. The earlier Monte Carlo
study supplies the separate approximation-quality comparison.
""", "transferred-belief-math"))
add(source_details("Key code: scalar moment map and method-specific likelihood",
    file_excerpt("tensorflow/python/gpt2/adjoint_variance.py", ("binary_marginals", "objective")),
    "transferred-belief-source", key=True))
add(code(r"""
folder = evidence / "transferred_gaussian_text"
subprocess.run([sys.executable, str(evidence / "matched/compare_text_beliefs.py"), str(folder)],
               check=True, capture_output=True, text=True)
agreement = json.loads((folder / "native_agreement.json").read_text())
assert agreement["passed"], "Investigate the matched-belief discrepancy before reporting equivalence."
records=[]
for method in ("mfvi", "ep"):
    for suffix, framework in (("_torch", "PyTorch"), ("", "JAX")):
        r=json.loads((folder / (method+suffix+".json")).read_text())
        records.append({"Belief":method.upper(), "Implementation":framework,
                        "Accuracy ↑":r["accuracy"], "NLL ↓":r["nll"], "ECE ↓":r["ece"]})
display(pd.DataFrame(records).round(7))
display(pd.DataFrame(agreement["rows"]).rename(columns={
    "max_abs":"Maximum absolute error ↓", "relative_l2":"Relative L2 error ↓"}))
""", cell_id="transferred-belief-check"))

add(markdown(r"""
# 7. Conditional moment MFVI: GPT-2 and RoBERTa LoRA

The BLoB-style conditional Gaussian adapter is a second **MFVI** construction.
It changes which LoRA factor is uncertain and how likelihood and KL updates
are applied. Its shared nonlinear activation carriers preserve correlations
that the simplest diagonal activation approximation drops. It is not EP and
does not use Vadam. These differences explain why it has its own transformer
implementation instead of being a flag on the image-layer example.
"""))
add(markdown(derivation("transformer"), "transformer-derivation"))
add(source_details("Key code: stochastic conditional-moment training transition (source_stream_moments.py)",
    file_excerpt("tensorflow/python/blob_mirror/source_stream_moments.py", ("transition",)),
    "spherical-training-key-source", key=True))
add(source_details("Key code: nonlinear moments and tangent normalization (compact_moments.py)",
    file_excerpt("tensorflow/python/blob_mirror/compact_moments.py", ("elementwise_moments", "tangent_norm", "tangent_softmax")),
    "conditional-carrier-key-source", key=True))
add(source_details("Native PyTorch source: separate BLoB transformer model",
                   (REPO / "tensorflow" / "python" / "blob_mirror" / "transfer_model.py").read_text(),
                   "roberta-model-source"))
add(source_details("Native PyTorch source: compact transformer moments",
                   (REPO / "tensorflow" / "python" / "blob_mirror" / "compact_moments.py").read_text(),
                   "roberta-moment-source"))
add(code(r"""
for path in ["python/blob_mirror/transfer_model.py", "python/blob_mirror/compact_moments.py"]:
    tree = ast.parse((_reference_repo / path).read_text())
    definitions = [node.name for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))]
    print(path, "->", ", ".join(definitions))
""", cell_id="roberta-native-code-map"))
add(code(r"""
text, counts = transformer_summary(_reference_repo, "clinc")
display(pd.DataFrame([counts]))
display(HTML("<details><summary><b>Complete torchinfo layer summary</b></summary><pre>"+html.escape(text)+"</pre></details>"))
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    receipt = export_transformer(_reference_repo, "clinc", _architecture_work / "clinc.onnx")
display(pd.DataFrame([receipt]))
display(HTML(architecture_link("clinc", _architecture_work / "clinc.onnx")))
""", cell_id="roberta-native-architecture"))
add(markdown(r"""
These summaries execute real full-size module graphs at randomly initialized
weights, avoiding pretrained downloads for a shape inspection. GPT-2 uses
the original Gaussian Q/V wrapper. The RoBERTa summary inspects its mean
Transformers/PEFT architecture, before adding BLoB variance parameters; its
trainable count is explicitly the mean model's count. Neither summary is a
pretrained performance evaluation. Both complete transformer exports are
checked against ONNX Runtime. Their weights are explicit graph inputs, so
the downloadable diagrams contain no pretrained checkpoint. The actual
attention, residual, normalization and adapter operations remain inspectable.
""", "transformer-inspection-scope"))
add(markdown(r"""
## Fixed-checkpoint quality and measured cost

SST-2 uses the fixed epoch-3 comparison. CLINC uses 13,500 in-scope training
examples, 1,500 stopping examples and 4,500 official in-scope test examples,
with the epoch-6 checkpoint selected on the stopping set. Its rank-8 Q/V
adapters surround a frozen RoBERTa-base backbone; the point classification
head learns 150 intents. Input and output embeddings are not an uncertain
tied language-model head in this classification setup.

The compact predictors use one nonlinear covariance direction for SST-2 and
two first-order directions for CLINC. The saved full reporting-set probabilities
below come from the final optimized PyTorch predictor. The cost table distinguishes
full historical fits from restored-step timing; multiplying a faster step
by an epoch count would not be a measured full training time.
""", "compact-results-protocol"))
add(code(r"""
compact_sst2 = show_benchmark("compact_sst2", "GPT-2 / SST-2: conditional moment MFVI")
compact_clinc = show_benchmark("compact_clinc", "RoBERTa LoRA / CLINC: conditional moment MFVI")
predictions = pd.read_csv(evidence / "compact_transformers/prediction.csv")
display(predictions[["task", "method", "prediction", "seconds", "setup_seconds", "inference_multiple"]])
steps = pd.read_csv(evidence / "compact_transformers/training_steps.csv")
display(steps.groupby(["task", "mode"])[["seconds", "step_multiple"]].mean())
fits = pd.read_csv(evidence / "compact_transformers/historical_full_fit_cost.csv")
display(fits[~fits.task.str.contains("MRPC")][["task", "method", "hardware", "training_seconds"]])
display(pd.read_csv(evidence / "compact_transformers/paired_intervals.csv"))
""", cell_id="native-compact-results"))
add(markdown(r"""
The reported point estimates improve accuracy/NLL/ECE for these matched
checkpoints. The paired intervals condition on the existing fits, not on
retraining with new seeds. In particular, the stronger fixed-epoch SST-2 NLL
contrast partly reflects ordinary-model overfitting: its earlier epoch-1 NLL
was 0.21775, close to the compact predictor's 0.21489. CLINC's ECE interval
includes zero. These distinctions let the reader assess the demonstrated
calibration and efficiency without interpreting every favorable number as
an independently confirmed win. OOD performance for this compact predictor
is not inferred from a different checkpoint's OOD results.
""", "compact-results-reading"))

add(markdown(r"""
# 8. Validation, reproduction and reading order

Run the semantic suite before any benchmark. It checks Fisher-coordinate
round trips, the EP/MFVI scalar counterexample, literal/projected layer
messages, repeated-parent accumulation, full graph updates, correlated
carriers, the power-EP cavity/field chain rule, all five training paths, metrics
and plots.
"""))
add(code(r"""
import subprocess
tests = _repo / "pytorch" / "tests"
if tests.exists():
    completed = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", str(tests), "-q"],
        env={**os.environ, "PYTHONPATH": str(_repo / "src") if (_repo/"src").exists()
             else str(_repo/"pytorch"/"src")}, text=True, capture_output=True
    )
    print(completed.stdout + completed.stderr)
    if completed.returncode: raise RuntimeError("semantic suite failed")
""", cell_id="semantic-suite"))
add(markdown(r"""
## Evidence boundary

The exact, EP/MFVI-equivalence, approximation/cost, architecture and graph
checks run in this notebook. Benchmark metrics and figures are recalculated
from the bundled original PyTorch prediction arrays. Original fit time is not
the notebook rebuild time. No checkpoints, R-generated coordinates or figures
are embedded. `evidence/manifest.json` preserves source hashes, seeds, cohort,
objective/settings and costs. The native predictors are not all the same
implementation as the compact teaching modules. LeNet now includes the
Gaussian-pair max pool and independent literal/projected reverse calculations;
the five-method Dirty-MNIST experiment above executes complete 30-epoch fits
through those compact modules. Historical clean-MNIST and CIFAR results retain
their original optimized training source and measured costs.

### Running training rather than replaying evidence

For the selected GPT-2 runs, `python/gpt2/run_selected.py --method adamw`,
`--method sampled_mfvi`, or `--method moment_ep` downloads the pinned public
GPT-2/SST-2 assets and starts the original training program. It preserves the
three-epoch schedule even when reporting the epoch-one checkpoint. The separate
conditional-MFVI code is in `python/blob_mirror/train_fixed_transfer.py` with
its source dependency and data preparation described alongside it. Running a
full fit requires those public assets and adequate GPU time; the defaults of
this reader only reproduce figures and numerical checks.

The compact image trainer exposes the complete minibatch update and is exercised
end to end by the native Dirty-MNIST fits. Its eager, readable traversal and the
optimized historical trainer have different overheads; their timings are labeled
separately. Public image data and the complete training entry point are included.
Transformer ONNX exports inspect the actual native modules, while the matched
Gaussian-adapter check compares the same saved beliefs across implementations.

## Recommended reading order

Read `geometry.py`, then the Dense forward/literal reverse pair in `layers.py`,
then `graph.py`. Compare the MFVI and EP heads before reading `training.py`.
Only then move to `correlation.py`, the image architectures, and the transformer
carriers. Every output in this notebook can be regenerated without R.
"""))

# Keep experimental computations and their outputs in place. Collapse the
# plotting/reporting mechanics, not their scientific mathematics or results.
support_cells = {
    "native-code-map": "Support: source inventory",
    "exact-gaussian-equivalence": "Support: equivalence plot formatting",
    "native-approximation-cost": "Support: integration benchmark and plotting",
    "both-objective-cost": "Support: EP/MFVI cost tables and plot formatting",
    "native-approximation-identity": "Support: forward/backward scatter formatting",
    "both-objective-backward": "Support: centered-second scatter formatting",
    "gpt2-code-map": "Support: GPT-2 source inventory",
    "roberta-native-code-map": "Support: conditional-MFVI source inventory",
    "semantic-suite": "Support: complete numerical regression suite",
}
for cell in cells:
    if cell.get("id") in support_cells:
        cell["metadata"].update(tags=["collapse-input"], jupyter={"source_hidden": True}, cellView="form")
        cell["metadata"]["code_label"] = support_cells[cell["id"]]
        cell["source"] = "#@title " + support_cells[cell["id"]] + " (expand to inspect)\n" + cell["source"]

for index, cell in enumerate(cells):
    # nbformat 4.5 requires stable cell identifiers. Explicit IDs also make
    # review links and execution receipts resilient to notebook reserialization.
    cell.setdefault("id", f"cell-{index:03d}")

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

candidate = ROOT / "README.full.ipynb"
import sys
sys.path.insert(0, str(REPO / "site"))
from tutorial_content import restore_topics
restore_topics(notebook, 'pytorch')
candidate.write_text(json.dumps(notebook, indent=1))
print(candidate, len(notebook['cells']), "cells")
