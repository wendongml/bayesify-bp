"""Readable PyTorch counterpart of the Bayesify moment/message kernel."""

from .geometry import (
    DiagonalMoments,
    EuclideanAdjoint,
    FullMoments,
    FullMomentSignal,
    MomentSignal,
    canonical_mfvi_update,
    decode_diagonal,
    decode_full,
    encode_diagonal,
    encode_full,
    tilted_update,
)
from .gaussian_chain import (
    AffineGaussianLayer,
    GaussianChainForward,
    GaussianChainMessages,
    GaussianChainPosterior,
    backprop_gaussian_chain,
    condition_gaussian_chain,
    forward_gaussian_chain,
    gaussian_chain_fixture,
    install_full,
    posterior_signal,
    smooth_gaussian_chain,
)
from .correlation import SharedMoments, StructuredGaussian, gaussian_readout
from .graph import GraphMessages, GraphPass, MomentDAG, Node
from .heads import (
    categorical_quadrature_head,
    ep_gaussian_head,
    mfvi_categorical_head,
    mfvi_gaussian_head,
)
from .inference import apply_global_ep_tilt, apply_mfvi, named_gaussian_parameters
from .layers import (
    GaussianParameter,
    LayerMessages,
    LocalRecord,
    MomentAdd,
    MomentAvgPool2d,
    MomentMaxPool2d,
    MomentBatchNorm2d,
    MomentCifarShortcut,
    MomentConcat,
    MomentConv2d,
    MomentFlatten,
    MomentGlobalAvgPool2d,
    MomentIdentity,
    MomentLinear,
    MomentReLU,
)
from .models import MomentLeNet, MomentMLP, TinyResidualMLP
from .metrics import ood_metrics, uncertainty_metrics
from .architectures import MomentCifarNet, densenet40, resnet20
from .optimizers import EPAdam, Vadam, gaussian_kl, power_cavity
from .training import (
    moment_ep_gradients,
    moment_likelihood_gradients,
    moment_predict,
    moment_vadam_second,
    sampled_ep_gradients,
    sampled_likelihood_gradients,
    sampled_predict,
)
from .benchmark import GaussianTrainer, TrainingConfig, evaluate_batches
from .plotting import (
    agreement_plot,
    calibration_plot,
    learning_curves,
    ood_roc_plot,
    risk_coverage_plot,
    uncertainty_panel,
)
from .tutorial_evidence import (
    approximation_benchmark,
    exact_gaussian_equivalence,
    nonlinear_message_equivalence,
)
from .inspect import (
    export_onnx,
    model_summary,
    parameter_summary,
    write_architecture_catalog,
)

__all__ = [name for name in globals() if not name.startswith("_")]
