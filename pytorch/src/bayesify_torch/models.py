"""Small reference architectures built from the same native moment modules."""

from __future__ import annotations

from torch import Tensor, nn

from .geometry import DiagonalMoments
from .graph import GraphPass, MomentDAG
from .layers import (
    MomentAdd,
    MomentAvgPool2d,
    MomentMaxPool2d,
    MomentConv2d,
    MomentFlatten,
    MomentLinear,
    MomentReLU,
)


class TinyResidualMLP(nn.Module):
    """A non-sequential graph used to test shared-parent message accumulation."""

    def __init__(self, inputs: int = 4, hidden: int = 6, outputs: int = 3):
        super().__init__()
        # ``graph`` is reserved by torch.fx.GraphModule; use a normal module
        # name so the deterministic architecture remains symbolically traceable.
        self.network = MomentDAG(
            [
                ("input_projection", MomentLinear(inputs, hidden), ("input",)),
                ("activation", MomentReLU(), ("input_projection",)),
                ("residual_projection", MomentLinear(hidden, hidden), ("activation",)),
                ("residual_add", MomentAdd(), ("activation", "residual_projection")),
                ("readout", MomentLinear(hidden, outputs), ("residual_add",)),
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)

    def forward_moments(self, x: DiagonalMoments, *, literal: bool = False) -> GraphPass:
        return self.network.forward_moments(x, literal=literal)


class MomentMLP(nn.Module):
    """784-400-400-10 MNIST MLP built from the shared moment modules."""

    def __init__(self, inputs: int = 784, hidden: int = 400, classes: int = 10):
        super().__init__()
        self.network = MomentDAG(
            [
                ("flatten", MomentFlatten(), ("input",)),
                ("dense1", MomentLinear(inputs, hidden), ("flatten",)),
                ("relu1", MomentReLU(), ("dense1",)),
                ("dense2", MomentLinear(hidden, hidden), ("relu1",)),
                ("relu2", MomentReLU(), ("dense2",)),
                ("readout", MomentLinear(hidden, classes), ("relu2",)),
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x.flatten(1))

    def forward_moments(self, x: DiagonalMoments, *, literal: bool = False) -> GraphPass:
        flattened = DiagonalMoments(x.mean.flatten(1), x.variance.flatten(1))
        return self.network.forward_moments(flattened, literal=literal)


class MomentLeNet(nn.Module):
    """61,706-parameter Conv2d/max-pool LeNet; Gaussian-pair moment pooling."""

    def __init__(self, classes: int = 10):
        super().__init__()
        self.network = MomentDAG(
            [
                ("conv1", MomentConv2d(1, 6, 5, padding=2), ("input",)),
                # max(ReLU(x)) = ReLU(max(x)) for point values. Gaussian
                # projections do NOT commute: match R/JAX's pool-then-ReLU
                # closure, rather than silently defining another approximation.
                ("pool1", MomentMaxPool2d(), ("conv1",)),
                ("relu1", MomentReLU(), ("pool1",)),
                ("conv2", MomentConv2d(6, 16, 5), ("relu1",)),
                ("pool2", MomentMaxPool2d(), ("conv2",)),
                ("relu2", MomentReLU(), ("pool2",)),
                ("flatten", MomentFlatten(), ("relu2",)),
                ("dense1", MomentLinear(16 * 5 * 5, 120), ("flatten",)),
                ("relu3", MomentReLU(), ("dense1",)),
                ("dense2", MomentLinear(120, 84), ("relu3",)),
                ("relu4", MomentReLU(), ("dense2",)),
                ("readout", MomentLinear(84, classes), ("relu4",)),
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)

    def forward_moments(self, x: DiagonalMoments, *, literal: bool = False) -> GraphPass:
        return self.network.forward_moments(x, literal=literal)
