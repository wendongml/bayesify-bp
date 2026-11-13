"""Complete MNIST and CIFAR architectures assembled from native modules.

The builders expose the Bayesian layers directly rather than wrapping a
separately trained deterministic network.  One set of Gaussian parameters
therefore defines the point, sampled, diagonal-moment and reverse-message
paths.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .geometry import DiagonalMoments
from .graph import GraphPass, MomentDAG
from .layers import (
    MomentAdd,
    MomentAvgPool2d,
    MomentBatchNorm2d,
    MomentCifarShortcut,
    MomentConcat,
    MomentConv2d,
    MomentGlobalAvgPool2d,
    MomentLinear,
    MomentReLU,
)
from .models import MomentLeNet


def _conv(cin: int, cout: int, kernel: int = 3, stride: int = 1) -> MomentConv2d:
    layer = MomentConv2d(
        cin, cout, kernel, stride=stride, padding=kernel // 2, bias=False, prior_variance=1e-3
    )
    nn.init.kaiming_normal_(layer.weight.mean, mode="fan_out", nonlinearity="relu")
    return layer


class MomentCifarNet(nn.Module):
    """A native PyTorch model with deterministic and moment/message entry points."""

    def __init__(self, graph: MomentDAG, architecture: str):
        super().__init__()
        self.network, self.architecture = graph, architecture

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)

    def forward_moments(self, x: DiagonalMoments) -> GraphPass:
        # CIFAR BatchNorm and option-A shortcut use the projected local VJP.
        # The primitive dense/conv/ReLU equations retain separate literal tests.
        return self.network.forward_moments(x, literal=False)

    @property
    def mean_parameter_count(self) -> int:
        return sum(module.mean.numel() for module in self.modules() if hasattr(module, "mean"))


def resnet20(classes: int = 10) -> MomentCifarNet:
    """Original CIFAR ResNet-20: three stages, three residual blocks per stage."""
    nodes: list[tuple[str, nn.Module, tuple[str, ...]]] = []

    def add(name: str, layer: nn.Module, *parents: str) -> str:
        nodes.append((name, layer, tuple(parents)))
        return name

    x = add("stem_conv", _conv(3, 16), "input")
    x = add("stem_bn", MomentBatchNorm2d(16), x)
    x = add("stem_relu", MomentReLU(), x)
    channels = 16
    for stage, output_channels in enumerate((16, 32, 64), start=1):
        for block in range(1, 4):
            prefix = f"stage{stage}_block{block}"
            stride = 2 if stage > 1 and block == 1 else 1
            skip = x if stride == 1 else add(f"{prefix}_shortcut", MomentCifarShortcut(), x)
            h = add(f"{prefix}_conv1", _conv(channels, output_channels, stride=stride), x)
            h = add(f"{prefix}_bn1", MomentBatchNorm2d(output_channels), h)
            h = add(f"{prefix}_relu1", MomentReLU(), h)
            h = add(f"{prefix}_conv2", _conv(output_channels, output_channels), h)
            h = add(f"{prefix}_bn2", MomentBatchNorm2d(output_channels), h)
            x = add(f"{prefix}_sum", MomentAdd(), skip, h)
            x = add(f"{prefix}_relu2", MomentReLU(), x)
            channels = output_channels
    x = add("global_pool", MomentGlobalAvgPool2d(), x)
    add("classifier", MomentLinear(channels, classes, prior_variance=1e-3), x)
    return MomentCifarNet(MomentDAG(nodes), "resnet20")


def densenet40(classes: int = 10, growth: int = 12) -> MomentCifarNet:
    """CIFAR DenseNet-40, growth 12, without bottlenecks or compression."""
    nodes: list[tuple[str, nn.Module, tuple[str, ...]]] = []

    def add(name: str, layer: nn.Module, *parents: str) -> str:
        nodes.append((name, layer, tuple(parents)))
        return name

    channels = 16
    x = add("stem_conv", _conv(3, channels), "input")
    for stage in range(1, 4):
        for block in range(1, 13):
            prefix = f"dense{stage}_{block}"
            h = add(f"{prefix}_bn", MomentBatchNorm2d(channels), x)
            h = add(f"{prefix}_relu", MomentReLU(), h)
            h = add(f"{prefix}_conv", _conv(channels, growth), h)
            x = add(f"{prefix}_concat", MomentConcat(dim=1), x, h)
            channels += growth
        if stage < 3:
            prefix = f"transition{stage}"
            x = add(f"{prefix}_bn", MomentBatchNorm2d(channels), x)
            x = add(f"{prefix}_relu", MomentReLU(), x)
            x = add(f"{prefix}_conv", _conv(channels, channels, kernel=1), x)
            x = add(f"{prefix}_pool", MomentAvgPool2d(2), x)
    x = add("final_bn", MomentBatchNorm2d(channels), x)
    x = add("final_relu", MomentReLU(), x)
    x = add("global_pool", MomentGlobalAvgPool2d(), x)
    add("classifier", MomentLinear(channels, classes, prior_variance=1e-3), x)
    return MomentCifarNet(MomentDAG(nodes), "densenet40")


__all__ = ["MomentLeNet", "MomentCifarNet", "resnet20", "densenet40"]
