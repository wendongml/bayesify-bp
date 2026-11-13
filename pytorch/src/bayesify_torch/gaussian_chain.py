"""Exact linear-Gaussian chains and their Fisher-message backpropagation.

This module is deliberately independent of the nonlinear moment layers.  It
provides three computations of the same posterior:

1. one dense Gaussian conditioning calculation over all weight blocks;
2. a forward prediction / backward smoothing recursion; and
3. ordinary PyTorch autodiff of the terminal log normalizer, followed by the
   Gaussian Fisher-coordinate encoder from :mod:`bayesify_torch.geometry`.

Agreement of all three paths is the exact positive control used by the
PyTorch tutorial.  Every result and numerical coordinate is computed by this
native module.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from .geometry import FullMomentSignal, FullMoments, encode_full


@dataclass(frozen=True)
class AffineGaussianLayer:
    """One layer ``h_next = A h + B w + offset``."""

    activation: Tensor
    weight: Tensor
    offset: Tensor


@dataclass(frozen=True)
class GaussianChainForward:
    activations: tuple[FullMoments, ...]


@dataclass(frozen=True)
class GaussianChainPosterior:
    activations: tuple[FullMoments, ...]
    weights: tuple[FullMoments, ...]


@dataclass(frozen=True)
class GaussianChainMessages:
    activations: tuple[FullMomentSignal, ...]
    weights: tuple[FullMomentSignal, ...]


def _symmetrize(matrix: Tensor) -> Tensor:
    return (matrix + matrix.mT) / 2


def _solve_spd(matrix: Tensor, rhs: Tensor) -> Tensor:
    """Solve ``matrix @ x = rhs`` by Cholesky, without explicit inversion."""

    chol = torch.linalg.cholesky(_symmetrize(matrix))
    vector = rhs.ndim == 1
    result = torch.cholesky_solve(rhs[:, None] if vector else rhs, chol)
    return result[:, 0] if vector else result


def _condition(
    prior: FullMoments,
    cross_covariance: Tensor,
    observed_prior: FullMoments,
    observed_posterior: FullMoments,
) -> FullMoments:
    """Condition one Gaussian boundary through a smoothed child boundary."""

    gain = _solve_spd(observed_prior.covariance, cross_covariance.mT).mT
    mean = prior.mean + gain @ (observed_posterior.mean - observed_prior.mean)
    covariance = prior.covariance + gain @ (
        observed_posterior.covariance - observed_prior.covariance
    ) @ gain.mT
    return FullMoments(mean, _symmetrize(covariance))


def forward_gaussian_chain(
    layers: tuple[AffineGaussianLayer, ...],
    weights: tuple[FullMoments, ...],
    input_belief: FullMoments,
) -> GaussianChainForward:
    """Propagate exact means and covariances through an affine chain."""

    if len(layers) != len(weights):
        raise ValueError("one independent Gaussian weight belief is required per layer")
    states = [input_belief]
    for layer, weight in zip(layers, weights):
        parent = states[-1]
        mean = layer.activation @ parent.mean + layer.weight @ weight.mean + layer.offset
        covariance = (
            layer.activation @ parent.covariance @ layer.activation.mT
            + layer.weight @ weight.covariance @ layer.weight.mT
        )
        states.append(FullMoments(mean, _symmetrize(covariance)))
    return GaussianChainForward(tuple(states))


def smooth_gaussian_chain(
    layers: tuple[AffineGaussianLayer, ...],
    weights: tuple[FullMoments, ...],
    input_belief: FullMoments,
    observation: Tensor,
    observation_covariance: Tensor,
) -> tuple[GaussianChainForward, GaussianChainPosterior]:
    """Exact terminal conditioning followed by local Gaussian smoothing."""

    forward = forward_gaussian_chain(layers, weights, input_belief)
    posterior_activations = list(forward.activations)
    posterior_weights = list(weights)

    terminal = forward.activations[-1]
    innovation = terminal.covariance + observation_covariance
    terminal_gain = _solve_spd(innovation, terminal.covariance).mT
    terminal_mean = terminal.mean + terminal_gain @ (observation - terminal.mean)
    terminal_covariance = terminal.covariance - terminal_gain @ terminal.covariance
    posterior_activations[-1] = FullMoments(
        terminal_mean, _symmetrize(terminal_covariance)
    )

    for index in range(len(layers) - 1, -1, -1):
        layer = layers[index]
        child_prior = forward.activations[index + 1]
        child_posterior = posterior_activations[index + 1]
        posterior_weights[index] = _condition(
            weights[index],
            weights[index].covariance @ layer.weight.mT,
            child_prior,
            child_posterior,
        )
        # A deterministic input has no Gaussian Fisher block to decode.  Its
        # posterior remains the observed tensor.  Positive-definite uncertain
        # inputs and hidden states use the same smoothing identity.
        parent = forward.activations[index]
        if torch.linalg.matrix_rank(parent.covariance) == parent.covariance.shape[0]:
            posterior_activations[index] = _condition(
                parent,
                parent.covariance @ layer.activation.mT,
                child_prior,
                child_posterior,
            )

    return forward, GaussianChainPosterior(
        tuple(posterior_activations), tuple(posterior_weights)
    )


def condition_gaussian_chain(
    layers: tuple[AffineGaussianLayer, ...],
    weights: tuple[FullMoments, ...],
    input_belief: FullMoments,
    observation: Tensor,
    observation_covariance: Tensor,
) -> GaussianChainPosterior:
    """Independent dense reference: condition all concatenated weights once.

    The reference intentionally accepts a deterministic input.  This is the
    neural-network use case: uncertainty enters through shared weight blocks,
    and the composed Jacobians retain all induced cross-layer dependence.
    """

    if torch.count_nonzero(input_belief.covariance):
        raise ValueError("the dense weight-space reference expects a deterministic input")
    sizes = [weight.mean.numel() for weight in weights]
    starts = [0]
    for size in sizes:
        starts.append(starts[-1] + size)
    total = starts[-1]
    prior_mean = torch.cat([weight.mean for weight in weights])
    prior_covariance = prior_mean.new_zeros((total, total))
    for index, weight in enumerate(weights):
        left, right = starts[index], starts[index + 1]
        prior_covariance[left:right, left:right] = weight.covariance

    jacobians = [input_belief.mean.new_zeros((input_belief.mean.numel(), total))]
    offsets = [input_belief.mean]
    for index, layer in enumerate(layers):
        jacobian = layer.activation @ jacobians[-1]
        left, right = starts[index], starts[index + 1]
        jacobian[:, left:right] = jacobian[:, left:right] + layer.weight
        jacobians.append(jacobian)
        offsets.append(layer.activation @ offsets[-1] + layer.offset)

    design, offset = jacobians[-1], offsets[-1]
    observed_mean = offset + design @ prior_mean
    innovation = design @ prior_covariance @ design.mT + observation_covariance
    gain = _solve_spd(innovation, design @ prior_covariance).mT
    posterior_mean = prior_mean + gain @ (observation - observed_mean)
    posterior_covariance = prior_covariance - gain @ design @ prior_covariance
    posterior_covariance = _symmetrize(posterior_covariance)

    activation_posteriors = []
    for jacobian, offset in zip(jacobians, offsets):
        activation_posteriors.append(
            FullMoments(
                offset + jacobian @ posterior_mean,
                _symmetrize(jacobian @ posterior_covariance @ jacobian.mT),
            )
        )
    weight_posteriors = []
    for index in range(len(weights)):
        left, right = starts[index], starts[index + 1]
        weight_posteriors.append(
            FullMoments(
                posterior_mean[left:right],
                posterior_covariance[left:right, left:right],
            )
        )
    return GaussianChainPosterior(tuple(activation_posteriors), tuple(weight_posteriors))


def posterior_signal(prior: FullMoments, posterior: FullMoments) -> FullMomentSignal:
    """Raw-moment correction taking ``prior`` exactly to ``posterior``."""

    prior_second = prior.covariance + torch.outer(prior.mean, prior.mean)
    posterior_second = posterior.covariance + torch.outer(posterior.mean, posterior.mean)
    return FullMomentSignal(
        posterior.mean - prior.mean,
        posterior_second - prior_second,
    )


def backprop_gaussian_chain(
    layers: tuple[AffineGaussianLayer, ...],
    weights: tuple[FullMoments, ...],
    input_belief: FullMoments,
    observation: Tensor,
    observation_covariance: Tensor,
) -> tuple[GaussianChainForward, GaussianChainMessages]:
    """Differentiate the exact log normalizer and encode Fisher messages.

    Every activation boundary and weight block is a separate differentiation
    target.  Autodiff supplies ordinary Euclidean adjoints only; the explicit
    :func:`encode_full` call is what turns them into Gaussian moment messages.
    """

    weight_means = tuple(weight.mean.detach().clone().requires_grad_() for weight in weights)
    weight_covariances = tuple(
        weight.covariance.detach().clone().requires_grad_() for weight in weights
    )
    means = [input_belief.mean]
    covariances = [input_belief.covariance]
    for layer, mean, covariance in zip(layers, weight_means, weight_covariances):
        means.append(layer.activation @ means[-1] + layer.weight @ mean + layer.offset)
        covariances.append(
            layer.activation @ covariances[-1] @ layer.activation.mT
            + layer.weight @ covariance @ layer.weight.mT
        )

    residual = observation - means[-1]
    innovation = _symmetrize(covariances[-1] + observation_covariance)
    chol = torch.linalg.cholesky(innovation)
    quadratic = residual @ torch.cholesky_solve(residual[:, None], chol)[:, 0]
    objective = -(
        observation.numel() * math.log(2 * math.pi)
        + 2 * torch.log(torch.diagonal(chol)).sum()
        + quadratic
    ) / 2

    # The deterministic input is excluded: its covariance has no invertible
    # Fisher block.  Every stochastic hidden boundary is positive definite in
    # the validation fixture.
    targets = tuple(means[1:] + covariances[1:]) + tuple(weight_means + weight_covariances)
    derivatives = torch.autograd.grad(objective, targets)
    layers_count = len(layers)
    activation_gradients = zip(
        derivatives[:layers_count], derivatives[layers_count : 2 * layers_count]
    )
    weight_gradients = zip(
        derivatives[2 * layers_count : 3 * layers_count],
        derivatives[3 * layers_count : 4 * layers_count],
    )
    forward = GaussianChainForward(
        tuple(
            [input_belief]
            + [FullMoments(mean.detach(), covariance.detach()) for mean, covariance in zip(means[1:], covariances[1:])]
        )
    )
    activation_messages = tuple(
        encode_full(forward.activations[index + 1], (gm, gc))
        for index, (gm, gc) in enumerate(activation_gradients)
    )
    weight_messages = tuple(
        encode_full(weight, (gm, gc))
        for weight, (gm, gc) in zip(weights, weight_gradients)
    )
    return forward, GaussianChainMessages(activation_messages, weight_messages)


def install_full(prior: FullMoments, signal: FullMomentSignal) -> FullMoments:
    """Install one finite raw-moment correction, including the quadratic term."""

    mean = prior.mean + signal.delta_mean
    prior_second = prior.covariance + torch.outer(prior.mean, prior.mean)
    second = prior_second + signal.delta_raw_second
    return FullMoments(mean, _symmetrize(second - torch.outer(mean, mean)))


def gaussian_chain_fixture(seed: int = 17) -> tuple[
    tuple[AffineGaussianLayer, ...],
    tuple[FullMoments, ...],
    FullMoments,
    Tensor,
    Tensor,
]:
    """Small deterministic-input fixture used by tests and the tutorial."""

    generator = torch.Generator().manual_seed(seed)
    dimensions = (3, 4, 3, 2)
    layers, weights = [], []
    for index, (parent, child) in enumerate(zip(dimensions[:-1], dimensions[1:])):
        weight_dimension = child + 2
        activation = torch.randn(child, parent, generator=generator) / math.sqrt(parent)
        weight_map = torch.randn(child, weight_dimension, generator=generator)
        # Ensure a well-conditioned stochastic boundary even for a random map.
        weight_map[:, :child] += torch.eye(child)
        offset = torch.randn(child, generator=generator) * 0.1
        mean = torch.randn(weight_dimension, generator=generator) * 0.15
        root = torch.randn(weight_dimension, weight_dimension, generator=generator)
        covariance = root @ root.mT / weight_dimension + 0.3 * torch.eye(weight_dimension)
        layers.append(AffineGaussianLayer(activation, weight_map, offset))
        weights.append(FullMoments(mean, covariance))
    input_belief = FullMoments(torch.randn(dimensions[0], generator=generator), torch.zeros(dimensions[0], dimensions[0]))
    observation = torch.randn(dimensions[-1], generator=generator)
    observation_covariance = 0.4 * torch.eye(dimensions[-1])
    return tuple(layers), tuple(weights), input_belief, observation, observation_covariance


__all__ = [
    "AffineGaussianLayer",
    "GaussianChainForward",
    "GaussianChainPosterior",
    "GaussianChainMessages",
    "forward_gaussian_chain",
    "smooth_gaussian_chain",
    "condition_gaussian_chain",
    "posterior_signal",
    "backprop_gaussian_chain",
    "install_full",
    "gaussian_chain_fixture",
]
