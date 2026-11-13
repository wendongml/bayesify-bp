"""Native PyTorch evidence used by the standalone tutorial.

The functions in this file are small executable experiments, not stored result
replayers.  They create every plotted coordinate from PyTorch tensors:

* exact Gaussian conditioning versus Fisher-message backpropagation;
* literal layer messages versus ``torch.func.vjp`` for EP and MFVI; and
* deterministic diagonal/shared-direction approximations versus a large
  reparameterized Monte Carlo calculation on one fixed Gaussian belief.
"""

from __future__ import annotations

import time
import math

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.func import functional_call, vmap

from .correlation import SharedMoments, add_shared, linear_shared, relu_shared
from .gaussian_chain import (
    backprop_gaussian_chain,
    condition_gaussian_chain,
    gaussian_chain_fixture,
    posterior_signal,
)
from .geometry import DiagonalMoments, EuclideanAdjoint, MomentSignal, encode_diagonal
from .heads import categorical_quadrature_head
from .inference import named_gaussian_parameters
from .models import TinyResidualMLP
from .training import moment_likelihood_gradients


def _numpy(value: Tensor):
    return value.detach().cpu().reshape(-1).numpy()


def exact_gaussian_equivalence(seed: int = 17) -> dict[str, tuple[Tensor, Tensor]]:
    """Coordinates for the exact Gaussian activation/weight four-panel plot."""

    fixture = gaussian_chain_fixture(seed)
    layers, weights, input_belief, observation, observation_covariance = fixture
    forward, candidate = backprop_gaussian_chain(*fixture)
    exact = condition_gaussian_chain(*fixture)
    activation_reference = [
        posterior_signal(prior, posterior)
        for prior, posterior in zip(forward.activations[1:], exact.activations[1:])
    ]
    weight_reference = [
        posterior_signal(prior, posterior) for prior, posterior in zip(weights, exact.weights)
    ]

    def collect(reference, computed, field):
        return torch.cat([getattr(value, field).reshape(-1) for value in reference]), torch.cat(
            [getattr(value, field).reshape(-1) for value in computed]
        )

    return {
        "activation mean": collect(activation_reference, candidate.activations, "delta_mean"),
        "activation raw second": collect(
            activation_reference, candidate.activations, "delta_raw_second"
        ),
        "weight mean": collect(weight_reference, candidate.weights, "delta_mean"),
        "weight raw second": collect(weight_reference, candidate.weights, "delta_raw_second"),
    }


def nonlinear_message_equivalence(
    method: str, seed: int = 7
) -> dict[str, tuple[Tensor, Tensor]]:
    """Literal message equations versus an autodiff VJP on the same closure."""

    if method not in {"ep", "mfvi"}:
        raise ValueError("method must be 'ep' or 'mfvi'")
    torch.manual_seed(seed)
    model = TinyResidualMLP(4, 6, 3).double()
    inputs = DiagonalMoments(torch.randn(16, 4), torch.zeros(16, 4))
    labels = torch.arange(16) % 3
    projected = model.forward_moments(inputs, literal=False)
    literal = model.forward_moments(inputs, literal=True)
    child = categorical_quadrature_head(projected.output, labels, method).signal
    computed, reference = projected.backward(child), literal.backward(child)

    def collect(left, right, field):
        common = sorted(left.keys() & right.keys())
        return torch.cat([getattr(right[key], field).reshape(-1) for key in common]), torch.cat(
            [getattr(left[key], field).reshape(-1) for key in common]
        )

    return {
        "activation mean": collect(
            computed.activations, reference.activations, "delta_mean"
        ),
        "activation raw second": collect(
            computed.activations, reference.activations, "delta_raw_second"
        ),
        "weight mean": collect(computed.parameters, reference.parameters, "delta_mean"),
        "weight raw second": collect(
            computed.parameters, reference.parameters, "delta_raw_second"
        ),
    }


def _parameters(graph):
    return dict(named_gaussian_parameters(graph))


def _epsilon(parameters, samples: int, seed: int) -> dict[str, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        name: torch.randn(
            (samples,) + parameter.mean.shape,
            generator=generator,
            dtype=parameter.mean.dtype,
            device=parameter.mean.device,
        )
        for name, parameter in parameters.items()
    }


def _mc_logits(graph, inputs: Tensor, epsilon: dict[str, Tensor], means, log_variances) -> Tensor:
    replacements = {
        f"layers.{name}.mean": mean.unsqueeze(0)
        + (0.5 * log_variance).exp().unsqueeze(0) * epsilon[name]
        for (name, _), mean, log_variance in zip(
            _parameters(graph).items(), means, log_variances
        )
    }
    return vmap(lambda values: functional_call(graph, values, (inputs,), strict=False))(
        replacements
    )


def _rademacher(parameters, rank: int, seed: int) -> dict[str, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        name: (
            2
            * torch.randint(
                0,
                2,
                (rank,) + parameter.mean.shape,
                generator=generator,
                device=parameter.mean.device,
            ).to(parameter.mean.dtype)
            - 1
        )
        for name, parameter in parameters.items()
    }


def _shared_output(graph, inputs: Tensor, signs, means, log_variances) -> SharedMoments:
    """First-Hermite shared-direction pass through ``TinyResidualMLP``."""

    parameters = _parameters(graph)
    values = {
        name: (mean, (0.5 * log_variance).exp().unsqueeze(0) * signs[name])
        for (name, _), mean, log_variance in zip(parameters.items(), means, log_variances)
    }
    rank = next(iter(signs.values())).shape[0]
    state = SharedMoments(inputs, inputs.new_zeros((rank,) + inputs.shape))

    def linear(state, prefix):
        weight_mean, weight_factors = values[f"{prefix}.weight"]
        bias_mean, bias_factors = values[f"{prefix}.bias"]
        return linear_shared(state, weight_mean, weight_factors, bias_mean, bias_factors)

    projected = relu_shared(linear(state, "input_projection"))
    residual = linear(projected, "residual_projection")
    return linear(add_shared(projected, residual), "readout")


def _signals_from_loss_gradients(graph, derivatives) -> dict[str, MomentSignal]:
    parameters = _parameters(graph)
    messages = {}
    for (name, parameter), mean_gradient, log_variance_gradient in zip(
        parameters.items(), derivatives[0::2], derivatives[1::2]
    ):
        # The loss is minimized; messages encode the maximized log likelihood.
        messages[name] = encode_diagonal(
            parameter.moments(),
            EuclideanAdjoint(-mean_gradient, -log_variance_gradient / parameter.variance),
        )
    return messages


def _node_loss(logits, labels, method):
    """MFVI E[log p] versus EP log E[p], one likelihood factor per example."""
    logp = logits.log_softmax(-1).gather(
        -1, labels[None, :, None].expand(logits.shape[0], -1, 1)
    ).squeeze(-1)
    if method == "mfvi":
        return -logp.mean(0).sum()
    if method == "ep":
        return -(torch.logsumexp(logp, 0) - math.log(logits.shape[0])).sum()
    raise ValueError("method must be 'mfvi' or 'ep'")


def _mc_estimate(graph, inputs, labels, samples: int, seed: int, method="mfvi"):
    parameters = _parameters(graph)
    means = tuple(parameter.mean for parameter in parameters.values())
    log_variances = tuple(parameter.log_variance for parameter in parameters.values())
    epsilon = _epsilon(parameters, samples, seed)
    start = time.perf_counter()
    logits = _mc_logits(graph, inputs, epsilon, means, log_variances)
    loss = _node_loss(logits, labels, method)
    coordinates = tuple(value for pair in zip(means, log_variances) for value in pair)
    derivatives = torch.autograd.grad(loss, coordinates)
    mean, variance = logits.detach().mean(0), logits.detach().var(0, unbiased=False)
    messages = _signals_from_loss_gradients(graph, derivatives)
    return mean, variance, messages, time.perf_counter() - start


def _direction_estimate(graph, inputs, labels, rank: int, seed: int, method="mfvi"):
    parameters = _parameters(graph)
    means = tuple(parameter.mean for parameter in parameters.values())
    log_variances = tuple(parameter.log_variance for parameter in parameters.values())
    signs = _rademacher(parameters, rank, seed)
    start = time.perf_counter()
    state = _shared_output(graph, inputs, signs, means, log_variances)
    nodes = torch.cat(
        (state.mean.unsqueeze(0) + state.factors, state.mean.unsqueeze(0) - state.factors), 0
    )
    loss = _node_loss(nodes, labels, method)
    coordinates = tuple(value for pair in zip(means, log_variances) for value in pair)
    derivatives = torch.autograd.grad(loss, coordinates)
    mean, variance = state.mean.detach(), state.variance.detach()
    messages = _signals_from_loss_gradients(graph, derivatives)
    return mean, variance, messages, time.perf_counter() - start


def _relative_error(estimate: Tensor, reference: Tensor) -> float:
    numerator = torch.linalg.vector_norm(estimate - reference)
    denominator = torch.linalg.vector_norm(reference).clamp_min(torch.finfo(reference.dtype).eps)
    return float((numerator / denominator).detach())


def _message_error(estimate, reference, field: str) -> float:
    keys = sorted(reference)
    left = torch.cat([getattr(estimate[key], field).reshape(-1) for key in keys])
    right = torch.cat([getattr(reference[key], field).reshape(-1) for key in keys])
    return _relative_error(left, right)


def approximation_benchmark(
    reference_samples: int = 4096,
    ranks: tuple[int, ...] = (1, 4, 16, 64),
    mc_budgets: tuple[int, ...] = (8, 32, 128, 512),
    seed: int = 23,
    method: str = "mfvi",
) -> tuple[list[dict[str, float | str]], dict[str, object]]:
    """Compare native deterministic closures with reparameterized Monte Carlo.

    The returned rows contain measured CPU wall time and relative errors at one
    fixed, randomly initialized Gaussian belief.  The same large Monte Carlo
    run supplies the forward and backward references.  This is a semantic
    approximation study, not a trained-model accuracy claim.
    """

    torch.manual_seed(seed)
    model = TinyResidualMLP(4, 8, 3).double()
    graph = model.network
    inputs = torch.randn(24, 4)
    labels = torch.arange(24) % 3

    reference = _mc_estimate(graph, inputs, labels, reference_samples, seed + 1000, method)
    reference_mean, reference_variance, reference_messages, reference_seconds = reference
    rows: list[dict[str, float | str]] = []

    # Exclude one-time dispatcher/cache initialization from the comparison.
    graph.forward_moments(DiagonalMoments(inputs, torch.zeros_like(inputs)))
    moment_likelihood_gradients(graph, inputs, labels, method)
    start = time.perf_counter()
    passed = graph.forward_moments(DiagonalMoments(inputs, torch.zeros_like(inputs)))
    head = categorical_quadrature_head(passed.output, labels, method)
    diagonal_messages = passed.backward(head.signal).parameters
    diagonal_seconds = time.perf_counter() - start
    rows.append(
        {
            "method": "diagonal moments",
            "budget": 1,
            "seconds": diagonal_seconds,
            "mean_error": _relative_error(passed.output.mean, reference_mean),
            "variance_error": _relative_error(passed.output.variance, reference_variance),
            "mean_message_error": _message_error(
                diagonal_messages, reference_messages, "delta_mean"
            ),
            "second_message_error": _message_error(
                diagonal_messages, reference_messages, "delta_raw_second"
            ),
        }
    )

    estimates: dict[str, object] = {
        "reference": reference,
        "diagonal": (passed.output.mean.detach(), passed.output.variance.detach(), diagonal_messages),
    }
    for rank in ranks:
        _direction_estimate(graph, inputs, labels, rank, seed + rank, method)
        estimate = _direction_estimate(graph, inputs, labels, rank, seed + rank, method)
        mean, variance, messages, seconds = estimate
        name = f"shared directions {rank}"
        estimates[name] = estimate
        rows.append(
            {
                "method": name,
                "budget": 2 * rank,
                "seconds": seconds,
                "mean_error": _relative_error(mean, reference_mean),
                "variance_error": _relative_error(variance, reference_variance),
                "mean_message_error": _message_error(messages, reference_messages, "delta_mean"),
                "second_message_error": _message_error(
                    messages, reference_messages, "delta_raw_second"
                ),
            }
        )
    for samples in mc_budgets:
        _mc_estimate(graph, inputs, labels, samples, seed + samples, method)
        estimate = _mc_estimate(graph, inputs, labels, samples, seed + samples, method)
        mean, variance, messages, seconds = estimate
        name = f"Monte Carlo {samples}"
        estimates[name] = estimate
        rows.append(
            {
                "method": name,
                "budget": samples,
                "seconds": seconds,
                "mean_error": _relative_error(mean, reference_mean),
                "variance_error": _relative_error(variance, reference_variance),
                "mean_message_error": _message_error(messages, reference_messages, "delta_mean"),
                "second_message_error": _message_error(
                    messages, reference_messages, "delta_raw_second"
                ),
            }
        )
    estimates["reference_seconds"] = reference_seconds
    repeat = _mc_estimate(graph, inputs, labels, reference_samples, seed + 2000, method)
    estimates["reference_repeat"] = repeat
    # Centering removes the 2*m*d_m contribution.  A good raw-second signal
    # alone can otherwise hide a bad variance-direction approximation.
    def centered(messages):
        return torch.cat([(messages[k].delta_raw_second - 2*p.mean*messages[k].delta_mean)
                          .detach().reshape(-1) for k, p in sorted(_parameters(graph).items())])
    estimates["centered_reference"] = centered(reference_messages)
    for row in rows:
        key = "diagonal" if row["method"] == "diagonal moments" else row["method"]
        row["centered_message_error"] = _relative_error(
            centered(estimates[key][2]), estimates["centered_reference"])
    estimates["reference_repeat_errors"] = {
        "mean": _relative_error(repeat[0], reference_mean),
        "variance": _relative_error(repeat[1], reference_variance),
        "mean message": _message_error(repeat[2], reference_messages, "delta_mean"),
        "centered second message": _relative_error(centered(repeat[2]), centered(reference_messages)),
    }
    estimates["centered"] = {k: centered(v[2]) for k, v in estimates.items()
                            if isinstance(v, tuple) and len(v) >= 3}
    return rows, estimates


__all__ = [
    "exact_gaussian_equivalence",
    "nonlinear_message_equivalence",
    "approximation_benchmark",
]
