"""One complete sampled or moment minibatch calculation.

The functions here deliberately stop before choosing an optimizer.  They expose
the quantities required by Vadam, EP-Adam, canonical MFVI, or a plain PyTorch
optimizer without conflating those update rules with moment propagation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn
from torch.func import functional_call, grad, jvp

from .geometry import DiagonalMoments, decode_diagonal
from .graph import GraphMessages, MomentDAG
from .heads import categorical_quadrature_head, mfvi_categorical_head
from .inference import named_gaussian_parameters
from .optimizers import power_cavity
from .layers import MomentBatchNorm2d
from .correlation import SharedMoments, RootMoments, FactorMessages, shared_forward, shared_likelihood, root_nodes, root_objective


@dataclass(frozen=True)
class MomentGradients:
    objective: Tensor
    mean: dict[str, Tensor]
    variance: dict[str, Tensor]
    messages: GraphMessages
    statistics: dict | None = None


@dataclass(frozen=True)
class SampledGradients:
    loss: Tensor
    mean: dict[str, Tensor]
    second: dict[str, Tensor]
    statistics: dict | None = None


@dataclass(frozen=True)
class EPGradients:
    """Derivatives of a minimized tied power-EP energy at the posterior belief."""

    loss: Tensor
    mean: dict[str, Tensor]
    variance: dict[str, Tensor]
    messages: GraphMessages | None
    statistics: dict | None = None


def normalization_observations(graph, states):
    """Running-statistic observations of actual inputs to each BatchNorm.

    These are observations, not installed running state. Independent sampled
    forward passes are averaged before the ONE running-state update.
    """
    result = {}
    for node in graph.nodes:
        if isinstance(graph.layers[node.name], MomentBatchNorm2d):
            x = states[node.parents[0]]
            x = x.mean if isinstance(x, (DiagonalMoments, SharedMoments)) else x
            x = x.detach()
            # Like native BatchNorm: population variance normalizes training
            # activations, while the unbiased estimate updates running state.
            result[node.name] = (x.mean((0, 2, 3)), x.var((0, 2, 3), unbiased=True))
    return result


def average_observations(observations):
    return {name: tuple(torch.stack([row[name][i] for row in observations]).mean(0)
                       for i in (0, 1)) for name in observations[0]}


class _MomentPass(nn.Module):
    """Make ``forward_moments`` available to ``torch.func.functional_call``.

    This tiny adapter is important: a power cavity is a differentiable function
    of the persistent posterior.  Replacing the parameters functionally keeps
    that chain rule intact; copying cavity tensors into the modules would not.
    """

    def __init__(self, graph: MomentDAG, full_readout=False):
        super().__init__()
        self.graph = graph
        self.full_readout = full_readout

    def forward(self, inputs: Tensor):
        observed = DiagonalMoments(inputs, torch.zeros_like(inputs))
        return self.graph.forward_moments(observed, literal=False, full_readout=self.full_readout)


def _cavity_replacements(cavity) -> dict[str, Tensor]:
    replacements: dict[str, Tensor] = {}
    for name, belief in cavity.beliefs.items():
        replacements[f"graph.layers.{name}.mean"] = belief.mean
        replacements[f"graph.layers.{name}.log_variance"] = belief.variance.log()
    return replacements


def moment_ep_gradients(
    graph: MomentDAG,
    inputs: Tensor,
    labels: Tensor,
    observations: int,
    alpha: float,
    power: float = 1.0,
    scale: float = 1.0,
    rank: int = 0,
    seed: int = 41,
    full_readout: bool = False,
) -> EPGradients:
    """One tied power-EP moment calculation, including the external field.

    With ``k=alpha/N``, the cavity is

    ``eta_c = (1-k) eta_q + k eta_0``.

    Let ``logA`` be the Gaussian normalization correction returned by
    :func:`power_cavity`.  The minimized energy is

    ``L = -log Z_power(cavity) - logA/power``.

    The likelihood derivative is transported by explicit moment messages at
    the cavity.  ``torch.autograd.grad`` is used only for the analytic map from
    posterior coordinates to cavity coordinates and for the external field.
    The external field is applied once globally, preventing the prior from
    being diluted by the number of minibatches.
    """
    parameters = dict(named_gaussian_parameters(graph))
    cavity = power_cavity(parameters, observations, alpha)
    if rank:
        qs = {n: (q.mean,q.variance) for n,q in cavity.beliefs.items()}
        value, _, adjoints, signals, passed = shared_likelihood(
            graph,inputs,labels,rank,seed,qs,"ep",power,scale)
        messages = FactorMessages(signals,adjoints)
    else:
        wrapper = _MomentPass(graph, full_readout)
        passed = functional_call(
            wrapper, _cavity_replacements(cavity), (inputs,), strict=False
        )
        if full_readout:
            child,value=torch.func.grad_and_value(lambda m,r:root_objective(
                RootMoments(m,r),labels,'ep',power,scale),argnums=(0,1))(*passed.output)
            messages=passed.backward(child)
        else:
            head = categorical_quadrature_head(passed.output, labels, "ep", power, scale)
            value, messages = head.value, passed.backward(head.signal)

    cavity_mean, cavity_variance = [], []
    mean_adjoint, variance_adjoint = [], []
    for name, belief in cavity.beliefs.items():
        ordinary = decode_diagonal(belief, messages.parameters[name])
        cavity_mean.append(belief.mean)
        cavity_variance.append(belief.variance)
        mean_adjoint.append(ordinary.mean)
        variance_adjoint.append(ordinary.variance)

    posterior_coordinates = tuple(
        value for parameter in parameters.values() for value in (parameter.mean, parameter.log_variance)
    )
    objective_gradients = torch.autograd.grad(
        tuple(cavity_mean + cavity_variance + [cavity.log_field]),
        posterior_coordinates,
        grad_outputs=tuple(
            mean_adjoint
            + variance_adjoint
            + [cavity.log_field.new_tensor(1 / power)]
        ),
        allow_unused=False,
    )
    mean_gradient, variance_gradient = {}, {}
    for (name, parameter), gm, ga in zip(
        parameters.items(), objective_gradients[0::2], objective_gradients[1::2]
    ):
        # ``ga`` is d(max objective)/d log(v). EPAdam expects d(min loss)/dv.
        mean_gradient[name] = -gm
        variance_gradient[name] = -ga / parameter.variance
    loss = -value - cavity.log_field / power
    return EPGradients(loss, mean_gradient, variance_gradient, messages,
                       normalization_observations(graph, passed.states))


def sampled_ep_gradients(
    graph: MomentDAG,
    inputs: Tensor,
    labels: Tensor,
    observations: int,
    alpha: float,
    samples: int = 4,
    power: float = 1.0,
    generator: torch.Generator | None = None,
    scale: float = 1.0,
) -> EPGradients:
    """Reparameterized reference for the same tied power-EP energy.

    ``scale`` multiplies the likelihood sum only. Training uses ``1/B``;
    the one global external field must not be divided by the minibatch size.
    """
    if samples <= 0:
        raise ValueError("samples must be positive")
    parameters = dict(named_gaussian_parameters(graph))
    cavity = power_cavity(parameters, observations, alpha)
    likelihoods, statistics = [], []
    for _ in range(samples):
        replacements = {}
        for name, belief in cavity.beliefs.items():
            epsilon = torch.randn(
                belief.mean.shape,
                dtype=belief.mean.dtype,
                device=belief.mean.device,
                generator=generator,
            )
            replacements[f"layers.{name}.mean"] = belief.mean + belief.variance.sqrt() * epsilon
        states = functional_call(graph, replacements, (inputs,), {"return_states": True}, strict=False)
        logits = states[graph.nodes[-1].name]
        statistics.append(normalization_observations(graph, states))
        likelihoods.append(-F.cross_entropy(logits, labels, reduction="none"))
    log_likelihood = torch.stack(likelihoods)
    log_normalizer = (
        torch.logsumexp(power * log_likelihood, 0) - math.log(samples)
    ).sum() / power
    loss = -scale * log_normalizer - cavity.log_field / power
    coordinates = tuple(
        value for parameter in parameters.values() for value in (parameter.mean, parameter.log_variance)
    )
    derivatives = torch.autograd.grad(loss, coordinates)
    mean_gradient, variance_gradient = {}, {}
    for (name, parameter), gm, ga in zip(
        parameters.items(), derivatives[0::2], derivatives[1::2]
    ):
        mean_gradient[name] = gm.detach()
        variance_gradient[name] = (ga / parameter.variance).detach()
    return EPGradients(loss.detach(), mean_gradient, variance_gradient, None,
                       average_observations(statistics))


def moment_vadam_second(
    graph: MomentDAG,
    inputs: Tensor,
    labels: Tensor,
    mean_gradient: dict[str, Tensor],
    generator: torch.Generator | None = None,
) -> dict[str, Tensor]:
    """One-direction curvature statistic used by moment Vadam.

    If ``g(m)`` is the point-network minibatch gradient and
    ``d=sqrt(v) epsilon``, the local Gaussian approximation gives

    ``E[g(W)^2] approximately E[g(W)]^2 + (H(m)d)^2``.

    This is the deterministic counterpart of the sampled Vadam second moment;
    the optimizer's exponential accumulator remains a separate state.
    """
    parameters = dict(named_gaussian_parameters(graph))
    means = tuple(parameter.mean for parameter in parameters.values())
    directions = tuple(
        parameter.variance.sqrt()
        * (
            2
            * torch.randint(
                0,
                2,
                parameter.mean.shape,
                device=parameter.mean.device,
                generator=generator,
            ).to(parameter.mean.dtype)
            - 1
        )
        for parameter in parameters.values()
    )

    names = tuple(parameters)

    def point_loss(*candidate_means: Tensor) -> Tensor:
        replacements = {
            f"layers.{name}.mean": candidate for name, candidate in zip(names, candidate_means)
        }
        logits = functional_call(graph, replacements, (inputs,), strict=False)
        return F.cross_entropy(logits, labels)

    gradient_function = grad(point_loss, argnums=tuple(range(len(means))))
    _, hessian_direction = jvp(gradient_function, means, directions)
    return {
        name: mean_gradient[name].square() + hd.detach().square()
        for name, hd in zip(names, hessian_direction)
    }


def moment_likelihood_gradients(
    graph: MomentDAG,
    inputs: Tensor,
    labels: Tensor,
    method: str = "mfvi",
    power: float = 1.0,
    scale: float = 1.0,
    head: str = "quadrature",
    rank: int = 0,
    seed: int = 41,
    full_readout: bool = False,
) -> MomentGradients:
    """Forward moments -> method-specific head -> explicit reverse messages.

    The returned ordinary derivatives minimize the negative likelihood
    objective.  The unnegated moment signals remain available for canonical
    MFVI or finite EP updates.
    """
    if rank:
        if head != "quadrature":
            raise ValueError("the shared-covariance head uses contrast quadrature")
        value, weights, adjoints, signals, passed = shared_likelihood(
            graph,inputs,labels,rank,seed,method=method,power=power,scale=scale)
        return MomentGradients(value,{n:-g[0] for n,g in weights.items()},
            {n:-g[1] for n,g in weights.items()},FactorMessages(signals,adjoints),
            normalization_observations(graph,passed.states))
    observed = DiagonalMoments(inputs, torch.zeros_like(inputs))
    passed = graph.forward_moments(observed, literal=False, full_readout=full_readout)
    if full_readout:
        child,value=torch.func.grad_and_value(lambda m,r:root_objective(
            RootMoments(m,r),labels,method,power,scale),argnums=(0,1))(*passed.output)
        messages=passed.backward(child)
    elif head == "quadrature":
        result = categorical_quadrature_head(passed.output, labels, method, power, scale)
    elif head == "second_order" and method == "mfvi":
        result = mfvi_categorical_head(passed.output, labels, scale)
    else:
        raise ValueError("use quadrature for MFVI/EP, or second_order for MFVI")
    if not full_readout:
        value,messages=result.value,passed.backward(result.signal)
    parameters = dict(named_gaussian_parameters(graph))
    mean_gradient, variance_gradient = {}, {}
    for name, parameter in parameters.items():
        ordinary = decode_diagonal(parameter.moments(), messages.parameters[name])
        # Heads maximize log likelihood; optimizers in optimizers.py minimize.
        mean_gradient[name] = -ordinary.mean
        variance_gradient[name] = -ordinary.variance
    return MomentGradients(value, mean_gradient, variance_gradient, messages,
                           normalization_observations(graph, passed.states))


def sampled_likelihood_gradients(
    graph: MomentDAG,
    inputs: Tensor,
    labels: Tensor,
    samples: int = 1,
    generator: torch.Generator | None = None,
) -> SampledGradients:
    """Reparameterized whole-network draws and Vadam gradient moments.

    One sampled parameter tensor is reused for the complete minibatch.  Drawing
    independent weights for examples would estimate a different model.
    """
    if samples <= 0:
        raise ValueError("samples must be positive")
    parameters = dict(named_gaussian_parameters(graph))
    accumulated = {name: torch.zeros_like(p.mean) for name, p in parameters.items()}
    second = {name: torch.zeros_like(p.mean) for name, p in parameters.items()}
    losses, statistics = [], []
    for _ in range(samples):
        draws, replacements = {}, {}
        for name, parameter in parameters.items():
            epsilon = torch.randn(
                parameter.mean.shape,
                dtype=parameter.mean.dtype,
                device=parameter.mean.device,
                generator=generator,
            )
            draw = (parameter.mean.detach() + parameter.variance.detach().sqrt() * epsilon).requires_grad_()
            draws[name] = draw
            replacements[f"layers.{name}.mean"] = draw
        states = functional_call(graph, replacements, (inputs,), {"return_states": True}, strict=False)
        logits = states[graph.nodes[-1].name]
        statistics.append(normalization_observations(graph, states))
        loss = F.cross_entropy(logits, labels)
        gradients = torch.autograd.grad(loss, tuple(draws.values()))
        losses.append(loss.detach())
        for (name, _), gradient in zip(draws.items(), gradients):
            accumulated[name].add_(gradient.detach() / samples)
            second[name].add_(gradient.detach().square() / samples)
    return SampledGradients(torch.stack(losses).mean(), accumulated, second,
                            average_observations(statistics))


@torch.no_grad()
def sampled_predict(
    graph: MomentDAG,
    inputs: Tensor,
    samples: int = 32,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Bayesian model average: average probabilities from complete networks."""
    parameters = dict(named_gaussian_parameters(graph))
    probability = 0
    for _ in range(samples):
        replacements = {}
        for name, parameter in parameters.items():
            epsilon = torch.randn(
                parameter.mean.shape,
                dtype=parameter.mean.dtype,
                device=parameter.mean.device,
                generator=generator,
            )
            replacements[f"layers.{name}.mean"] = parameter.mean + parameter.variance.sqrt() * epsilon
        probability = probability + functional_call(graph, replacements, (inputs,), strict=False).softmax(-1)
    return probability / samples


@torch.no_grad()
def moment_predict(graph: MomentDAG, inputs: Tensor, rank: int = 0, seed: int = 41, full_readout=False) -> Tensor:
    """Single-pass ``+/-sqrt(K)`` integration of the diagonal logit belief."""
    if rank:
        z = shared_forward(graph, inputs, rank, seed).output
        return root_nodes(z).softmax(-1).mean(0)
    z = graph.forward_moment_values(DiagonalMoments(inputs, torch.zeros_like(inputs)),full_readout=full_readout)
    if full_readout:return root_nodes(z).softmax(-1).mean(0)
    classes = z.mean.shape[-1]
    eye = torch.eye(classes, dtype=z.mean.dtype, device=z.mean.device)
    offset = math.sqrt(classes) * z.variance.sqrt().unsqueeze(0) * eye[:, None, :]
    nodes = torch.cat((z.mean.unsqueeze(0) + offset, z.mean.unsqueeze(0) - offset), 0)
    return nodes.softmax(-1).mean(0)
