"""Gaussian optimizer recurrences for native PyTorch moment experiments.

These states are deliberately separate:

* a learned posterior variance belongs to the Gaussian belief;
* Adam/Vadam's exponential second accumulator belongs to the optimizer.

Confusing the two was one of the failure modes the tutorial is meant to expose.
All gradients accepted here minimize a loss; the message heads themselves use
maximized objectives and are negated by the training adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor

from .geometry import DiagonalMoments
from .layers import GaussianParameter


@dataclass(frozen=True)
class PowerCavity:
    beliefs: dict[str, DiagonalMoments]
    log_field: Tensor


def power_cavity(
    parameters: Mapping[str, GaussianParameter], observations: int, alpha: float
) -> PowerCavity:
    """Remove an ``alpha/N`` share of the posterior and restore the prior.

    In information coordinates, with ``k=alpha/N``,

    ``eta_cavity = (1-k) eta_q + k eta_prior``.

    The returned ``log_field`` is the corresponding Gaussian normalizer term;
    it is applied once globally, not once per minibatch example.
    """
    if observations <= 0 or not 0 <= alpha <= observations:
        raise ValueError("require observations>0 and 0<=alpha<=observations")
    k = alpha / observations
    beliefs: dict[str, DiagonalMoments] = {}
    first = next(iter(parameters.values())).mean
    log_field = first.new_zeros(())
    for name, parameter in parameters.items():
        q, prior = parameter.moments(), parameter.prior()
        e = k * (q.variance / prior.variance - 1)
        denominator = 1 + e
        if torch.any(denominator <= 0):
            raise ValueError(f"invalid cavity precision for {name}")
        variance = q.variance / denominator
        mean = prior.mean + (1 - k) * (q.mean - prior.mean) / denominator
        beliefs[name] = DiagonalMoments(mean, variance)
        log_field = log_field - 0.5 * torch.sum(
            torch.log1p(e)
            - k * torch.log(q.variance / prior.variance)
            + k
            * (1 - k)
            * (q.mean - prior.mean).square()
            / (prior.variance * denominator)
        )
    return PowerCavity(beliefs, log_field)


def gaussian_kl(parameters: Sequence[GaussianParameter]) -> Tensor:
    """``KL[N(m,v) || N(m0,v0)]``, summed over independent blocks."""
    if not parameters:
        raise ValueError("at least one Gaussian parameter is required")
    answer = parameters[0].mean.new_zeros(())
    for parameter in parameters:
        q, p = parameter.moments(), parameter.prior()
        answer = answer + 0.5 * torch.sum(
            (q.variance + (q.mean - p.mean).square()) / p.variance
            - 1
            + torch.log(p.variance / q.variance)
        )
    return answer


class Vadam:
    """Published diagonal Vadam recurrence, written without optimizer wrappers.

    Let ``N_eff=N/tau`` and ``lambda=1/(N_eff v0)``. For likelihood-gradient
    moments ``g=E[grad loss]`` and ``g2=E[(grad loss)^2]``:

    ``r <- b1 r + (1-b1) [g + lambda (m-m0)]``
    ``s <- b2 s + (1-b2) g2``
    ``m <- m - lr r_hat/(sqrt(s_hat)+lambda)``
    ``v <- 1/(N_eff s + 1/v0)``.
    """

    def __init__(
        self,
        parameters: Sequence[GaussianParameter],
        observations: int,
        temperature: float,
        learning_rate: float,
        betas: tuple[float, float] = (0.9, 0.99),
    ):
        if observations <= 0 or temperature <= 0:
            raise ValueError("observations and temperature must be positive")
        self.parameters = list(parameters)
        self.effective_n = observations / temperature
        self.learning_rate, self.betas, self.steps = learning_rate, betas, 0
        self.first = [torch.zeros_like(p.mean) for p in self.parameters]
        self.second = [
            (1 / p.variance.detach() - 1 / p.prior_variance) / self.effective_n
            for p in self.parameters
        ]

    @torch.no_grad()
    def step(self, gradient: Sequence[Tensor], gradient_second: Sequence[Tensor]) -> None:
        if not (len(gradient) == len(gradient_second) == len(self.parameters)):
            raise ValueError("one first and second gradient moment is required per parameter")
        self.steps += 1
        beta1, beta2 = self.betas
        for parameter, g, g2, first, second in zip(
            self.parameters, gradient, gradient_second, self.first, self.second
        ):
            regularization = 1 / (self.effective_n * parameter.prior_variance)
            first.mul_(beta1).add_((1 - beta1) * (g + regularization * (parameter.mean - parameter.prior_mean)))
            second.mul_(beta2).add_((1 - beta2) * g2)
            first_hat = first / (1 - beta1**self.steps)
            second_hat = second / (1 - beta2**self.steps)
            parameter.mean.add_(
                -self.learning_rate * first_hat / (second_hat.sqrt() + regularization)
            )
            variance = 1 / (self.effective_n * second + 1 / parameter.prior_variance)
            if torch.any(variance <= 0):
                raise ValueError("Vadam produced an invalid posterior variance")
            parameter.log_variance.copy_(variance.log())


class EPAdam:
    """Separate Adam recurrences for Gaussian means and log variances.

    If ``g_v=d loss/dv`` and ``a=log(v)``, the chain rule gives
    ``d loss/da = v*g_v``.  This quantity is distinct from Adam's second
    accumulator and from Vadam's ``E[g^2]``.
    """

    def __init__(
        self,
        parameters: Sequence[GaussianParameter],
        learning_rate: float,
        variance_learning_rate: float,
        betas: tuple[float, float] = (0.9, 0.999),
        epsilon: float = 1e-8,
    ):
        self.parameters = list(parameters)
        self.learning_rate, self.variance_learning_rate = learning_rate, variance_learning_rate
        self.betas, self.epsilon, self.steps = betas, epsilon, 0
        self.mean_first = [torch.zeros_like(p.mean) for p in self.parameters]
        self.mean_second = [torch.zeros_like(p.mean) for p in self.parameters]
        self.var_first = [torch.zeros_like(p.mean) for p in self.parameters]
        self.var_second = [torch.zeros_like(p.mean) for p in self.parameters]

    @torch.no_grad()
    def step(self, mean_gradient: Sequence[Tensor], variance_gradient: Sequence[Tensor]) -> None:
        if not (len(mean_gradient) == len(variance_gradient) == len(self.parameters)):
            raise ValueError("one mean and variance gradient is required per parameter")
        self.steps += 1
        beta1, beta2 = self.betas
        for p, gm, gv, rm, sm, rv, sv in zip(
            self.parameters,
            mean_gradient,
            variance_gradient,
            self.mean_first,
            self.mean_second,
            self.var_first,
            self.var_second,
        ):
            ga = p.variance * gv
            rm.mul_(beta1).add_((1 - beta1) * gm)
            sm.mul_(beta2).add_((1 - beta2) * gm.square())
            rv.mul_(beta1).add_((1 - beta1) * ga)
            sv.mul_(beta2).add_((1 - beta2) * ga.square())
            rmh, smh = rm / (1 - beta1**self.steps), sm / (1 - beta2**self.steps)
            rvh, svh = rv / (1 - beta1**self.steps), sv / (1 - beta2**self.steps)
            p.mean.add_(-self.learning_rate * rmh / (smh.sqrt() + self.epsilon))
            p.log_variance.add_(
                -self.variance_learning_rate * rvh / (svh.sqrt() + self.epsilon)
            )
