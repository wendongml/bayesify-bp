"""MFVI expected-log-likelihood and EP log-normalizer heads."""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor

from .geometry import DiagonalMoments, EuclideanAdjoint, MomentSignal, encode_diagonal


class HeadResult(NamedTuple):
    value: Tensor
    signal: MomentSignal


def mfvi_gaussian_head(
    z: DiagonalMoments, y: Tensor, noise_variance: float = 1.0, scale: float = 1.0
) -> HeadResult:
    """Exact ``E_q[log N(y; z, R)]`` from two moments."""
    residual = y - z.mean
    noise = torch.as_tensor(noise_variance, dtype=z.mean.dtype, device=z.mean.device)
    value = -scale * torch.sum(
        (residual.square() + z.variance) / noise + torch.log(2 * math.pi * noise)
    ) / 2
    adjoint = EuclideanAdjoint(
        scale * residual / noise,
        -scale * torch.ones_like(z.variance) / (2 * noise),
    )
    return HeadResult(value, encode_diagonal(z, adjoint))


def ep_gaussian_head(
    z: DiagonalMoments, y: Tensor, noise_variance: float = 1.0, scale: float = 1.0
) -> HeadResult:
    """Exact Gaussian tilted normalizer ``log N(y; m, R+v)``."""
    noise = torch.as_tensor(noise_variance, dtype=z.mean.dtype, device=z.mean.device)
    total = noise + z.variance
    residual = y - z.mean
    value = -scale * torch.sum(torch.log(2 * math.pi * total) + residual.square() / total) / 2
    adjoint = EuclideanAdjoint(
        scale * residual / total,
        scale * (residual.square() / total.square() - 1 / total) / 2,
    )
    return HeadResult(value, encode_diagonal(z, adjoint))


def mfvi_categorical_head(z: DiagonalMoments, labels: Tensor, scale: float = 1.0) -> HeadResult:
    """Declared second-order approximation to ``E_q[log softmax(z)_y]``."""
    probabilities = z.mean.softmax(-1)
    targets = F.one_hot(labels, z.mean.shape[-1]).to(z.mean.dtype)
    correction = z.variance * (1 - 2 * probabilities)
    mean_adjoint = scale * (
        targets
        - probabilities
        - 0.5
        * probabilities
        * (correction - (probabilities * correction).sum(-1, keepdim=True))
    )
    variance_adjoint = -0.5 * scale * probabilities * (1 - probabilities)
    value = scale * (
        (targets * z.mean).sum()
        - torch.logsumexp(z.mean, -1).sum()
        - 0.5 * (z.variance * probabilities * (1 - probabilities)).sum()
    )
    return HeadResult(value, encode_diagonal(z, EuclideanAdjoint(mean_adjoint, variance_adjoint)))


def categorical_quadrature_head(
    z: DiagonalMoments,
    labels: Tensor,
    method: str = "mfvi",
    power: float = 1.0,
    scale: float = 1.0,
) -> HeadResult:
    """Deterministic probability integration for categorical MFVI or power EP.

    For ``K`` diagonal Gaussian logits, the ``+/- sqrt(K)`` coordinate axes
    have exactly the requested mean and covariance.  They define ``2K``
    deterministic integration nodes. MFVI averages ``log p_y``; power EP uses

    ``log Z_alpha = log mean exp(alpha log p_y) / alpha``.

    Differentiating this declared scalar objective supplies the ordinary
    mean/variance adjoints; the same Fisher encoder starts the reverse pass.
    ``method='ep'`` is therefore not an MFVI loss with a renamed update.
    """
    if method not in {"mfvi", "ep"} or power <= 0:
        raise ValueError("method must be 'mfvi' or 'ep', with positive power")

    def objective(mean: Tensor, variance: Tensor) -> Tensor:
        classes = mean.shape[-1]
        eye = torch.eye(classes, dtype=mean.dtype, device=mean.device)
        offset = math.sqrt(classes) * variance.sqrt().unsqueeze(0) * eye[:, None, :]
        nodes = torch.cat((mean.unsqueeze(0) + offset, mean.unsqueeze(0) - offset), 0)
        log_probability = nodes.log_softmax(-1)
        index = labels[None, :, None].expand(2 * classes, -1, 1)
        log_likelihood = log_probability.gather(-1, index).squeeze(-1)
        if method == "mfvi":
            return scale * log_likelihood.mean(0).sum()
        return scale * (
            torch.logsumexp(power * log_likelihood, 0) - math.log(2 * classes)
        ).sum() / power

    (mean_adjoint, variance_adjoint), value = torch.func.grad_and_value(objective, argnums=(0, 1))(
        z.mean, z.variance
    )
    return HeadResult(value, encode_diagonal(z, EuclideanAdjoint(mean_adjoint, variance_adjoint)))
def sampled_readout_adjoint(features, logits, label, method="mfvi"):
    """Per-network derivatives for an independent Gaussian affine readout.

    Condition on h. For z=Wh+b, Price's identity gives
    d_m E[f]=E[h df/dz], d_v E[f]=E[h² d²f/dz²]/2.
    log p_y has r=e_y-p, diag(H)=-p(1-p); p_y has Hessian
    p_y(H+rrᵀ). EP must average these unnormalized derivatives and p_y
    before dividing. Shapes: h[S,I], z[S,O], gm/gv[S,O,I].
    The hidden activations need not be Gaussian.
    """
    logp=logits.log_softmax(-1);p=logp.exp()
    r=torch.nn.functional.one_hot(torch.as_tensor(label,device=p.device),p.shape[-1]).to(p.dtype)-p
    H=-p*(1-p)
    if method=="mfvi":value=logp[:,label];a=torch.ones_like(value)
    elif method=="ep":value=p[:,label];a=value;H=H+r.square()
    else:raise ValueError("use mfvi or ep")
    h=features[:,None,:];a=a[:,None,None]
    return value,a*r[:,:,None]*h,.5*a*H[:,:,None]*h.square()
