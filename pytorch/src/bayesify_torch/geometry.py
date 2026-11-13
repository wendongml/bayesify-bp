"""Gaussian Fisher-coordinate conversions and finite belief updates.

All objectives are maximized.  Diagonal beliefs store ``(m, v)`` while their
raw second moment is ``s = m**2 + v``.  The formulas match the paper directly:

    dm = v * dPhi/dm
    ds = 2*m*dm + 2*v**2 * dPhi/dv.

The centered quantity ``ds - 2*m*dm`` is retained explicitly to avoid a
catastrophic subtraction when variances are small.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import Tensor


class DiagonalMoments(NamedTuple):
    mean: Tensor
    variance: Tensor


class EuclideanAdjoint(NamedTuple):
    mean: Tensor
    variance: Tensor


class MomentSignal(NamedTuple):
    delta_mean: Tensor
    delta_raw_second: Tensor
    centered: Tensor
    mean_at_zero: Tensor
    variance_at_zero: Tensor


class FullMoments(NamedTuple):
    mean: Tensor
    covariance: Tensor


class FullMomentSignal(NamedTuple):
    delta_mean: Tensor
    delta_raw_second: Tensor


def _require_nonnegative(x: Tensor, name: str) -> None:
    if torch.any(x < 0):
        raise ValueError(f"{name} must be nonnegative")


def _require_positive(x: Tensor, name: str) -> None:
    if torch.any(x <= 0):
        raise ValueError(f"{name} must be positive")


def encode_diagonal(q: DiagonalMoments, g: EuclideanAdjoint) -> MomentSignal:
    """Apply the diagonal-Gaussian Fisher matrix to an ordinary adjoint."""
    _require_nonnegative(q.variance, "variance")
    # Decoding divides the centered signal by ``2 v^2``.  When ``v`` is below
    # sqrt(the smallest normal number), ``v^2`` underflows even though ``v``
    # itself is nonzero.  Such a coordinate is numerically deterministic; keep
    # its ordinary adjoint in the explicit zero-Fisher fields instead of
    # creating a 0/0 during a later reverse traversal.
    active = q.variance > math.sqrt(torch.finfo(q.variance.dtype).tiny)
    dm = torch.where(active, q.variance * g.mean, torch.zeros_like(g.mean))
    centered = torch.where(
        active, 2 * q.variance.square() * g.variance, torch.zeros_like(g.variance)
    )
    return MomentSignal(
        dm,
        2 * q.mean * dm + centered,
        centered,
        torch.where(active, torch.zeros_like(g.mean), g.mean),
        torch.where(active, torch.zeros_like(g.variance), g.variance),
    )


def decode_diagonal(q: DiagonalMoments, d: MomentSignal) -> EuclideanAdjoint:
    """Invert ``encode_diagonal`` on positive-variance coordinates."""
    _require_nonnegative(q.variance, "variance")
    positive = q.variance > math.sqrt(torch.finfo(q.variance.dtype).tiny)
    safe = torch.where(positive, q.variance, torch.ones_like(q.variance))
    return EuclideanAdjoint(
        torch.where(positive, d.delta_mean / safe, d.mean_at_zero),
        torch.where(positive, d.centered / (2 * safe.square()), d.variance_at_zero),
    )


def add_signals(a: MomentSignal | None, b: MomentSignal) -> MomentSignal:
    """Add branch messages at one frozen forward snapshot."""
    if a is None:
        return b
    return MomentSignal(*(x + y for x, y in zip(a, b)))


def encode_full(q: FullMoments, g: tuple[Tensor, Tensor]) -> FullMomentSignal:
    """Full-covariance counterpart of the diagonal Fisher conversion."""
    gm, gc = g
    gc = (gc + gc.mT) / 2
    dm = q.covariance @ gm
    dM = (
        torch.outer(q.mean, dm)
        + torch.outer(dm, q.mean)
        + 2 * q.covariance @ gc @ q.covariance
    )
    return FullMomentSignal(dm, dM)


def decode_full(q: FullMoments, d: FullMomentSignal) -> tuple[Tensor, Tensor]:
    """Decode with Cholesky solves; no explicit inverse or hidden jitter."""
    chol = torch.linalg.cholesky(q.covariance)

    def solve(rhs: Tensor) -> Tensor:
        vector = rhs.ndim == 1
        answer = torch.cholesky_solve(rhs[:, None] if vector else rhs, chol)
        return answer[:, 0] if vector else answer

    gm = solve(d.delta_mean)
    residual = (
        d.delta_raw_second
        - torch.outer(q.mean, d.delta_mean)
        - torch.outer(d.delta_mean, q.mean)
    )
    gc = solve(solve(residual).mT).mT / 2
    return gm, (gc + gc.mT) / 2


def tilted_update(q: DiagonalMoments, d: MomentSignal, rho: float = 1.0) -> DiagonalMoments:
    """Install an EP tilted-moment correction in raw-moment coordinates."""
    mean = q.mean + rho * d.delta_mean
    variance = q.variance + q.mean.square() + rho * d.delta_raw_second - mean.square()
    _require_positive(variance, "updated EP variance")
    return DiagonalMoments(mean, variance)


def canonical_mfvi_update(
    q: DiagonalMoments,
    likelihood_signal: MomentSignal,
    prior: DiagonalMoments,
    rho: float = 1.0,
) -> DiagonalMoments:
    """One canonical-coordinate natural-gradient MFVI update.

    ``tau+ = (1-rho)/v + rho*(1/v0 - 2*g_v)`` and
    ``h+ = (1-rho)*m/v + rho*(m0/v0 + g_m - 2*m*g_v)``.
    """
    _require_positive(q.variance, "MFVI variance")
    _require_positive(prior.variance, "prior variance")
    g = decode_diagonal(q, likelihood_signal)
    tau = (1 - rho) / q.variance + rho * (1 / prior.variance - 2 * g.variance)
    h = (1 - rho) * q.mean / q.variance + rho * (
        prior.mean / prior.variance + g.mean - 2 * q.mean * g.variance
    )
    _require_positive(tau, "updated MFVI precision")
    return DiagonalMoments(h / tau, 1 / tau)
