"""Paired nonlinear correction to Gaussian forward/backward integration.

For a FROZEN linear surrogate h0 and integration proposal r,
E_q[h] = E_q[h0] + E_r[(q/r)(h-h0)], also for pathwise derivatives.
MFVI uses h=log p; EP uses h=p and divides dZ by Z AFTER correction.
The proposal is an integration device; the network belief remains one Gaussian.
This module mirrors the residual integration equations in R/covariance.R.
"""

from typing import NamedTuple
import torch
from torch import Tensor
from .geometry import DiagonalMoments, EuclideanAdjoint, MomentSignal, encode_diagonal


class ResidualResult(NamedTuple):
    gradient: EuclideanAdjoint
    signal: MomentSignal
    normalizer: Tensor


def residual_backward(q, gaussian, nonlinear, surrogate, weight, method="mfvi"):
    """Correct unnormalized expectations, then encode the Gaussian message.

    Each dictionary contains gm, gv; the sampled ones also contain
    log_likelihood. Sample arrays are [draw, parameter]. Gaussian gm/gv
    integrate derivatives of E[h0], NOT of log E[h0] for EP.
    Integration nodes, proposal ratios and the surrogate stay fixed during
    this derivative calculation. No derivative of a fitted proposal is added.
    """
    if method not in ("mfvi", "ep") or torch.any(q.variance <= 0):
        raise ValueError("use mfvi/ep and a positive-variance Gaussian")
    if torch.any(weight < 0) or not torch.isfinite(weight).all():
        raise ValueError("importance ratios must be finite and nonnegative")
    a = nonlinear["log_likelihood"].exp() if method == "ep" else torch.ones_like(weight)
    b = surrogate["log_likelihood"].exp() if method == "ep" else torch.ones_like(weight)
    gm = gaussian["gm"] + (weight[:, None] * (
        a[:, None] * nonlinear["gm"] - b[:, None] * surrogate["gm"])).mean(0)
    gv = gaussian["gv"] + (weight[:, None] * (
        a[:, None] * nonlinear["gv"] - b[:, None] * surrogate["gv"])).mean(0)
    Z = gaussian["Z"] + (weight * (nonlinear["log_likelihood"].expm1()
        - surrogate["log_likelihood"].expm1())).mean() if method == "ep" else q.mean.new_tensor(1.)
    if not torch.isfinite(Z) or Z <= 0 or not torch.isfinite(gm).all() or not torch.isfinite(gv).all():
        raise ValueError("invalid residual estimate; do not clip a failed EP normalizer")
    gradient = EuclideanAdjoint(gm / Z, gv / Z)
    return ResidualResult(gradient, encode_diagonal(q, gradient), Z)


def network_residual_backward(predict, q, label, *, pairs=128, head_power=14, seed=173):
    """A query's logits and weight signals, using a frozen full-class surrogate.

    predict(w) accepts a FLAT parameter vector and returns K logits. Its
    K-by-P Jacobian is intended for diagnostics/small query sets, not a dense
    P-by-P approximation. Each antithetic group uses the SAME weight noise
    in the nonlinear and surrogate paths. Returned methods are separate
    MFVI E[log p] and EP log E[p] calculations at the supplied belief.
    """
    if q.mean.ndim != 1 or q.mean.shape != q.variance.shape or torch.any(q.variance <= 0):
        raise ValueError("q must be a flat, positive-variance Gaussian")
    if pairs < 1 or head_power < 1:
        raise ValueError("positive integration budgets are required")
    # Detaching freezes the reference point, J, variance and proposal. The
    # pathwise derivatives below act on w only, not on how J was constructed.
    m, v = q.mean.detach(), q.variance.detach()
    q = DiagonalMoments(m, v)
    z = predict(m).detach()
    J = torch.func.jacrev(predict)(m).detach()
    K = z.numel()
    if z.ndim != 1 or not 0 <= label < K or K < 2:
        raise ValueError("predict must return a vector of at least two logits")
    C = (J * v) @ J.T
    other = [i for i in range(K) if i != label]
    eye = torch.eye(K, device=m.device, dtype=m.dtype)
    D = eye[other] - eye[label]
    a = D @ z
    # No artificial jitter: a singular contrast covariance is a different
    # integration problem and must be reported, not silently changed.
    L = torch.linalg.cholesky(D @ C @ D.T)
    competitor_index = int(a.argmax())
    delta = L[competitor_index]
    u = torch.quasirandom.SobolEngine(K - 1, scramble=True, seed=seed).draw_base2(
        head_power - 1, dtype=m.dtype).to(m.device)
    normal = torch.special.ndtri(u)
    nodes = torch.cat((normal, normal + delta))
    # r(e) = [phi(e) + phi(e-delta)]/2; hence phi/r = 2 sigmoid(delta²/2-e.delta).
    ratio = 2 * torch.sigmoid(delta.square().sum() / 2 - nodes @ delta)
    logits = (a + nodes @ L.T) @ eye[other]
    p = logits.softmax(-1)
    score = eye[label] - p
    H = p[:, :, None] * p[:, None, :] - torch.diag_embed(p)
    py = p[:, label]

    shift = v.sqrt() * (J[other[competitor_index]] - J[label])
    generator = torch.Generator(device=m.device).manual_seed(seed + 1)
    logp = lambda w: predict(w).log_softmax(-1)[label]
    # Stream independent groups. Keeping all S-by-P weights/gradients at once
    # adds no statistical information and can exceed accelerator memory.
    correction = {method:[torch.zeros_like(m),torch.zeros_like(v),m.new_tensor(0.)]
                  for method in ("mfvi","ep")}
    for begin in range(0,pairs,16):
        e = torch.randn(min(16,pairs-begin),m.numel(),device=m.device,dtype=m.dtype,generator=generator)
        # Four dependent nodes per independent group.
        e = torch.stack((e,-e,e+shift,-e+shift),1).flatten(0,1)
        weight = 2*torch.sigmoid(shift.square().sum()/2-e@shift)
        weights = m+v.sqrt()*e
        ell = torch.func.vmap(logp,chunk_size=16)(weights).detach()
        g = torch.func.vmap(torch.func.grad(logp),chunk_size=16)(weights).detach()
        lp0 = (z+(e*v.sqrt())@J.T).log_softmax(-1)
        g0 = (eye[label]-lp0.exp())@J
        for method in correction:
            a=ell.exp() if method=="ep" else torch.ones_like(ell)
            b=lp0[:,label].exp() if method=="ep" else torch.ones_like(ell)
            difference=weight[:,None]*(a[:,None]*g-b[:,None]*g0)
            correction[method][0]+=difference.sum(0)/(4*pairs)
            correction[method][1]+=(difference*e/(2*v.sqrt())).sum(0)/(4*pairs)
            if method=="ep":correction[method][2]+=(weight*(ell.expm1()-lp0[:,label].expm1())).sum()/(4*pairs)
    result = {}
    for method in ("mfvi", "ep"):
        factor = py if method == "ep" else torch.ones_like(py)
        gz = (ratio[:, None] * factor[:, None] * score).mean(0)
        hessian = H + score[:, :, None] * score[:, None, :] if method == "ep" else H
        GC = (ratio[:, None, None] * factor[:, None, None] * hessian).mean(0) / 2
        # Price's identity: d E[h]/dC = E[Hessian h]/2.
        gaussian = dict(gm=J.T @ gz, gv=torch.einsum("ik,ij,jk->k", J, GC, J),
                        Z=(ratio * py).mean() if method == "ep" else m.new_tensor(1.))
        dm,dv,dZ=correction[method];Z=gaussian["Z"]+dZ
        gradient=EuclideanAdjoint((gaussian["gm"]+dm)/Z,(gaussian["gv"]+dv)/Z)
        if not torch.isfinite(Z) or Z<=0 or not all(torch.isfinite(g).all() for g in gradient):
            raise ValueError("invalid residual estimate; do not clip a failed EP normalizer")
        result[method]=ResidualResult(gradient,encode_diagonal(q,gradient),Z)
    return result
