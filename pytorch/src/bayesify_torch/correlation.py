"""Shared covariance directions and structured Gaussian beliefs.

``SharedMoments(m,U)`` represents

    H = m + (1/sqrt(R)) sum_r U[r] epsilon[r],  epsilon ~ N(0,I_R),

so ``Cov(vec(H)) = U_flat.T @ U_flat / R``.  The same direction index follows
every branch, token and example; concatenation and residual addition therefore
retain their represented cross covariance.

The nonlinear rule is labelled: Gaussian ReLU means are exact for each scalar
Gaussian marginal, while multiplying every direction by ``Phi(m/s)`` keeps only
the first Hermite covariance component.  It is not called exact covariance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import NamedTuple


@dataclass(frozen=True)
class SharedMoments:
    mean: Tensor
    factors: Tensor  # [rank, *mean.shape]

    def __post_init__(self) -> None:
        if self.factors.ndim != self.mean.ndim + 1 or self.factors.shape[1:] != self.mean.shape:
            raise ValueError("factors must have shape [rank, *mean.shape]")

    @property
    def rank(self) -> int:
        return self.factors.shape[0]

    @property
    def variance(self) -> Tensor:
        return self.factors.square().mean(0)


@dataclass(frozen=True)
class StructuredGaussian:
    """One Gaussian ``N(m, diag(d) + L L^T)`` over a flattened parameter block."""

    mean: Tensor
    diagonal: Tensor
    factors: Tensor  # [rank, mean.numel()], rows are columns of L transposed

    def __post_init__(self) -> None:
        if self.diagonal.shape != self.mean.shape or torch.any(self.diagonal < 0):
            raise ValueError("diagonal residual must be nonnegative and match the mean")
        if self.factors.ndim != 2 or self.factors.shape[1] != self.mean.numel():
            raise ValueError("factors must have shape [rank, mean.numel()]")

    def covariance(self) -> Tensor:
        return torch.diag(self.diagonal.reshape(-1)) + self.factors.mT @ self.factors

    def rsample(self, samples: int = 1, generator: torch.Generator | None = None) -> Tensor:
        shape = (samples,) + self.mean.shape
        local = torch.randn(shape, dtype=self.mean.dtype, device=self.mean.device, generator=generator)
        global_noise = torch.randn(
            (samples, len(self.factors)),
            dtype=self.mean.dtype,
            device=self.mean.device,
            generator=generator,
        )
        correlated = (global_noise @ self.factors).reshape(shape)
        return self.mean + self.diagonal.sqrt() * local + correlated


def covariance(state: SharedMoments) -> Tensor:
    """Full covariance over the last coordinate, separately for each batch row."""
    if state.mean.ndim != 2:
        raise ValueError("the compact covariance helper expects [batch, features]")
    return torch.einsum("rbi,rbj->bij", state.factors, state.factors) / state.rank


def rademacher_directions(variance: Tensor, rank: int, seed: int) -> Tensor:
    """A deterministic unbiased covariance sketch: ``sqrt(v) * +/-1``."""
    generator = torch.Generator(device=variance.device).manual_seed(seed)
    signs = torch.randint(
        0, 2, (rank,) + variance.shape, generator=generator, device=variance.device
    ).to(variance.dtype)
    return variance.sqrt() * (2 * signs - 1)


def relu_shared(state: SharedMoments) -> SharedMoments:
    """Exact Gaussian marginal mean plus first-Hermite covariance directions.

    With ``s=sqrt(v)``, ``a=m/s``, ``P=Phi(a)`` and ``p=phi(a)``, the exact
    marginal mean is ``m P + s p``.  Price's identity gives the first-order
    response ``d E[ReLU(H)] / dm = P``; hence each shared direction becomes
    ``P U_r``.  Higher Hermite covariance components are deliberately omitted.
    """
    variance = state.variance
    positive = variance > torch.finfo(variance.dtype).tiny
    safe = torch.where(positive, variance, torch.ones_like(variance))
    scale = safe.sqrt()
    limit = 40.0 if variance.dtype == torch.float64 else 12.0
    central = positive & (state.mean.abs() <= limit * scale)
    mean0 = torch.where(central, state.mean, torch.zeros_like(state.mean))
    scale0 = torch.where(central, scale, torch.ones_like(scale))
    standardized = mean0 / scale0
    probability = 0.5 * torch.erfc(-standardized / math.sqrt(2))
    density = torch.exp(-0.5 * standardized.square()) / math.sqrt(2 * math.pi)
    mean = torch.where(central, mean0 * probability + scale0 * density, F.relu(state.mean))
    gate = torch.where(central, probability, (state.mean > 0).to(state.mean.dtype))
    return SharedMoments(mean, state.factors * gate.unsqueeze(0))


def linear_shared(
    x: SharedMoments,
    weight_mean: Tensor,
    weight_factors: Tensor,
    bias_mean: Tensor,
    bias_factors: Tensor,
) -> SharedMoments:
    """First-order shared-factor affine map.

    For direction ``r``, differentiating ``Y=X A^T+b`` gives
    ``U_y[r]=U_x[r] A^T + X U_A[r]^T + U_b[r]``.  The same ``r`` is used in
    all three terms; treating them as independent would discard covariance.
    """
    if not (x.rank == len(weight_factors) == len(bias_factors)):
        raise ValueError("all shared-factor ranks must agree")
    mean = F.linear(x.mean, weight_mean, bias_mean)
    inherited = F.linear(x.factors.flatten(0, 1), weight_mean).reshape(
        x.rank, *mean.shape
    )
    injected = torch.einsum("bi,roi->rbo", x.mean, weight_factors)
    return SharedMoments(mean, inherited + injected + bias_factors[:, None, :])


def conv2d_shared(
    x: SharedMoments,
    weight_mean: Tensor,
    weight_factors: Tensor,
    bias_mean: Tensor,
    bias_factors: Tensor,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
) -> SharedMoments:
    """Convolutional analogue of :func:`linear_shared`, direction by direction."""
    mean = F.conv2d(x.mean, weight_mean, bias_mean, stride=stride, padding=padding)
    inherited = F.conv2d(
        x.factors.flatten(0, 1), weight_mean, stride=stride, padding=padding
    ).reshape(x.rank, *mean.shape)
    injected = torch.stack(
        [F.conv2d(x.mean, weight_factors[r], stride=stride, padding=padding) for r in range(x.rank)]
    )
    return SharedMoments(mean, inherited + injected + bias_factors[:, None, :, None, None])


def add_shared(x: SharedMoments, y: SharedMoments) -> SharedMoments:
    if x.rank != y.rank:
        raise ValueError("shared-factor ranks must agree")
    return SharedMoments(x.mean + y.mean, x.factors + y.factors)


def concat_shared(x: SharedMoments, y: SharedMoments, dim: int) -> SharedMoments:
    if x.rank != y.rank:
        raise ValueError("shared-factor ranks must agree")
    # ``factors`` has one leading rank dimension, so activation dimension d is d+1.
    return SharedMoments(torch.cat((x.mean, y.mean), dim), torch.cat((x.factors, y.factors), dim + 1))


def average_pool_shared(x: SharedMoments, kernel_size: int = 2) -> SharedMoments:
    mean = F.avg_pool2d(x.mean, kernel_size)
    factors = F.avg_pool2d(x.factors.flatten(0, 1), kernel_size).reshape(x.rank, *mean.shape)
    return SharedMoments(mean, factors)


def flatten_shared(x: SharedMoments) -> SharedMoments:
    return SharedMoments(x.mean.flatten(1), x.factors.flatten(2))


def gaussian_readout(
    x: SharedMoments,
    weight_mean: Tensor,
    weight_variance: Tensor,
    bias_mean: Tensor,
    bias_variance: Tensor,
) -> tuple[Tensor, Tensor]:
    """Exact readout moments for the supplied hidden factors and diagonal head.

    ``Cov(z)=W Cov(h) W^T + Diag(E[h^2] Var(W)^T + Var(b))``.  This is exact
    for the represented hidden covariance; approximation occurred upstream.
    """
    mean = F.linear(x.mean, weight_mean, bias_mean)
    projected = torch.einsum("rbi,oi->rbo", x.factors, weight_mean)
    hidden_covariance = torch.einsum("rbi,rbj->bij", projected, projected) / x.rank
    hidden_second = x.mean.square() + x.variance
    independent = F.linear(hidden_second, weight_variance, bias_variance)
    return mean, hidden_covariance + torch.diag_embed(independent)


class RootMoments(NamedTuple):
    """Full classifier covariance C = root root^T, without losing small modes."""
    mean: Tensor
    root: Tensor

    @property
    def covariance(self):
        return self.root @ self.root.mT


class FactorMessages(NamedTuple):
    # Gaussian moment signals at weights; Euclidean (m,U) adjoints at factors.
    # A rank-deficient U U^T has no invertible activation Fisher block.
    parameters: dict
    activation_adjoints: dict


def shared_layer(layer, parents, beliefs, noise, *, readout=False):
    """Extend the SAME native layer using shared Gaussian directions.

    Hidden boundaries carry (m,U), U.shape=[rank,batch,...]. A terminal affine
    readout integrates its independent Gaussian weights analytically. Local
    VJPs act on these factors; a finite-rank covariance is never inverted.
    """
    from .layers import GaussianParameter, MomentReLU
    from torch.func import functional_call, jvp, vmap
    states = tuple(SharedMoments(*p) for p in parents)
    if isinstance(layer, MomentReLU):
        y = relu_shared(states[0]);return y.mean, y.factors
    names = tuple(n for n,p in layer.named_children() if isinstance(p,GaussianParameter))
    means = tuple(q[0] for q in beliefs)
    factors = tuple(q[1].sqrt() * e for q,e in zip(beliefs,noise))
    if readout:
        x = states[0];w,b = beliefs;rank = x.rank
        projected = F.linear(x.factors, w[0])
        diagonal = F.linear(x.mean.square()+x.variance, w[1], b[1])
        root = torch.cat((projected.permute(1,2,0)/math.sqrt(rank),
                          torch.diag_embed(diagonal.sqrt())), -1)
        return F.linear(x.mean,w[0],b[0]),root
    def point(*arguments):
        weights = {name+'.mean':a for name,a in zip(names,arguments[len(states):])}
        return functional_call(layer,weights,arguments[:len(states)],strict=False)
    primals = tuple(x.mean for x in states)+means
    tangents = tuple(x.factors for x in states)+factors
    mean = point(*primals)
    U = vmap(lambda *d:jvp(point,primals,d)[1])(*tangents)
    return mean,U


@dataclass(frozen=True)
class SharedPass:
    output: RootMoments
    states: dict
    records: tuple
    beliefs: dict

    def backward(self, child):
        """Local VJP traversal, then Gaussian encoding at persistent weights."""
        from .geometry import DiagonalMoments,EuclideanAdjoint,encode_diagonal
        pending={self.records[-1][0].name:child};weight={}
        def add(table,name,pair):
            table[name]=pair if name not in table else tuple(a+b for a,b in zip(table[name],pair))
        for node,names,parents,beliefs,local in reversed(self.records):
            if node.name not in pending:continue
            _,pull=torch.func.vjp(local,parents,beliefs)
            dp,dw=pull(pending[node.name])
            for n,g in zip(node.parents,dp):add(pending,n,g)
            for n,g in zip(names,dw):add(weight,n,g)
        signals={n:encode_diagonal(DiagonalMoments(*self.beliefs[n]),EuclideanAdjoint(*g))
                 for n,g in weight.items()}
        return weight,pending,signals


def shared_forward(graph,inputs,rank=4,seed=41,beliefs=None,noise=None):
    """Diagonal and correlated paths share the graph and parameter objects.

    Noise IDs are reused across branches. One immutable forward snapshot is
    retained for the complete reverse traversal. No optimizer or model copy
    is introduced by the covariance extension.
    """
    from .inference import named_gaussian_parameters
    from .layers import GaussianParameter,MomentLinear
    if rank<1:raise ValueError('rank must be positive')
    if beliefs is None:beliefs={n:(p.mean,p.variance) for n,p in named_gaussian_parameters(graph)}
    if noise is None:
        generator=torch.Generator(device=inputs.device).manual_seed(seed)
        noise={n:(2*torch.randint(0,2,(rank,)+m.shape,device=m.device,generator=generator)-1).to(m.dtype)
               for n,(m,v) in sorted(beliefs.items())}
    states={'input':SharedMoments(inputs,inputs.new_zeros((rank,)+inputs.shape))};records=[]
    for index,node in enumerate(graph.nodes):
        layer=graph.layers[node.name]
        names=tuple(node.name+'.'+n for n,p in layer.named_children() if isinstance(p,GaussianParameter))
        parents=tuple((states[n].mean,states[n].factors) for n in node.parents)
        qs=tuple(beliefs[n] for n in names);es=tuple(noise[n] for n in names)
        readout=index==len(graph.nodes)-1
        if readout and not isinstance(layer,MomentLinear):
            raise ValueError('the full-covariance terminal boundary must be an affine readout')
        local=lambda ps,ws,layer=layer,es=es,readout=readout:shared_layer(layer,ps,ws,es,readout=readout)
        out=local(parents,qs)
        states[node.name]=RootMoments(*out) if readout else SharedMoments(*out)
        records.append((node,names,parents,qs,local))
    return SharedPass(states[graph.nodes[-1].name],states,tuple(records),beliefs)


def root_nodes(z):
    """Integrate K-1 logit contrasts with matching Gaussian covariance.

    Softmax ignores the shared additive logit. QR of the transposed contrast
    root yields a square covariance root without first forming a nearly
    singular dense covariance. The +/- axes preserve its first two moments.
    """
    # Match R/JAX and the original image experiments: the last class is the
    # reference. Finite cubature is not invariant to arbitrary rotations.
    mean=z.mean[:,:-1]-z.mean[:,-1:]
    root=z.root[:,:-1,:]-z.root[:,-1:,:]
    triangular=torch.linalg.qr(root.mT,mode='reduced').R.mT
    d=mean.shape[-1];eye=torch.eye(d,dtype=mean.dtype,device=mean.device)*math.sqrt(d)
    delta=torch.einsum('bij,rj->rbi',triangular,torch.cat((eye,-eye)))
    contrasts=mean[None]+delta
    return torch.cat((contrasts,torch.zeros_like(contrasts[...,:1])),-1)


def root_objective(z,labels,method='mfvi',power=1.,scale=1.):
    nodes=root_nodes(z);logp=nodes.log_softmax(-1)
    selected=logp.gather(-1,labels[None,:,None].expand(len(nodes),-1,1)).squeeze(-1)
    if method=='mfvi':return scale*selected.mean(0).sum()
    if method=='ep' and power>0:
        return scale*(torch.logsumexp(power*selected,0)-math.log(len(nodes))).sum()/power
    raise ValueError('use mfvi or ep with positive power')


def shared_likelihood(graph,inputs,labels,rank=4,seed=41,beliefs=None,
                      method='mfvi',power=1.,scale=1.,noise=None):
    passed=shared_forward(graph,inputs,rank,seed,beliefs,noise)
    head=lambda m,r:root_objective(RootMoments(m,r),labels,method,power,scale)
    child,value=torch.func.grad_and_value(head,argnums=(0,1))(*passed.output)
    weights,activations,signals=passed.backward(child)
    return value,weights,activations,signals,passed
