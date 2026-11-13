"""Native ``nn.Module`` moment layers and two independent backward paths."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .geometry import (
    DiagonalMoments,
    EuclideanAdjoint,
    MomentSignal,
    decode_diagonal,
    encode_diagonal,
)
from .correlation import RootMoments


@dataclass(frozen=True)
class LayerMessages:
    parents: tuple[MomentSignal, ...]
    parameters: dict[str, MomentSignal]


@dataclass(frozen=True)
class LocalRecord:
    output: DiagonalMoments | RootMoments
    pullback: Callable[[MomentSignal | tuple[Tensor, Tensor]], LayerMessages]


def _projected_local(
    forward: Callable[..., DiagonalMoments],
    parents: Sequence[DiagonalMoments],
) -> tuple[DiagonalMoments, Callable[[MomentSignal], tuple[MomentSignal, ...]]]:
    """Decode child signal -> local VJP -> encode every parent signal."""
    flat = tuple(value for q in parents for value in q)

    def flat_forward(*values: Tensor) -> tuple[Tensor, Tensor]:
        states = tuple(DiagonalMoments(*values[i : i + 2]) for i in range(0, len(values), 2))
        out = forward(*states)
        return out.mean, out.variance

    (mean, variance), vjp = torch.func.vjp(flat_forward, *flat)
    output = DiagonalMoments(mean, variance)

    def pullback(child: MomentSignal) -> tuple[MomentSignal, ...]:
        g = decode_diagonal(output, child)
        derivatives = vjp((g.mean, g.variance))
        return tuple(
            encode_diagonal(q, EuclideanAdjoint(derivatives[i], derivatives[i + 1]))
            for q, i in zip(parents, range(0, len(derivatives), 2))
        )

    return output, pullback


class GaussianParameter(nn.Module):
    """A diagonal Gaussian parameter with a serializable prior."""

    def __init__(self, shape: Sequence[int], prior_variance: float, initial_scale: float = 0.01):
        super().__init__()
        self.mean = nn.Parameter(torch.empty(tuple(shape)))
        self.log_variance = nn.Parameter(
            torch.full(tuple(shape), math.log(prior_variance * initial_scale))
        )
        self.register_buffer("prior_mean", torch.zeros(tuple(shape)))
        self.register_buffer("prior_variance", torch.full(tuple(shape), prior_variance))

    @property
    def variance(self) -> Tensor:
        return self.log_variance.exp()

    def moments(self) -> DiagonalMoments:
        return DiagonalMoments(self.mean, self.variance)

    def prior(self) -> DiagonalMoments:
        return DiagonalMoments(self.prior_mean, self.prior_variance)

    def install(self, belief: DiagonalMoments) -> None:
        """Atomically copy one validated Gaussian belief into this module."""
        if torch.any(belief.variance <= 0):
            raise ValueError("installed parameter variance must be positive")
        with torch.no_grad():
            self.mean.copy_(belief.mean)
            self.log_variance.copy_(belief.variance.log())


def dense_moments(
    x: DiagonalMoments, weight: DiagonalMoments, bias: DiagonalMoments
) -> DiagonalMoments:
    """Independent affine moments using PyTorch's ``[out, in]`` layout.

    For ``Y = X A^T + b``, independence of ``X`` and ``A`` gives
    ``E[Y] = m A^T + m_b`` and
    ``Var[Y] = v (A^2)^T + (m^2 + v) S^T + v_b``, where ``(A,S)`` are the
    weight mean and variance.  Only two moments are used; ``X`` need not be
    Gaussian.
    """
    mean = F.linear(x.mean, weight.mean, bias.mean)
    variance = (
        F.linear(x.variance, weight.mean.square())
        + F.linear(x.mean.square() + x.variance, weight.variance)
        + bias.variance
    )
    return DiagonalMoments(mean, variance)


def dense_literal_backward(
    x: DiagonalMoments,
    weight: DiagonalMoments,
    bias: DiagonalMoments,
    child: MomentSignal,
) -> tuple[MomentSignal, MomentSignal, MomentSignal]:
    # Decode the child's Fisher signal to ordinary adjoints (a,b).  Directly
    # differentiating the displayed forward equations gives
    #   g_m = a A + 2 m (b S),       g_v = b (A^2 + S),
    #   g_A = a^T m + 2 A (b^T v),  g_S = b^T (m^2 + v).
    # Encoding each pair at its own belief returns the parent moment messages.
    y = dense_moments(x, weight, bias)
    a, b = decode_diagonal(y, child)
    gx = EuclideanAdjoint(
        a @ weight.mean + 2 * x.mean * (b @ weight.variance),
        b @ (weight.mean.square() + weight.variance),
    )
    gw = EuclideanAdjoint(
        a.mT @ x.mean + 2 * weight.mean * (b.mT @ x.variance),
        b.mT @ (x.mean.square() + x.variance),
    )
    gb = EuclideanAdjoint(a.sum(0), b.sum(0))
    return (
        encode_diagonal(x, gx),
        encode_diagonal(weight, gw),
        encode_diagonal(bias, gb),
    )


def conv2d_moments(
    x: DiagonalMoments,
    weight: DiagonalMoments,
    bias: DiagonalMoments,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    groups: int = 1,
) -> DiagonalMoments:
    """Independent Conv2d moments; spatial covariance is projected out.

    Write ``C(x,A)`` for convolution.  The Dense identities become
    ``M=C(m,A)+m_b`` and ``V=C(v,A^2)+C(m^2+v,S)+v_b``.  A shared filter
    induces spatial/channel covariance, but this diagonal representation keeps
    only its marginal variances.
    """
    convolve = lambda value, kernel, additive=None: F.conv2d(
        value, kernel, additive, stride, padding, dilation, groups
    )
    mean = convolve(x.mean, weight.mean, bias.mean)
    variance = (
        convolve(x.variance, weight.mean.square())
        + convolve(x.mean.square() + x.variance, weight.variance)
        + bias.variance[None, :, None, None]
    )
    return DiagonalMoments(mean, variance)


def conv2d_literal_backward(
    x: DiagonalMoments,
    weight: DiagonalMoments,
    bias: DiagonalMoments,
    child: MomentSignal,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    groups: int = 1,
) -> tuple[MomentSignal, MomentSignal, MomentSignal]:
    """Handwritten transpose-convolution messages for ``conv2d_moments``."""
    y = conv2d_moments(x, weight, bias, stride, padding, dilation, groups)
    a, b = decode_diagonal(y, child)

    # Let I(K,g) and K(x,g) be the input and filter adjoints of C(x,K).
    # The chain rule gives
    #   g_m = I(A,a) + 2m I(S,b),       g_v = I(A^2+S,b),
    #   g_A = K(m,a) + 2A K(v,b),       g_S = K(m^2+v,b).
    # torch.nn.grad supplies these linear transpose operators; it performs no
    # inference and is independent of the projected torch.func VJP.
    input_adjoint = lambda kernel, out: torch.nn.grad.conv2d_input(
        x.mean.shape, kernel, out, stride, padding, dilation, groups
    )
    weight_adjoint = lambda value, out: torch.nn.grad.conv2d_weight(
        value, weight.mean.shape, out, stride, padding, dilation, groups
    )
    gx = EuclideanAdjoint(
        input_adjoint(weight.mean, a) + 2 * x.mean * input_adjoint(weight.variance, b),
        input_adjoint(weight.mean.square() + weight.variance, b),
    )
    gw = EuclideanAdjoint(
        weight_adjoint(x.mean, a) + 2 * weight.mean * weight_adjoint(x.variance, b),
        weight_adjoint(x.mean.square() + x.variance, b),
    )
    gb = EuclideanAdjoint(a.sum((0, 2, 3)), b.sum((0, 2, 3)))
    return (
        encode_diagonal(x, gx),
        encode_diagonal(weight, gw),
        encode_diagonal(bias, gb),
    )


def relu_terms(x: DiagonalMoments) -> tuple[DiagonalMoments, tuple[Tensor, ...]]:
    positive = x.variance > 0
    scale = torch.sqrt(torch.where(positive, x.variance, torch.ones_like(x.variance)))
    alpha = x.mean / scale
    probability = 0.5 * torch.erfc(-alpha / math.sqrt(2))
    density = torch.exp(-alpha.square() / 2) / math.sqrt(2 * math.pi)
    mean_positive = x.mean * probability + scale * density
    second = (x.mean.square() + x.variance) * probability + x.mean * scale * density
    raw_variance = second - mean_positive.square()
    active = raw_variance > 0
    mean = torch.where(positive, mean_positive, F.relu(x.mean))
    variance = torch.where(positive, raw_variance.clamp_min(0), torch.zeros_like(x.variance))
    gate = (x.mean > 0).to(x.mean.dtype)
    return DiagonalMoments(mean, variance), (
        positive,
        active,
        scale,
        probability,
        density,
        mean_positive,
        gate,
    )


def relu_moments(x: DiagonalMoments) -> DiagonalMoments:
    return relu_terms(x)[0]


def relu_literal_backward(x: DiagonalMoments, child: MomentSignal) -> tuple[MomentSignal]:
    y, terms = relu_terms(x)
    positive, active, scale, probability, density, mean_positive, gate = terms
    a, b = decode_diagonal(y, child)
    dmean_dm = torch.where(positive, probability, gate)
    dmean_dv = torch.where(positive, density / (2 * scale), torch.zeros_like(scale))
    dvar_dm = torch.where(
        positive & active, 2 * mean_positive * (1 - probability), torch.zeros_like(scale)
    )
    dvar_dv = torch.where(
        positive & active,
        probability - mean_positive * density / scale,
        torch.zeros_like(scale),
    )
    return (
        encode_diagonal(
            x,
            EuclideanAdjoint(
                a * dmean_dm + b * dvar_dm,
                a * dmean_dv + b * dvar_dv,
            ),
        ),
    )


def average_pool_moments(x: DiagonalMoments, kernel_size: int) -> DiagonalMoments:
    area = kernel_size * kernel_size
    return DiagonalMoments(
        F.avg_pool2d(x.mean, kernel_size, stride=kernel_size),
        F.avg_pool2d(x.variance, kernel_size, stride=kernel_size) / area,
    )


def _average_pool_adjoint(g: Tensor, input_shape: torch.Size, kernel_size: int) -> Tensor:
    area = kernel_size * kernel_size
    up = g.repeat_interleave(kernel_size, -2).repeat_interleave(kernel_size, -1) / area
    pad_h = input_shape[-2] - up.shape[-2]
    pad_w = input_shape[-1] - up.shape[-1]
    return F.pad(up, (0, pad_w, 0, pad_h))


def average_pool_literal_backward(
    x: DiagonalMoments, child: MomentSignal, kernel_size: int
) -> tuple[MomentSignal]:
    y = average_pool_moments(x, kernel_size)
    a, b = decode_diagonal(y, child)
    mean_adjoint = _average_pool_adjoint(a, x.mean.shape, kernel_size)
    variance_adjoint = _average_pool_adjoint(b, x.mean.shape, kernel_size) / (
        kernel_size * kernel_size
    )
    return (encode_diagonal(x, EuclideanAdjoint(mean_adjoint, variance_adjoint)),)


def flatten_moments(x: DiagonalMoments) -> DiagonalMoments:
    return DiagonalMoments(x.mean.flatten(1), x.variance.flatten(1))


def flatten_literal_backward(x: DiagonalMoments, child: MomentSignal) -> tuple[MomentSignal]:
    y = flatten_moments(x)
    a, b = decode_diagonal(y, child)
    return (encode_diagonal(x, EuclideanAdjoint(a.reshape_as(x.mean), b.reshape_as(x.variance))),)


def add_moments(
    x: DiagonalMoments, y: DiagonalMoments | None = None, *, same_parent: bool = False
) -> DiagonalMoments:
    if same_parent:
        return DiagonalMoments(2 * x.mean, 4 * x.variance)
    if y is None:
        raise ValueError("a distinct second parent is required")
    return DiagonalMoments(x.mean + y.mean, x.variance + y.variance)


def add_literal_backward(
    x: DiagonalMoments,
    child: MomentSignal,
    y: DiagonalMoments | None = None,
    *,
    same_parent: bool = False,
) -> tuple[MomentSignal, ...]:
    out = add_moments(x, y, same_parent=same_parent)
    a, b = decode_diagonal(out, child)
    if same_parent:
        return (encode_diagonal(x, EuclideanAdjoint(2 * a, 4 * b)),)
    assert y is not None
    return (
        encode_diagonal(x, EuclideanAdjoint(a, b)),
        encode_diagonal(y, EuclideanAdjoint(a, b)),
    )


def concat_moments(x: DiagonalMoments, y: DiagonalMoments, dim: int) -> DiagonalMoments:
    return DiagonalMoments(torch.cat((x.mean, y.mean), dim), torch.cat((x.variance, y.variance), dim))


def concat_literal_backward(
    x: DiagonalMoments, y: DiagonalMoments, child: MomentSignal, dim: int
) -> tuple[MomentSignal, MomentSignal]:
    out = concat_moments(x, y, dim)
    a, b = decode_diagonal(out, child)
    am = torch.split(a, (x.mean.shape[dim], y.mean.shape[dim]), dim)
    bv = torch.split(b, (x.variance.shape[dim], y.variance.shape[dim]), dim)
    return (
        encode_diagonal(x, EuclideanAdjoint(am[0], bv[0])),
        encode_diagonal(y, EuclideanAdjoint(am[1], bv[1])),
    )


def dense_full_moments(x, w, b):
    """C_z = W diag(v_x) Wᵀ + diag((m_x²+v_x) V_Wᵀ + v_b).

    Keep a rectangular covariance root; prediction needs no derivative tape.
    At a deterministic input, the zero standard deviation has zero tangent.
    """
    positive = x.variance > 0
    sd = torch.where(positive, torch.where(positive, x.variance,
        torch.ones_like(x.variance)).sqrt(), torch.zeros_like(x.variance))
    root = w.mean[None] * sd[:, None, :]
    diagonal = F.linear(x.mean.square() + x.variance, w.variance, b.variance)
    return RootMoments(F.linear(x.mean, w.mean, b.mean),
                       torch.cat((root, torch.diag_embed(diagonal.sqrt())), -1))


class MomentLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, prior_variance: float = 1.0):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.weight = GaussianParameter((out_features, in_features), prior_variance)
        self.bias = GaussianParameter((out_features,), prior_variance)
        bound = 1 / math.sqrt(in_features)
        nn.init.uniform_(self.weight.mean, -bound, bound)
        nn.init.uniform_(self.bias.mean, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.mean, self.bias.mean)

    def forward_moments(self, x: DiagonalMoments) -> DiagonalMoments:
        return dense_moments(x, self.weight.moments(), self.bias.moments())

    def forward_full_moments(self, x: DiagonalMoments) -> RootMoments:
        return dense_full_moments(x, self.weight.moments(), self.bias.moments())

    def project_full(self, x: DiagonalMoments) -> LocalRecord:
        """Keep the small logit covariance even with diagonal hidden moments.

        C_z = W diag(v_x) W^T + diag((m_x²+v_x) V_W^T + v_b).
        A rectangular square root retains small positive modes. Its incoming
        adjoint is Euclidean; local parent/weight adjoints are encoded as the
        same Gaussian signals consumed by the ordinary graph traversal.
        """
        from .correlation import RootMoments
        parents=(x,self.weight.moments(),self.bias.moments())
        def forward(q):return tuple(dense_full_moments(*q))
        output,pull=torch.func.vjp(forward,parents)
        def backward(child):
            gradients=pull(child)[0]
            signals=tuple(encode_diagonal(q,EuclideanAdjoint(*g)) for q,g in zip(parents,gradients))
            return LayerMessages((signals[0],),{'weight':signals[1],'bias':signals[2]})
        return LocalRecord(RootMoments(*output),backward)

    def project(self, x: DiagonalMoments) -> LocalRecord:
        parents = (x, self.weight.moments(), self.bias.moments())
        output, backward = _projected_local(dense_moments, parents)

        def pullback(child: MomentSignal) -> LayerMessages:
            messages = backward(child)
            return LayerMessages((messages[0],), {"weight": messages[1], "bias": messages[2]})

        return LocalRecord(output, pullback)

    def literal(self, x: DiagonalMoments) -> LocalRecord:
        weight, bias = self.weight.moments(), self.bias.moments()
        output = dense_moments(x, weight, bias)

        def pullback(child: MomentSignal) -> LayerMessages:
            messages = dense_literal_backward(x, weight, bias, child)
            return LayerMessages((messages[0],), {"weight": messages[1], "bias": messages[2]})

        return LocalRecord(output, pullback)


class MomentConv2d(nn.Module):
    """``nn.Conv2d`` with a diagonal Gaussian belief over kernel and bias."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        prior_variance: float = 1.0,
    ):
        super().__init__()
        if in_channels % groups or out_channels % groups:
            raise ValueError("in_channels and out_channels must be divisible by groups")
        kernel = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size, self.stride, self.padding = kernel, stride, padding
        self.dilation, self.groups = dilation, groups
        self.weight = GaussianParameter(
            (out_channels, in_channels // groups, *kernel), prior_variance
        )
        self.bias = GaussianParameter((out_channels,), prior_variance) if bias else None
        fan_in = (in_channels // groups) * kernel[0] * kernel[1]
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.weight.mean, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias.mean, -bound, bound)

    def _bias_moments(self) -> DiagonalMoments:
        if self.bias is not None:
            return self.bias.moments()
        zero = self.weight.mean.new_zeros(self.out_channels)
        return DiagonalMoments(zero, zero)

    def _moments(self, x: DiagonalMoments) -> DiagonalMoments:
        return conv2d_moments(
            x,
            self.weight.moments(),
            self._bias_moments(),
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def forward(self, x: Tensor) -> Tensor:
        return F.conv2d(
            x,
            self.weight.mean,
            None if self.bias is None else self.bias.mean,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def forward_moments(self, x: DiagonalMoments) -> DiagonalMoments:
        return self._moments(x)

    def project(self, x: DiagonalMoments) -> LocalRecord:
        parents = (x, self.weight.moments())
        if self.bias is not None:
            parents += (self.bias.moments(),)
            forward = lambda a, w, b: conv2d_moments(
                a, w, b, self.stride, self.padding, self.dilation, self.groups
            )
        else:
            forward = lambda a, w: conv2d_moments(
                a, w, self._bias_moments(), self.stride, self.padding, self.dilation, self.groups
            )
        output, backward = _projected_local(forward, parents)

        def pullback(child: MomentSignal) -> LayerMessages:
            messages = backward(child)
            parameters = {"weight": messages[1]}
            if self.bias is not None:
                parameters["bias"] = messages[2]
            return LayerMessages((messages[0],), parameters)

        return LocalRecord(output, pullback)

    def literal(self, x: DiagonalMoments) -> LocalRecord:
        weight, bias = self.weight.moments(), self._bias_moments()
        output = self._moments(x)

        def pullback(child: MomentSignal) -> LayerMessages:
            messages = conv2d_literal_backward(
                x,
                weight,
                bias,
                child,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
            parameters = {"weight": messages[1]}
            if self.bias is not None:
                parameters["bias"] = messages[2]
            return LayerMessages((messages[0],), parameters)

        return LocalRecord(output, pullback)


class MomentReLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.relu(x)

    def forward_moments(self, x: DiagonalMoments) -> DiagonalMoments:
        return relu_moments(x)

    def project(self, x: DiagonalMoments) -> LocalRecord:
        output, backward = _projected_local(relu_moments, (x,))
        return LocalRecord(output, lambda child: LayerMessages(backward(child), {}))

    def literal(self, x: DiagonalMoments) -> LocalRecord:
        return LocalRecord(relu_moments(x), lambda child: LayerMessages(relu_literal_backward(x, child), {}))


def gaussian_max_pair(a: DiagonalMoments, b: DiagonalMoments) -> DiagonalMoments:
    """Exact two moments of max(A,B) for independent scalar Gaussians.

    s²=v_a+v_b, t=(m_a-m_b)/s, P=Phi(t), p=phi(t):
      mean = m_a P + m_b (1-P) + s p
      second = (v_a+m_a²)P + (v_b+m_b²)(1-P) + (m_a+m_b)s p.
    Pairwise repetition is a Gaussian *projection*, not an exact four-way max.
    At zero variance use the deterministic maximum, without dividing by zero.
    """
    positive = a.variance + b.variance > 0
    s = torch.where(positive, a.variance + b.variance, torch.ones_like(a.mean)).sqrt()
    t = (a.mean-b.mean)/s
    P = .5*torch.erfc(-t/math.sqrt(2))
    p = torch.exp(-.5*t.square())/math.sqrt(2*math.pi)
    m = a.mean*P+b.mean*(1-P)+s*p
    second = (a.variance+a.mean.square())*P+(b.variance+b.mean.square())*(1-P)+(a.mean+b.mean)*s*p
    take_a = a.mean >= b.mean
    return DiagonalMoments(torch.where(positive, m, torch.maximum(a.mean,b.mean)),
        torch.where(positive, (second-m.square()).clamp_min(0), torch.where(take_a,a.variance,b.variance)))


def _max_pair_adjoint(a, b, adjoint):
    """Hand derivatives of the pair's mean/raw second, then chain v=s2-m²."""
    positive = a.variance+b.variance > 0
    s = torch.where(positive,a.variance+b.variance,torch.ones_like(a.mean)).sqrt()
    t = (a.mean-b.mean)/s
    P = .5*torch.erfc(-t/math.sqrt(2))
    p = torch.exp(-.5*t.square())/math.sqrt(2*math.pi)
    m = a.mean*P+b.mean*(1-P)+s*p
    second = (a.variance+a.mean.square())*P+(b.variance+b.mean.square())*(1-P)+(a.mean+b.mean)*s*p
    gv = adjoint.variance*(second-m.square() >= 0)
    gm = adjoint.mean-2*m*gv
    common = p*((a.mean+b.mean)/(2*s)-(a.mean-b.mean)*(a.variance-b.variance)/(2*s**3))
    ga = EuclideanAdjoint(gm*P+gv*(2*a.mean*P+2*a.variance*p/s),
                         gm*p/(2*s)+gv*(P+common))
    gb = EuclideanAdjoint(gm*(1-P)+gv*(2*b.mean*(1-P)+2*b.variance*p/s),
                         gm*p/(2*s)+gv*(1-P+common))
    take = a.mean >= b.mean
    return tuple(EuclideanAdjoint(torch.where(positive,g.mean,adjoint.mean*mean_mask),
                                 torch.where(positive,g.variance,adjoint.variance*mask))
                 for g,mask,mean_mask in ((ga,take,(a.mean>b.mean)+.5*(a.mean==b.mean)),
                                         (gb,~take,(b.mean>a.mean)+.5*(a.mean==b.mean))))


def max_pool_moments(x: DiagonalMoments) -> DiagonalMoments:
    """2x2 stride-2 max pool, with two declared pairwise Gaussian projections."""
    states = [DiagonalMoments(x.mean[...,i::2,j::2],x.variance[...,i::2,j::2])
              for i,j in ((0,0),(0,1),(1,0),(1,1))]
    return gaussian_max_pair(gaussian_max_pair(*states[:2]), gaussian_max_pair(*states[2:]))


def max_pool_literal_backward(x, child):
    states = [DiagonalMoments(x.mean[...,i::2,j::2],x.variance[...,i::2,j::2])
              for i,j in ((0,0),(0,1),(1,0),(1,1))]
    left, right = gaussian_max_pair(*states[:2]),gaussian_max_pair(*states[2:])
    root = gaussian_max_pair(left,right)
    gl, gr = _max_pair_adjoint(left,right,decode_diagonal(root,child))
    derivatives = _max_pair_adjoint(*states[:2],gl)+_max_pair_adjoint(*states[2:],gr)
    gm,gv = torch.zeros_like(x.mean),torch.zeros_like(x.variance)
    for g,(i,j) in zip(derivatives,((0,0),(0,1),(1,0),(1,1))):
        gm[...,i::2,j::2],gv[...,i::2,j::2] = g.mean,g.variance
    return (encode_diagonal(x,EuclideanAdjoint(gm,gv)),)


class MomentMaxPool2d(nn.Module):
    """Ordinary max pooling and a Gaussian-pair moment approximation."""
    def forward(self,x): return F.max_pool2d(x,2)
    def forward_moments(self,x):
        if x.mean.shape[-1]%2 or x.mean.shape[-2]%2:
            raise ValueError("the pairwise teaching pool requires even spatial dimensions")
        return max_pool_moments(x)
    def project(self,x):
        out,backward = _projected_local(self.forward_moments,(x,))
        return LocalRecord(out,lambda child: LayerMessages(backward(child),{}))
    def literal(self,x):
        return LocalRecord(self.forward_moments(x),lambda child: LayerMessages(max_pool_literal_backward(x,child),{}))


class MomentAvgPool2d(nn.Module):
    def __init__(self, kernel_size: int = 2):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x: Tensor) -> Tensor:
        return F.avg_pool2d(x, self.kernel_size, stride=self.kernel_size)

    def forward_moments(self, x: DiagonalMoments) -> DiagonalMoments:
        return average_pool_moments(x, self.kernel_size)

    def project(self, x: DiagonalMoments) -> LocalRecord:
        forward = lambda state: average_pool_moments(state, self.kernel_size)
        output, backward = _projected_local(forward, (x,))
        return LocalRecord(output, lambda child: LayerMessages(backward(child), {}))

    def literal(self, x: DiagonalMoments) -> LocalRecord:
        output = average_pool_moments(x, self.kernel_size)
        return LocalRecord(
            output,
            lambda child: LayerMessages(average_pool_literal_backward(x, child, self.kernel_size), {}),
        )


class MomentFlatten(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.flatten(1)

    def forward_moments(self, x: DiagonalMoments) -> DiagonalMoments:
        return flatten_moments(x)

    def project(self, x: DiagonalMoments) -> LocalRecord:
        output, backward = _projected_local(flatten_moments, (x,))
        return LocalRecord(output, lambda child: LayerMessages(backward(child), {}))

    def literal(self, x: DiagonalMoments) -> LocalRecord:
        return LocalRecord(flatten_moments(x), lambda child: LayerMessages(flatten_literal_backward(x, child), {}))


class MomentAdd(nn.Module):
    def __init__(self, same_parent: bool = False):
        super().__init__()
        self.same_parent = same_parent

    def forward(self, x: Tensor, y: Tensor | None = None) -> Tensor:
        return 2 * x if self.same_parent else x + y

    def forward_moments(
        self, x: DiagonalMoments, y: DiagonalMoments | None = None
    ) -> DiagonalMoments:
        return add_moments(x, y, same_parent=self.same_parent)

    def project(self, x: DiagonalMoments, y: DiagonalMoments | None = None) -> LocalRecord:
        if self.same_parent:
            forward = lambda state: add_moments(state, same_parent=True)
            output, backward = _projected_local(forward, (x,))
        else:
            assert y is not None
            output, backward = _projected_local(add_moments, (x, y))
        return LocalRecord(output, lambda child: LayerMessages(backward(child), {}))

    def literal(self, x: DiagonalMoments, y: DiagonalMoments | None = None) -> LocalRecord:
        output = add_moments(x, y, same_parent=self.same_parent)
        return LocalRecord(
            output,
            lambda child: LayerMessages(
                add_literal_backward(x, child, y, same_parent=self.same_parent), {}
            ),
        )


class MomentConcat(nn.Module):
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.cat((x, y), self.dim)

    def forward_moments(self, x: DiagonalMoments, y: DiagonalMoments) -> DiagonalMoments:
        return concat_moments(x, y, self.dim)

    def project(self, x: DiagonalMoments, y: DiagonalMoments) -> LocalRecord:
        forward = lambda a, b: concat_moments(a, b, self.dim)
        output, backward = _projected_local(forward, (x, y))
        return LocalRecord(output, lambda child: LayerMessages(backward(child), {}))

    def literal(self, x: DiagonalMoments, y: DiagonalMoments) -> LocalRecord:
        output = concat_moments(x, y, self.dim)
        return LocalRecord(
            output,
            lambda child: LayerMessages(concat_literal_backward(x, y, child, self.dim), {}),
        )


class MomentBatchNorm2d(nn.Module):
    """Batch normalization with Gaussian scale and shift beliefs.

    Normalization statistics come from the activation mean.  Conditional on
    frozen statistics, if
    ``A=(X-mu)/sqrt(sigma2+eps)`` has moments ``(a,t)``, then

    ``E[Y]=a*m_gamma+m_beta`` and
    ``Var[Y]=t*m_gamma^2+(a^2+t)*v_gamma+v_beta``.

    Running-statistic installation is explicit through
    :meth:`commit_statistics`; it never occurs halfway through a reverse pass.
    """

    def __init__(self, channels: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.channels, self.eps, self.momentum = channels, eps, momentum
        self.weight = GaussianParameter((channels,), prior_variance=1.0, initial_scale=1e-5)
        self.bias = GaussianParameter((channels,), prior_variance=1.0, initial_scale=1e-5)
        nn.init.ones_(self.weight.mean)
        nn.init.zeros_(self.bias.mean)
        self.register_buffer("running_mean", torch.zeros(channels))
        self.register_buffer("running_var", torch.ones(channels))

    def _statistics(self, mean: Tensor) -> tuple[Tensor, Tensor]:
        if self.training:
            center = mean.mean((0, 2, 3))
            variance = (mean - center[None, :, None, None]).square().mean((0, 2, 3))
            return center, variance
        return self.running_mean, self.running_var

    def commit_statistics(self, mean: Tensor) -> None:
        """Update running statistics after one complete accepted forward pass."""
        center = mean.mean((0, 2, 3))
        variance = mean.var((0, 2, 3), unbiased=True)
        with torch.no_grad():
            self.running_mean.lerp_(center, self.momentum)
            self.running_var.lerp_(variance, self.momentum)

    def _moments(
        self, x: DiagonalMoments, weight: DiagonalMoments, bias: DiagonalMoments
    ) -> DiagonalMoments:
        center, variance = self._statistics(x.mean)
        scale = torch.rsqrt(variance[None, :, None, None] + self.eps)
        normalized_mean = (x.mean - center[None, :, None, None]) * scale
        normalized_variance = x.variance * scale.square()
        gamma_mean = weight.mean[None, :, None, None]
        gamma_variance = weight.variance[None, :, None, None]
        mean = normalized_mean * gamma_mean + bias.mean[None, :, None, None]
        output_variance = (
            normalized_variance * gamma_mean.square()
            + (normalized_mean.square() + normalized_variance) * gamma_variance
            + bias.variance[None, :, None, None]
        )
        return DiagonalMoments(mean, output_variance)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            # A covariance JVP is differentiated again in the reverse pass.
            # Express statistics in elementary operations so higher-order
            # derivatives retain both the centering and variance dependence.
            center, variance = self._statistics(x)
            normalized = (x-center[None,:,None,None])*torch.rsqrt(
                variance[None,:,None,None]+self.eps)
            return normalized*self.weight.mean[None,:,None,None]+self.bias.mean[None,:,None,None]
        return F.batch_norm(
            x,
            # Sampling, local derivatives and HVPs must not change running
            # state. The trainer commits one averaged observation per batch.
            None if self.training else self.running_mean,
            None if self.training else self.running_var,
            self.weight.mean,
            self.bias.mean,
            self.training,
            self.momentum,
            self.eps,
        )

    def forward_moments(self, x: DiagonalMoments) -> DiagonalMoments:
        return self._moments(x, self.weight.moments(), self.bias.moments())

    def project(self, x: DiagonalMoments) -> LocalRecord:
        parents = (x, self.weight.moments(), self.bias.moments())
        output, backward = _projected_local(self._moments, parents)

        def pullback(child: MomentSignal) -> LayerMessages:
            messages = backward(child)
            return LayerMessages((messages[0],), {"weight": messages[1], "bias": messages[2]})

        return LocalRecord(output, pullback)


class MomentGlobalAvgPool2d(nn.Module):
    """Global average pooling under the retained diagonal covariance."""

    def forward(self, x: Tensor) -> Tensor:
        return x.mean((-2, -1))

    def forward_moments(self, x: DiagonalMoments) -> DiagonalMoments:
        pixels = x.mean.shape[-2] * x.mean.shape[-1]
        return DiagonalMoments(x.mean.mean((-2, -1)), x.variance.sum((-2, -1)) / pixels**2)

    def project(self, x: DiagonalMoments) -> LocalRecord:
        output, backward = _projected_local(self.forward_moments, (x,))
        return LocalRecord(output, lambda child: LayerMessages(backward(child), {}))

    def literal(self, x: DiagonalMoments) -> LocalRecord:
        output = self.forward_moments(x)

        def pullback(child: MomentSignal) -> LayerMessages:
            a, b = decode_diagonal(output, child)
            height, width = x.mean.shape[-2:]
            gm = a[..., None, None].expand_as(x.mean) / (height * width)
            gv = b[..., None, None].expand_as(x.variance) / (height * width) ** 2
            return LayerMessages((encode_diagonal(x, EuclideanAdjoint(gm, gv)),), {})

        return LocalRecord(output, pullback)


class MomentCifarShortcut(nn.Module):
    """ResNet-CIFAR option-A shortcut: stride-two plus symmetric channel zeros."""

    def forward(self, x: Tensor) -> Tensor:
        z = x[:, :, ::2, ::2]
        channels = z.shape[1] // 2
        return F.pad(z, (0, 0, 0, 0, channels, channels))

    def forward_moments(self, x: DiagonalMoments) -> DiagonalMoments:
        return DiagonalMoments(self.forward(x.mean), self.forward(x.variance))

    def project(self, x: DiagonalMoments) -> LocalRecord:
        output, backward = _projected_local(self.forward_moments, (x,))
        return LocalRecord(output, lambda child: LayerMessages(backward(child), {}))


class MomentIdentity(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x

    def forward_moments(self, x: DiagonalMoments) -> DiagonalMoments:
        return x

    def project(self, x: DiagonalMoments) -> LocalRecord:
        return LocalRecord(x, lambda child: LayerMessages((child,), {}))

    literal = project
