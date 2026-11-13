"""Layerwise Gaussian moment approximation for the released BLoB RoBERTa.

State (m,U) represents mean m and covariance UU'/K after flattening activations.
The same column index follows every residual/attention branch. Local independent
noise is sketched with fixed Rademacher columns; finite K is an approximation,
not an exact dense covariance. Seeds/ranks must therefore be audited separately.

Random A and deterministic B give conditional adapter covariance
  scale^2 B diag(sigma_A^2 E[x^2]) B'.
This includes input variance and the induced channel correlations. The source
uses token-local Flipout signs: cross-token injected covariance is zero in
expectation over sketches. We approximate its non-Gaussian joint noise by a
Gaussian with these moments. This is NOT coherent shared-network sampling.

Nonlinear means are recomputed at every layer. GELU/tanh use Gaussian quadrature.
LayerNorm/softmax use symmetric cubature. Odd and even covariance columns retain
nonlinear channel correlations; fixed random signs compress their concatenation
back to K columns. The compression is unbiased BEFORE optional scalar variance
matching, but a finite sketch has cross-term error. There is no posterior
pilot, fitted output calibration, reference gradient or full-network draw here.
"""
from dataclasses import dataclass
import math
import numpy as np
import torch
import torch.nn.functional as F


def root(x):
    # Zero variance is deterministic; avoid sqrt'(0)=Inf contaminating VJPs.
    positive = x > 0
    return torch.where(positive, torch.sqrt(torch.where(positive,x,torch.ones_like(x))), torch.zeros_like(x))


@dataclass
class State:
    m: torch.Tensor
    U: torch.Tensor  # [K, batch, ...]

    @property
    def v(self):
        return self.U.square().mean(0)


class Moments:
    def __init__(self, rank=32, seed=197, nonlinear=True, order=20, covariance_rule='paired'):
        self.rank, self.seed, self.nonlinear = rank, seed, nonlinear
        self.covariance_rule = covariance_rule
        self.counter = 0
        self.nodes, self.weights = np.polynomial.hermite.hermgauss(order)
        self.weights /= np.sqrt(np.pi)

    def noise(self, x):
        # Fixed covariance probes, NOT stochastic network prediction draws.
        key = (self.counter, tuple(x.shape), str(x.device), x.dtype)
        self.counter += 1
        gen = torch.Generator(device=x.device)
        gen.manual_seed(self.seed + 104729 * key[0])
        e = torch.randint(0, 2, (self.rank,) + tuple(x.shape),
                          generator=gen, device=x.device,dtype=torch.int8)
        return (2 * e - 1).to(x.dtype)

    def independent_residual(self, mean, columns, variance):
        # Cov(g(X),X) is retained in columns. Residual is nonnegative for exact
        # Gaussian expectations; clamp only quadrature/roundoff discrepancies.
        residual = (variance - columns.square().mean(0)).clamp_min(0)
        return State(mean, columns + root(residual)[None] * self.noise(mean))

    def pair_columns(self, plus, minus):
        mean=(plus+minus).mean(0)/2
        odd=(plus-minus)/2
        even=(plus+minus)/2-mean[None]
        # Cubature covariance = (OO'+EE')/K. Random column signs compress
        # independent higher-order residual factors: E_s[(O+Es)(O+Es)']=OO'+EE'.
        # A sign is shared across all coordinates, preserving constraints such
        # as zero row sum for softmax perturbations, unlike diagonal residuals.
        signs=self.noise(mean.new_zeros(())).reshape((self.rank,)+(1,)*mean.ndim)
        return State(mean,odd+even*signs)

    def linear(self, layer, x):
        base = layer.base_layer if hasattr(layer, 'base_layer') else layer
        m = F.linear(x.m, base.weight, base.bias)
        U = F.linear(x.U, base.weight)
        if hasattr(layer, 'lora_A'):
            for name in layer.active_adapters:
                A, B = layer.lora_A[name].weight, layer.lora_B[name].weight
                scale = layer.scaling[name]
                m = m + F.linear(F.linear(x.m, A), B) * scale
                U = U + F.linear(F.linear(x.U, A), B) * scale
                if hasattr(layer, 'lora_A_rho'):
                    rho = layer.lora_A_rho[name]
                    sigma2 = rho.pow(4) if layer.bayes_eps >= 0 else F.softplus(rho).square()
                    rank_variance = F.linear(x.m.square() + (x.v if self.nonlinear else 0), sigma2)
                    e = self.noise(rank_variance)
                    U = U + F.linear(e * root(rank_variance)[None], B) * scale
        return State(m, U)

    @staticmethod
    def add(x, y):
        # This retains both residual cross-covariance terms automatically.
        return State(x.m + y.m, x.U + y.U)

    def elementwise(self, x, fn, derivative):
        if not self.nonlinear:
            return State(fn(x.m), derivative(x.m)[None] * x.U)
        # Gaussian regression: a=E[g'(X)], E[g]=sum_j w_j g(m+sqrt(2v)t_j).
        # Cov(g(X),X)=v E[g'(X)] follows from Gaussian integration by parts.
        sd = root(2 * x.v)
        shape=(len(self.nodes),)+(1,)*x.m.ndim
        nodes=torch.as_tensor(self.nodes,device=x.m.device,dtype=x.m.dtype).reshape(shape)
        weights=torch.as_tensor(self.weights,device=x.m.device,dtype=x.m.dtype).reshape(shape)
        z=x.m[None]+nodes*sd[None]
        y=fn(z)
        mean=(weights*y).sum(0)
        # Center before squaring: E[g^2]-E[g]^2 loses all significant digits
        # near the deterministic initialization and can destabilize derivatives.
        variance=(weights*(y-mean[None]).square()).sum(0)
        deterministic = x.v == 0
        mean = torch.where(deterministic,fn(x.m),mean)
        variance = torch.where(deterministic,torch.zeros_like(x.m),variance)
        if self.covariance_rule=='regression':
            gain=(weights*derivative(z)).sum(0)
            return self.independent_residual(mean, gain[None] * x.U, variance)
        paired=self.pair_columns(fn(x.m[None]+x.U),fn(x.m[None]-x.U))
        # Match scalar Gaussian variances while retaining a PSD correlated
        # covariance sketch. This rescaling is an additional approximation to
        # cross-covariances, not a theorem of exact joint Gaussian propagation.
        raw=paired.v
        # Below floating-point resolution, use the analytic zero-variance
        # limit, not a quotient of two underflowed empirical variances.
        threshold=(16*torch.finfo(x.m.dtype).eps*(1+x.m.abs())).square()
        active=(x.v>threshold)&(raw>torch.finfo(raw.dtype).tiny)
        ratio=torch.where(active,variance,torch.ones_like(variance))/torch.where(active,raw,torch.ones_like(raw))
        scale=root(ratio)
        columns=torch.where(active[None],paired.U*scale[None],derivative(x.m)[None]*x.U)
        return State(torch.where(active,mean,fn(x.m)),columns)

    def gelu(self, x):
        phi = lambda z: torch.exp(-z.square() / 2) / math.sqrt(2 * math.pi)
        return self.elementwise(x, lambda z: z * torch.special.ndtr(z),
                                lambda z: torch.special.ndtr(z) + z * phi(z))

    def tanh(self, x):
        return self.elementwise(x, torch.tanh, lambda z: 1 - z.tanh().square())

    def cubature(self, x, fn, linear):
        if not self.nonlinear:
            return State(fn(x.m), linear(x.m, x.U))
        # Spherical-radial rule: covariance factors are U/sqrt(K), so its
        # 2K nodes are m +/- U_j, each with weight 1/(2K).
        plus, minus = fn(x.m[None] + x.U), fn(x.m[None] - x.U)
        if self.covariance_rule=='paired':
            return self.pair_columns(plus,minus)
        mean = (plus + minus).mean(0) / 2
        odd = (plus - minus) / 2
        variance = ((plus - mean[None]).square() +
                    (minus - mean[None]).square()).mean(0) / 2
        return self.independent_residual(mean, odd, variance)

    def norm(self, layer, x):
        fn = lambda z: F.layer_norm(z, layer.normalized_shape, layer.weight,
                                    layer.bias, layer.eps)
        def jac(m, U):
            c, du = m - m.mean(-1, keepdim=True), U - U.mean(-1, keepdim=True)
            s2 = c.square().mean(-1, keepdim=True) + layer.eps
            return (du - c[None] * (du * c[None]).mean(-1, keepdim=True) / s2[None]) * layer.weight / s2.sqrt()[None]
        return self.cubature(x, fn, jac)

    def softmax(self, x, mask):
        fn = lambda z: (z + mask).softmax(-1)
        def jac(m, U):
            p = fn(m)
            return p[None] * (U - (U * p[None]).sum(-1, keepdim=True))
        return self.cubature(x, fn, jac)

    def attention(self, layer, x, mask):
        q, k, v = [self.linear(getattr(layer, n), x) for n in ['query','key','value']]
        B, T, D = q.m.shape
        H, d = layer.num_attention_heads, layer.attention_head_size
        def heads(z):
            return State(z.m.reshape(B,T,H,d).transpose(1,2),
                         z.U.reshape(self.rank,B,T,H,d).transpose(2,3))
        q,k,v = map(heads, (q,k,v))
        # E[Q K'] includes their shared input covariance. First-order columns
        # omit the quadratic residual covariance; this remains an approximation.
        scores_m = q.m @ k.m.transpose(-1,-2)
        if self.nonlinear:
            scores_m = scores_m + (q.U @ k.U.transpose(-1,-2)).mean(0)
        scores_U = q.U @ k.m.transpose(-1,-2)[None] + q.m[None] @ k.U.transpose(-1,-2)
        p = self.softmax(State(scores_m/math.sqrt(d), scores_U/math.sqrt(d)), mask)
        m = p.m @ v.m
        if self.nonlinear:
            m = m + (p.U @ v.U).mean(0)
        U = p.U @ v.m[None] + p.m[None] @ v.U
        return State(m.transpose(1,2).reshape(B,T,D),
                     U.transpose(2,3).reshape(self.rank,B,T,D))

    def contrast(self, layer, x):
        # Exact readout mean/variance for the supplied Gaussian hidden moments
        # and independent A. Avoid a sketch in the scalar terminal contraction.
        base = layer.base_layer if hasattr(layer, 'base_layer') else layer
        w = base.weight[1] - base.weight[0]
        bias = base.bias[1] - base.bias[0] if base.bias is not None else 0.
        local = torch.zeros_like(x.m[...,0])
        if hasattr(layer, 'lora_A'):
            for name in layer.active_adapters:
                A,B,s = layer.lora_A[name].weight,layer.lora_B[name].weight,layer.scaling[name]
                b = (B[1]-B[0])*s
                w = w + b @ A
                if hasattr(layer, 'lora_A_rho'):
                    rho = layer.lora_A_rho[name]
                    sigma2 = rho.pow(4) if layer.bayes_eps >= 0 else F.softplus(rho).square()
                    local = local + (x.m.square()+x.v) @ (b.square() @ sigma2)
        return x.m @ w + bias, (x.U @ w).square().mean(0) + local

    def block(self, layer, x, mask):
        a = self.attention(layer.attention.self, x, mask)
        a = self.linear(layer.attention.output.dense, a)
        x = self.norm(layer.attention.output.LayerNorm, self.add(x,a))
        h = self.gelu(self.linear(layer.intermediate.dense, x))
        h = self.linear(layer.output.dense, h)
        return self.norm(layer.output.LayerNorm, self.add(x,h))

    def forward(self, model, batch, collect=False, checkpoint_blocks=False):
        self.counter = 0
        net = model.base_model.get_base_model()
        roberta = net.roberta
        m = roberta.embeddings(input_ids=batch['input_ids'])
        x = State(m, torch.zeros((self.rank,)+tuple(m.shape), device=m.device, dtype=m.dtype))
        mask = (1 - batch['attention_mask'][:,None,None,:].to(m.dtype)) * torch.finfo(m.dtype).min
        boundaries = []
        for layer in roberta.encoder.layer:
            if checkpoint_blocks:
                from torch.utils.checkpoint import checkpoint
                start=self.counter
                def evaluate(m,U,layer=layer,start=start):
                    # Recompute exactly the SAME covariance probes in backward.
                    self.counter=start
                    state=self.block(layer,State(m,U),mask)
                    return state.m,state.U
                m,U=checkpoint(evaluate,x.m,x.U,use_reentrant=False,preserve_rng_state=False)
                x=State(m,U)
            else:
                x=self.block(layer,x,mask)
            if collect: boundaries.append((x.m.detach(),x.v.detach()))
        x = State(x.m[:,0],x.U[:,:,0])
        x = self.tanh(self.linear(net.classifier.dense,x))
        mean, variance = self.contrast(net.classifier.out_proj,x)
        return mean, variance, boundaries
