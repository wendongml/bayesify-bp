"""Conditional moments coupled to the RELEASED forward's random-number stream.

The source draws xi_* and Flipout signs from the global Torch generator.
Auxiliary cubature uses a separate generator and cannot advance that stream.
For a unit vector a and iid auxiliary Gaussian arrays Z_i, set
    Xi_i = a_i xi_* + Z_i - a_i sum_j a_j Z_j.
Conditional on a, the Xi_i are jointly iid standard Gaussian, and
sum_i a_i Xi_i = xi_*. Thus this is the same conditional-moment law as the
earlier rotated construction, with a coupling that preserves the source RNG.
This remains a stochastic HYBRID; it is not deterministic Gaussian closure.
"""
import math
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def transition(layer, h, mask, K, seed, reference=False):
    # Private cubature randomness: the original source consumes none of it.
    auxiliary = torch.Generator(device=h.device).manual_seed(seed)
    a = torch.randn(K, device=h.device, dtype=h.dtype, generator=auxiliary)
    a = a / a.norm()
    b = torch.randn(K, device=h.device, dtype=h.dtype, generator=auxiliary)
    b = b / b.norm()

    def linear(layer, x):
        base = layer.base_layer if hasattr(layer, 'base_layer') else layer
        y = F.linear(x, base.weight, base.bias)
        if not hasattr(layer, 'lora_A'):
            return y
        for name in layer.active_adapters:
            A, B = layer.lora_A[name].weight, layer.lora_B[name].weight
            scale = layer.scaling[name]
            y = y + F.linear(F.linear(x, A), B) * scale
            if hasattr(layer, 'lora_A_rho') and layer.blobsample:
                assert layer.bayes_eps >= 0
                # EXACT source draw order/shapes, with the node axis removed.
                r = torch.ones_like(x[0]).uniform_(-1, 1).sign()
                t = x.new_ones(x.shape[1:-1] + (A.shape[0],)).uniform_(-1, 1).sign()
                xi = torch.randn_like(A)
                if reference:
                    nodes = xi[None]
                else:
                    Z = torch.randn((K,) + tuple(A.shape), device=x.device,
                                    dtype=x.dtype, generator=auxiliary)
                    av = a[:, None, None]
                    nodes = av * xi + Z - av * (av * Z).sum(0)
                    nodes = torch.cat([nodes, -nodes])
                noise = torch.einsum('k...d,krd->k...r', x * r,
                                     nodes * layer.lora_A_rho[name].square()) * t
                y = y + F.linear(noise, B) * scale
        return y

    x = h[None].expand(1 if reference else 2*K, *h.shape)
    att = layer.attention.self
    N, B, T, D = x.shape
    H, d = att.num_attention_heads, att.attention_head_size
    q, k, v = [linear(getattr(att, n), x).reshape(N, B, T, H, d).transpose(2, 3)
               for n in ['query', 'key', 'value']]
    masks = mask[None].expand(N, *mask.shape).reshape(N*B, *mask.shape[1:])
    y = F.scaled_dot_product_attention(q.reshape(N*B, H, T, d),
                                      k.reshape(N*B, H, T, d),
                                      v.reshape(N*B, H, T, d),
                                      attn_mask=masks, dropout_p=0.)
    y = y.reshape(N, B, H, T, d).transpose(2, 3).reshape(N, B, T, D)
    y = linear(layer.attention.output.dense, y)
    ln = layer.attention.output.LayerNorm
    x = F.layer_norm(x+y, ln.normalized_shape, ln.weight, ln.bias, ln.eps)
    y = layer.intermediate.intermediate_act_fn(linear(layer.intermediate.dense, x))
    y = linear(layer.output.dense, y)
    ln = layer.output.LayerNorm
    y = F.layer_norm(x+y, ln.normalized_shape, ln.weight, ln.bias, ln.eps)
    if reference:
        return y[0]
    plus, minus = y[:K], y[K:]
    mean = (plus+minus).mean(0)/2
    odd, even = (plus-minus)/2, (plus+minus)/2-mean
    shape = (K,) + (1,)*mean.ndim
    return mean + (a.reshape(shape)*odd+b.reshape(shape)*even).sum(0)


class SourceStreamMoments:
    def __init__(self, nodes=2, seed=197, reference=False):
        self.nodes, self.seed, self.reference = nodes, seed, reference

    def forward(self, model, batch, checkpoint_blocks=True):
        net = model.base_model.get_base_model()
        h = net.roberta.embeddings(input_ids=batch['input_ids'])
        mask = ((1-batch['attention_mask'][:, None, None, :].to(h.dtype))
                * torch.finfo(h.dtype).min)
        for i, layer in enumerate(net.roberta.encoder.layer):
            def f(h, layer=layer, i=i):
                return transition(layer, h, mask, self.nodes,
                                  self.seed+104729*i, self.reference)
            # Replay the same source randomness during recomputation, without
            # advancing it a second time. Auxiliary randomness is seeded locally.
            h = checkpoint(f, h, use_reentrant=False, preserve_rng_state=True) if checkpoint_blocks else f(h)
        return net.classifier(h)
