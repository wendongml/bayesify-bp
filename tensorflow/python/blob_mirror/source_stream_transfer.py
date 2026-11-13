"""The fixed source conditional-moment rule, on the two transfer architectures.

This is a HYBRID: conditional moments within each block, a stochastic carrier
between blocks. It is not deterministic Gaussian moment training. No optimizer,
prior, training data or likelihood is defined here; those remain in the source
transfer protocol. The reference and K=1 paths are exactness controls only.
"""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from source_stream_moments import transition as roberta_transition


def gpt2_transition(layer, h, mask, K, seed, reference=False):
    generator = torch.Generator(device=h.device).manual_seed(seed)
    a = torch.randn(K, device=h.device, dtype=h.dtype, generator=generator)
    a = a / a.norm()
    b = torch.randn(K, device=h.device, dtype=h.dtype, generator=generator)
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
                # Exactly the released Flipout shapes and global RNG order.
                r = torch.ones_like(x[0]).uniform_(-1, 1).sign()
                t = x.new_ones(x.shape[1:-1] + (A.shape[0],)).uniform_(-1, 1).sign()
                xi = torch.randn_like(A)
                if reference:
                    nodes = xi[None]
                else:
                    Z = torch.randn((K,) + tuple(A.shape), device=x.device,
                                    dtype=x.dtype, generator=generator)
                    av = a[:, None, None]
                    # Xi_i = a_i xi + Z_i - a_i sum_j a_j Z_j.
                    # Cov(Xi_i,Xi_j | a) = delta_ij I and sum_i a_i Xi_i = xi.
                    nodes = av * xi + Z - av * (av * Z).sum(0)
                    nodes = torch.cat([nodes, -nodes])
                noise = torch.einsum('k...d,krd->k...r', x*r,
                                     nodes*layer.lora_A_rho[name].square()) * t
                y = y + F.linear(noise, B) * scale
        return y

    x = h[None].expand(1 if reference else 2*K, *h.shape)
    att = layer.attn
    assert att.scale_attn_weights and not att.scale_attn_by_inverse_layer_idx
    assert not att.reorder_and_upcast_attn
    N, B, T, D = x.shape
    H, d = att.num_heads, att.head_dim
    u = layer.ln_1(x)
    q, k, v = [linear(getattr(att.c_attn, n), u).reshape(N,B,T,H,d).transpose(2,3)
               for n in ['query', 'key', 'value']]
    causal = torch.ones(T,T,device=x.device,dtype=torch.bool).tril()
    am = mask.masked_fill(~causal, torch.finfo(x.dtype).min)
    am = am[None].expand(N,*am.shape).reshape(N*B,*am.shape[1:])
    y = F.scaled_dot_product_attention(q.reshape(N*B,H,T,d), k.reshape(N*B,H,T,d),
                                      v.reshape(N*B,H,T,d), attn_mask=am, dropout_p=0.)
    y = y.reshape(N,B,H,T,d).transpose(2,3).reshape(N,B,T,D)
    x = x + linear(att.c_proj, y)
    y = layer.mlp.act(linear(layer.mlp.c_fc, layer.ln_2(x)))
    y = x + linear(layer.mlp.c_proj, y)
    if reference:
        return y[0]
    plus, minus = y[:K], y[K:]
    mean = (plus+minus).mean(0)/2
    odd, even = (plus-minus)/2, (plus+minus)/2-mean
    shape = (K,) + (1,)*mean.ndim
    # Conditional E[T]=mean; Cov(T)=sum_i(odd_i odd_i' + even_i even_i')/K.
    return mean + (a.reshape(shape)*odd+b.reshape(shape)*even).sum(0)


class SourceStreamTransfer:
    def __init__(self, nodes=2, seed=197, reference=False):
        self.nodes, self.seed, self.reference = nodes, seed, reference

    def forward_task(self, model, batch, task, target_ids=None, checkpoint_blocks=False):
        net = model.base_model.get_base_model()
        if task == 'sst2':
            positions = torch.arange(batch['input_ids'].shape[1],device=batch['input_ids'].device)[None]
            h = net.transformer.wte(batch['input_ids']) + net.transformer.wpe(positions)
            layers, transition = net.transformer.h, gpt2_transition
        else:
            h = net.roberta.embeddings(input_ids=batch['input_ids'])
            layers, transition = net.roberta.encoder.layer, roberta_transition
        mask = (1-batch['attention_mask'][:,None,None,:].to(h.dtype))*torch.finfo(h.dtype).min
        for i, layer in enumerate(layers):
            def f(h, layer=layer, i=i):
                return transition(layer,h,mask,self.nodes,self.seed+104729*i,self.reference)
            h = checkpoint(f,h,use_reentrant=False,preserve_rng_state=True) if checkpoint_blocks else f(h)
        if task == 'sst2':
            h = net.transformer.ln_f(h)
            last = batch['attention_mask'].sum(-1)-1
            h = h[torch.arange(len(h),device=h.device),last]
            return F.linear(h,net.lm_head.weight[target_ids])
        return net.classifier(h)
