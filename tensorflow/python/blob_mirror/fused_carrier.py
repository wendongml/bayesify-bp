"""Fuse the SAME GPT-2 MLP carrier; no approximation or optimizer change."""
import math
import torch
import torch.nn.functional as F

def mlp_carrier(x,c,lnw,lnb,eps,w1,b1,w2,b2):
    # Linearity and sum(c)=1 permit the final affine map after the reduction.
    u=F.layer_norm(x,lnw.shape,lnw,lnb,eps)
    z=F.linear(u,w1,b1)
    z=.5*z*(1+torch.tanh(math.sqrt(2/math.pi)*(z+.044715*z.pow(3))))
    return torch.einsum('k,kbtd->btd',c,x)+F.linear(torch.einsum('k,kbtd->btd',c,z),w2,b2)

compiled_mlp_carrier=torch.compile(mlp_carrier,dynamic=True,fullgraph=True)

def apply_mlp(layer,x,c):
    return compiled_mlp_carrier(x,c,layer.ln_2.weight,layer.ln_2.bias,layer.ln_2.eps,
        layer.mlp.c_fc.weight,layer.mlp.c_fc.bias,layer.mlp.c_proj.weight,layer.mlp.c_proj.bias)
