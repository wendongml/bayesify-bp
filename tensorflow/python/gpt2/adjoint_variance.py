"""Exact delta-method scalar variance, contracted one module at a time.

For z=z(m)+sum_l J_l delta_w_l with independent weight blocks,
Var(z)=sum_l J_l diag(v_l) J_l^T. Reverse sensitivities retain all
residual/attention branch cross terms. No covariance sketch, BxP global
Jacobian or dense activation covariance is retained. This is not an exact
nonlinear Gaussian pushforward and is currently a binary-output method.
"""
import torch
from torch.nn import functional as F


def binary_marginals(engine,ids,mask,means,variances,joint=False):
    h=engine.embedding(ids);inputs=[]
    for i in range(len(engine.model.backbone.h)):
        inputs.append(h)
        h=engine.block_function(i,mask)(h,*means[4*i:4*i+4])
    z=engine.head_function(mask)(h);contrast=z[:,1]-z[:,0]
    def head_gradient(one_h,one_mask):
        def scalar(x):
            pair=engine.head_function(one_mask[None])(x[None])[0]
            return pair[1]-pair[0]
        return torch.func.grad(scalar)(one_h)
    sensitivity=torch.vmap(head_gradient)(h,mask)
    variance=contrast.new_zeros((len(ids),len(ids))) if joint else torch.zeros_like(contrast)
    for i in reversed(range(len(inputs))):
        weights=means[4*i:4*i+4]
        def local_pull(one_h,one_mask,one_sensitivity):
            f=engine.block_function(i,one_mask[None])
            def local(x,*w):return f(x[None],*w)[0]
            _,pull=torch.func.vjp(local,one_h,*weights)
            return pull(one_sensitivity)
        sensitivity,*J=torch.vmap(local_pull)(inputs[i],mask,sensitivity)
        for j,v in zip(J,variances[4*i:4*i+4]):
            if joint:
                weighted=(j*v.sqrt()).flatten(1)
                variance=variance+weighted@weighted.T
            else:variance=variance+(j.square()*v).flatten(1).sum(1)
    return contrast,variance


def objective(engine,ids,mask,y,means,variances,nodes,weights,expected=True):
    m,v=binary_marginals(engine,ids,mask,means,variances)
    z=m[:,None]+v.clamp_min(1e-30).sqrt()[:,None]*nodes
    ll=F.logsigmoid((2*y[:,None]-1)*z)
    return -(ll*weights).sum(-1).mean() if expected else -torch.logsumexp(ll+weights.log(),-1).mean()
