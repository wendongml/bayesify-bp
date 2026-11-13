"""Shared-factor GPT-2 moment modules, with explicit local reverse traversal.

Initial closure: relinearize at CURRENT means on every forward pass, not a
frozen checkpoint. U carries common uncertainty directions through Q/K/V,
LayerNorm and residual branches. C = mean_r U_r U_r^T is PSD but approximate.
This module is a baseline to diagnose against samples, not an exact Gaussian
transformer or a claim that low-rank covariance is already sufficient.
"""
import math
import torch
from torch.nn import functional as F


class TransformerMoments:
    def __init__(self,model,rank=4,seed=901):
        self.model=model;self.rank=rank
        g=torch.Generator(device=model.label_ids.device).manual_seed(seed)
        self.probes=[torch.randint(0,2,(rank,)+tuple(p.mean.shape),device=p.mean.device,generator=g).to(p.mean.dtype)*2-1
                     for p in model.layers]

    def embedding(self,ids):
        b=self.model.backbone
        positions=torch.arange(ids.shape[1],device=ids.device)
        return b.wte(ids)+b.wpe(positions)

    @staticmethod
    def norm(x,layer):
        return F.layer_norm(x,layer.normalized_shape,layer.weight,layer.bias,layer.eps)

    def block_function(self,index,mask):
        block=self.model.backbone.h[index];attn=block.attn;adapter=attn.c_attn
        assert not attn.scale_attn_by_inverse_layer_idx and not attn.reorder_and_upcast_attn
        def function(h,a,b,c,d):
            x=self.norm(h,block.ln_1)
            qkv=x@adapter.base.weight+adapter.base.bias
            q=(x@a)@b;v=(x@c)@d
            qkv=qkv+adapter.scale*torch.cat((q,torch.zeros_like(q),v),-1)
            q,k,v=qkv.split(h.shape[-1],dim=-1)
            B,T,D=q.shape;H=attn.num_heads
            reshape=lambda t:t.reshape(B,T,H,D//H).transpose(1,2)
            q,k,v=map(reshape,(q,k,v))
            scores=q@k.transpose(-1,-2)
            if attn.scale_attn_weights:scores=scores/math.sqrt(D//H)
            causal=torch.ones((T,T),dtype=torch.bool,device=h.device).tril()
            valid=causal[None,None]&mask[:,None,None,:].bool()
            scores=scores.masked_fill(~valid,torch.finfo(scores.dtype).min)
            context=(scores.softmax(-1)@v).transpose(1,2).reshape(B,T,D)
            h=h+context@attn.c_proj.weight+attn.c_proj.bias
            x=self.norm(h,block.ln_2)
            x=block.mlp.act(x@block.mlp.c_fc.weight+block.mlp.c_fc.bias)
            return h+x@block.mlp.c_proj.weight+block.mlp.c_proj.bias
        return function

    def head_function(self,mask):
        b=self.model.backbone
        def head(h):
            x=self.norm(h,b.ln_f)
            last=x[torch.arange(len(x),device=x.device),mask.sum(-1)-1]
            return F.linear(last,b.wte.weight[self.model.label_ids])
        return head

    @staticmethod
    def push(function,mean,factors,weights,directions):
        """All branch cross terms survive because every operand shares r."""
        primals=(mean,*weights)
        value=function(*primals)
        def tangent(*d):return torch.func.jvp(function,primals,d)[1]
        out=torch.vmap(tangent)(factors,*directions)
        return value,out

    def point(self,ids,mask,weights):
        h=self.embedding(ids)
        for i in range(len(self.model.backbone.h)):h=self.block_function(i,mask)(h,*weights[4*i:4*i+4])
        return self.head_function(mask)(h)

    def forward(self,ids,mask,means,variances,cache=False):
        h=self.embedding(ids);u=torch.zeros((self.rank,)+h.shape,device=h.device,dtype=h.dtype)
        directions=[v.sqrt()[None]*e for v,e in zip(variances,self.probes)]
        trace=[]
        for i in range(len(self.model.backbone.h)):
            f=self.block_function(i,mask);w=means[4*i:4*i+4];d=directions[4*i:4*i+4]
            if cache:trace.append((f,h,u,w,d))
            h,u=self.push(f,h,u,w,d)
        if cache:trace.append((self.head_function(mask),h,u,[],[]))
        m,U=self.push(self.head_function(mask),h,u,[],[])
        return (m,U),trace

    def reverse(self,trace,grad_mean,grad_factors,variances):
        """Local cached VJP; no exact-posterior or global-Jacobian substitution.

        Adjacent moment-signal encode/decode cancel at regular boundaries.
        Singular low-rank boundaries remain in factor coordinates rather than
        adding fake covariance to make a Fisher inverse possible.
        """
        gm=[];gd=[];a,b=grad_mean,grad_factors
        for f,h,u,w,d in reversed(trace):
            n=len(w)
            def local(hh,uu,*z):return self.push(f,hh,uu,z[:n],z[n:])
            _,pull=torch.func.vjp(local,h,u,*w,*d)
            result=pull((a,b));a,b=result[:2]
            if n:gm[0:0]=result[2:2+n];gd[0:0]=result[2+n:]
        gv=[(g*e).sum(0)/(2*v.sqrt()) for g,e,v in zip(gd,self.probes,variances)]
        return gm,gv


def binary_head(m,U,y,nodes,node_weights,expected=True,power=1.):
    """Integrate the scalar logit contrast, not independent class probabilities."""
    mean=m[:,1]-m[:,0]
    variance=(U[:,:,1]-U[:,:,0]).square().mean(0)
    z=mean[:,None]+variance.clamp_min(1e-30).sqrt()[:,None]*nodes
    ll=F.logsigmoid((2*y[:,None]-1)*z)
    if expected:return -(ll*node_weights).sum(-1).mean()
    return -(torch.logsumexp(power*ll+node_weights.log(),-1)/power).mean()


def binary_probability(m,U,nodes,node_weights):
    mean=m[:,1]-m[:,0];variance=(U[:,:,1]-U[:,:,0]).square().mean(0)
    positive=(torch.sigmoid(mean[:,None]+variance.clamp_min(1e-30).sqrt()[:,None]*nodes)*node_weights).sum(-1)
    return torch.stack((1-positive,positive),-1)
