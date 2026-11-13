"""Moment forward -> explicit local reverse -> the SAME optimizer recurrence."""
import math
import numpy as np
import torch
from torch.nn import functional as F
from moments import TransformerMoments,binary_head,binary_probability
from optimizers import cavity_field,field_pullback


class MomentStep:
    def __init__(self,model,rank=4,backend='shared_factors'):
        self.model=model;self.engine=TransformerMoments(model,rank);self.backend=backend
        x,w=np.polynomial.hermite.hermgauss(20)
        self.nodes=torch.tensor(x*math.sqrt(2),device='cuda',dtype=torch.float32)
        self.weights=torch.tensor(w/math.sqrt(math.pi),device='cuda',dtype=torch.float32)

    def step(self,ids,mask,y,opt,method,n,tau,p,generator):
        # Beliefs remain fixed throughout this entire forward/reverse snapshot.
        with torch.no_grad():
            means,vs=self.model.belief();means=[m.detach() for m in means]
            vi=method=='moment_vadam'
            cm,cv,field=(means,vs,0.) if vi else cavity_field(means,vs,n,p,tau,tau)
            if self.backend=='shared_factors':(m,U),cache=self.engine.forward(ids,mask,cm,cv,cache=True)
        if self.backend=='adjoint_contraction':
            from adjoint_variance import objective
            def loss(a,b):return objective(self.engine,ids,mask,y,a,b,self.nodes,self.weights,vi)
            (gm,gv),value=torch.func.grad_and_value(loss,argnums=(0,1))(cm,cv)
        elif self.backend=='shared_factors':
            value=binary_head(m,U,y,self.nodes,self.weights,expected=vi)
            a,b=torch.func.grad(binary_head,argnums=(0,1))(m,U,y,self.nodes,self.weights,vi)
            gm,gv=self.engine.reverse(cache,a,b,cv)
        else:raise ValueError(self.backend)
        if vi:
            # Gaussian tangent approximation to gradient variability, refreshed
            # every update; not squared mean gradient alone, not EP curvature.
            direction=[v.sqrt()*torch.randn(v.shape,device=v.device,generator=generator) for v in vs]
            gradient=torch.func.grad(lambda ww:F.cross_entropy(self.engine.point(ids,mask,ww),y))
            _,hd=torch.func.jvp(gradient,(means,),(direction,))
            opt.step(gm,[g.square()+d.square() for g,d in zip(gm,hd)])
        else:
            gm,gv=field_pullback(means,vs,gm,gv,n,p,tau,tau)
            opt.zero_grad(set_to_none=True)
            for layer,g,h,v in zip(self.model.layers,gm,gv,vs):
                layer.mean.grad=g.detach();layer.log_variance.grad=(v*h).detach()
            opt.step()
        return value.detach()+field,field

    @torch.no_grad()
    def probabilities(self,ids,mask):
        means,vs=self.model.belief()
        if self.backend=='adjoint_contraction':
            from adjoint_variance import binary_marginals
            m,v=binary_marginals(self.engine,ids,mask,means,vs)
            positive=(torch.sigmoid(m[:,None]+v.clamp_min(1e-30).sqrt()[:,None]*self.nodes)*self.weights).sum(-1)
            return torch.stack((1-positive,positive),-1)
        (m,U),_=self.engine.forward(ids,mask,means,vs)
        return binary_probability(m,U,self.nodes,self.weights)
