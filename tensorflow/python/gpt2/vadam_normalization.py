"""Explicitly labelled Vadam initialization/normalization ablation.

The original Vadam baseline is unchanged. Here the posterior-precision EMA
retains its declared nonzero initialization, while Adam normalization uses a
separate likelihood-only EMA initialized at zero. Algebraically,
s_t = beta2**t*s_0 + u_t. Thus u_t/(1-beta2**t), not s_t/(1-beta2**t),
is the ordinary bias-corrected gradient-square average. This changes transient
mean updates, not the stated variance formula. It is not an exact-inference claim.
"""
import torch
from optimizers import Vadam

class VadamLikelihoodEMA(Vadam):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.u=[torch.zeros_like(p.mean) for p in self.layers]

    @torch.no_grad()
    def step(self,first,second):
        assert self.rule=='coupled'
        self.t+=1;b1,b2=self.betas;lam=self.p/self.n
        for p,r,s,u,g,g2 in zip(self.layers,self.r,self.s,self.u,first,second):
            r.mul_(b1).add_((1-b1)*(g+lam*p.mean))
            s.mul_(b2).add_((1-b2)*g2)
            u.mul_(b2).add_((1-b2)*g2)
            metric=(u/(1-b2**self.t)).sqrt()+lam
            p.mean.add_(-self.lr*(r/(1-b1**self.t))/metric)
            p.log_variance.copy_((1/(self.n*s+self.p)).log())

    def state_dict(self):
        result=super().state_dict();result['likelihood_ema']=self.u
        result['normalization']='likelihood_ema_initial_zero';return result

    def load_state_dict(self,state):
        assert state['normalization']=='likelihood_ema_initial_zero'
        super().load_state_dict(state)
        for a,b in zip(self.u,state['likelihood_ema']):a.copy_(b)
