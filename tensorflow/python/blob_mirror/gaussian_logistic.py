"""Accurate Gaussian logistic expectations, including broad Gaussian contrasts.

For D~N(m,v), softplus(D)=max(D,0)+softplus(-abs(D)). Integrate the
Gaussian positive part analytically; the remaining integrand decays as exp(-|D|).
Splitting at zero also gives stable probability and Price-gradient formulas.
No fitted correction, clipping of variance, or Monte Carlo is used.
"""
from functools import lru_cache
import numpy as np
import torch

@lru_cache(None)
def rules(device):
    x,w=np.polynomial.hermite.hermgauss(128)
    t,a=np.polynomial.legendre.leggauss(128)
    # Integrate the exponentially decaying residual on [0,40].
    return tuple(torch.tensor(z,device=device,dtype=torch.float64)
                 for z in (np.sqrt(2)*x,w/np.sqrt(np.pi),20*(t+1),20*a))

def expectations(mean,variance,labels):
    """Return expected CE, dCE/dmean, dCE/dvariance, and both probabilities."""
    dtype=mean.dtype;m=mean.double();v=variance.double();y=labels.double()
    assert bool((v>=0).all())
    nodes,weights,z,zw=rules(str(mean.device))
    loss=torch.empty_like(m);p0=torch.empty_like(m);p1=torch.empty_like(m);gv=torch.empty_like(m)
    narrow=v<4
    if narrow.any():
        d=m[narrow,None]+v[narrow,None].sqrt()*nodes
        s=d.sigmoid();t=(-d).sigmoid()
        p1[narrow]=(s*weights).sum(-1);p0[narrow]=(t*weights).sum(-1)
        loss[narrow]=(torch.nn.functional.softplus((1-2*y[narrow,None])*d)*weights).sum(-1)
        gv[narrow]=(s*t*weights).sum(-1)/2
    broad=~narrow
    if broad.any():
        mu=m[broad,None];var=v[broad,None];sd=var.sqrt()
        plus=torch.exp(-(z-mu).square()/(2*var))/torch.sqrt(2*torch.pi*var)
        minus=torch.exp(-(-z-mu).square()/(2*var))/torch.sqrt(2*torch.pi*var)
        # Beyond 40, logistic(-z), softplus(-z), and logistic'(z) equal
        # exp(-z) to relative O(exp(-40)). Integrate that tail analytically;
        # discarding it would corrupt log probabilities for confident errors.
        tail_plus=torch.exp(-mu+var/2+torch.special.log_ndtr((mu-var-40)/sd)).squeeze(-1)
        tail_minus=torch.exp(mu+var/2+torch.special.log_ndtr((-mu-var-40)/sd)).squeeze(-1)
        correction=((minus-plus)*(-z).sigmoid()*zw).sum(-1)+tail_minus-tail_plus
        p1[broad]=torch.special.ndtr((mu/sd).squeeze(-1))+correction
        p0[broad]=torch.special.ndtr((-mu/sd).squeeze(-1))-correction
        residual=((plus+minus)*torch.nn.functional.softplus(-z)*zw).sum(-1)+tail_plus+tail_minus
        u=(1-2*y[broad])*m[broad];s=v[broad].sqrt();a=u/s
        positive=s*torch.exp(-a.square()/2)/np.sqrt(2*np.pi)+u*torch.special.ndtr(a)
        loss[broad]=positive+residual
        gv[broad]=(((plus+minus)*z.sigmoid()*(-z).sigmoid()*zw).sum(-1)+tail_plus+tail_minus)/2
    gm=torch.where(labels.bool(),-p0,p1)
    return tuple(t.to(dtype) for t in (loss,gm,gv,p0,p1))
