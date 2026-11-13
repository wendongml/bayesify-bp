"""Gaussian Fisher geometry; all objectives below are maximized.

q=N(m,v), raw second s=m²+v.  An ordinary adjoint (a,b) encodes as
  dm=v*a, ds=2*m*dm+2*v²*b.
The centered second signal is stored separately to avoid cancellation.
Deterministic observed inputs carry ordinary adjoints without a Fisher inverse.
"""
from typing import NamedTuple
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve

class Moments(NamedTuple):
    mean: object
    variance: object

class Signal(NamedTuple):
    mean: object
    second: object
    centered: object
    zero_mean: object
    zero_variance: object

class FullMoments(NamedTuple):
    mean: object
    covariance: object

class RootMoments(NamedTuple):
    """Same Gaussian, stored as C=R R' to preserve small positive directions."""
    mean: object
    root: object
    @property
    def covariance(self):
        import jax
        return jnp.einsum('...ir,...jr->...ij',self.root,self.root,precision=jax.lax.Precision.HIGHEST)


def encode(q, adjoint):
    if isinstance(q,FullMoments):return encode_full(q,adjoint)
    a,b=adjoint
    active=q.variance>jnp.sqrt(jnp.finfo(q.variance.dtype).tiny)
    dm=jnp.where(active,q.variance*a,0)
    c=jnp.where(active,2*q.variance**2*b,0)
    return Signal(dm,2*q.mean*dm+c,c,jnp.where(active,0,a),jnp.where(active,0,b))


def decode(q,d):
    if isinstance(q,FullMoments):return FullMoments(*decode_full(q,d))
    active=q.variance>jnp.sqrt(jnp.finfo(q.variance.dtype).tiny)
    v=jnp.where(active,q.variance,1)
    return Moments(jnp.where(active,d.mean/v,d.zero_mean),
                   jnp.where(active,d.centered/(2*v*v),d.zero_variance))


def add_signal(a,b):
    return b if a is None else Signal(*(x+y for x,y in zip(a,b)))


def encode_full(q,g):
    a,b=g;b=(b+jnp.swapaxes(b,-1,-2))/2;dm=jnp.einsum("...ij,...j->...i",q.covariance,a)
    return dm,q.mean[..., :,None]*dm[...,None,:]+dm[..., :,None]*q.mean[...,None,:]+2*q.covariance@b@q.covariance


def decode_full(q,d):
    dm,dM=d;L=jnp.linalg.cholesky(q.covariance)
    solve=lambda x:cho_solve((L,True),x)
    c=dM-q.mean[..., :,None]*dm[...,None,:]-dm[..., :,None]*q.mean[...,None,:]
    gm=solve(dm[...,None])[...,0]
    return gm,jnp.swapaxes(solve(jnp.swapaxes(solve(c),-1,-2)),-1,-2)/2


def ep_moment_update(q,d,rho=1.):
    # Raw-moment interpolation: v+=v+rho*c-rho²*dm². No lost quadratic term.
    return Moments(q.mean+rho*d.mean,q.variance+rho*d.centered-rho*rho*d.mean**2)


def mfvi_update(q,d,prior,rho=1.):
    # eta+=(1-rho)eta+rho(eta0+grad_mu E log likelihood).
    gm,gv=decode(q,d)
    precision=(1-rho)/q.variance+rho*(1/prior.variance-2*gv)
    information=(1-rho)*q.mean/q.variance+rho*(prior.mean/prior.variance+gm-2*q.mean*gv)
    return Moments(information/precision,1/precision)


def kl_diagonal(q,p):
    return .5*jnp.sum((q.variance+(q.mean-p.mean)**2)/p.variance-1+jnp.log(p.variance/q.variance))


def validate(q):
    # Explicit boundary validation outside jit; numerical kernels never clip a bad belief.
    import numpy as np
    if not np.all(np.isfinite(q.mean)) or not np.all(np.isfinite(q.variance)) or np.any(q.variance<0):
        raise ValueError('belief requires finite mean and nonnegative variance')
