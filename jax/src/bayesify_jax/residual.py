"""Nonlinear-residual correction, mirroring R/covariance.R.

Freeze the linearization and proposal for a COMPLETE backward pass. Correct
E[h] and its pathwise derivatives; for EP divide the corrected dZ by corrected
Z only at the end. The <=2 Gaussian-mixture importance ratio is analytic, not
estimated from particles. This changes integration, not the Gaussian belief.
"""
import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc
import jax,jax.numpy as j
from jax.flatten_util import ravel_pytree
from .gaussian import Moments,encode


def residual_backward(q,gaussian,nonlinear,surrogate,weight,method='mfvi'):
    """Rows are integration draws; gm/gv are pathwise LOG-likelihood derivatives."""
    a=j.exp(nonlinear['log_likelihood']) if method=='ep' else j.ones_like(weight)
    b=j.exp(surrogate['log_likelihood']) if method=='ep' else j.ones_like(weight)
    gm=gaussian['gm']+j.mean(weight[:,None]*(a[:,None]*nonlinear['gm']-b[:,None]*surrogate['gm']),0)
    gv=gaussian['gv']+j.mean(weight[:,None]*(a[:,None]*nonlinear['gv']-b[:,None]*surrogate['gv']),0)
    Z=gaussian['Z']+j.mean(weight*(j.expm1(nonlinear['log_likelihood'])-j.expm1(surrogate['log_likelihood']))) if method=='ep' else j.asarray(1.,q.mean.dtype)
    gradient=Moments(gm/Z,gv/Z)
    return gradient,encode(q,gradient),Z


def network_residual_backward(predict,beliefs,label,key,pairs=128,head_power=14):
    """One input: full-class Gaussian surrogate + antithetic shifted correction.

    predict(weights) returns K logits. No dense P×P covariance is constructed;
    J is K×P, so this diagnostic is intended for small query/output sets.
    Both EP and MFVI share samples and a frozen linear map. The returned group
    contributions support independent-group Monte Carlo standard errors.
    """
    m,unravel=ravel_pytree({n:p.mean for n,p in beliefs.items()});v,_=ravel_pytree({n:p.variance for n,p in beliefs.items()})
    f=lambda w:predict(unravel(w));z=f(m);J=jax.jacrev(f)(m);C=(J*v)@J.T;K=len(z)
    other=[i for i in range(K) if i!=int(label)];D=j.eye(K)[j.array(other)]-j.eye(K)[label]
    a=D@z;L=j.linalg.cholesky(D@C@D.T);competitor=other[int(j.argmax(a))]
    delta=L[other.index(competitor)];normal=j.asarray(ndtri(qmc.Sobol(K-1,scramble=True,seed=173).random_base2(head_power-1)),dtype=m.dtype)
    e=j.concatenate((normal,normal+delta));ratio=2*jax.nn.sigmoid(j.sum(delta*delta)/2-e@delta)
    nodes=j.zeros((len(e),K),m.dtype).at[:,j.array(other)].set(a+e@L.T)
    p=jax.nn.softmax(nodes);score=jax.nn.one_hot(label,K)-p;H=p[:,:,None]*p[:,None,:]-jax.vmap(j.diag)(p);py=p[:,label]
    shift=j.sqrt(v)*(J[competitor]-J[label]);eps=jax.random.normal(key,(pairs,len(m)),dtype=m.dtype)
    # Four dependent nodes per independent group: +/-epsilon, shifted/unshifted.
    eps=j.stack((eps,-eps,eps+shift,-eps+shift),1).reshape(-1,len(m))
    w=2*jax.nn.sigmoid(j.sum(shift*shift)/2-eps@shift)
    logp=lambda t:jax.nn.log_softmax(f(t))[label]
    ell,g=jax.vmap(jax.value_and_grad(logp))(m+j.sqrt(v)*eps)
    lp0=jax.nn.log_softmax(z+(eps*j.sqrt(v))@J.T);ell0=lp0[:,label];g0=(jax.nn.one_hot(label,K)-j.exp(lp0))@J
    result={}
    for method in ['mfvi','ep']:
        factor=py if method=='ep' else j.ones_like(py)
        gz=j.mean(ratio[:,None]*factor[:,None]*score,0)
        hessian=H+score[:,:,None]*score[:,None,:] if method=='ep' else H
        GC=.5*j.mean(ratio[:,None,None]*factor[:,None,None]*hessian,0)
        Z=j.mean(ratio*py) if method=='ep' else j.asarray(1.,m.dtype)
        gaussian={'gm':J.T@gz,'gv':j.einsum('ik,ij,jk->k',J,GC,J),'Z':Z}
        nonlinear={'log_likelihood':ell,'gm':g,'gv':g*eps/(2*j.sqrt(v))}
        surrogate={'log_likelihood':ell0,'gm':g0,'gv':g0*eps/(2*j.sqrt(v))}
        gradient,signal,normalizer=residual_backward(Moments(m,v),gaussian,nonlinear,surrogate,w,method)
        result[method]={'gradient':gradient,'signal':signal,'Z':normalizer,'surrogate':gaussian}
    return result
