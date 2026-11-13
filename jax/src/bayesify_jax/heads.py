"""EP log-normalizer and MFVI expected log likelihood are separate objectives."""
import math
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from .gaussian import encode,Moments,FullMoments,RootMoments


def sampled_readout_adjoint(features,logits,label,method='mfvi'):
    """Independent MC reference for a Gaussian last-layer weight block.

    Conditional on hidden h, z=h W+b is affine in independent Gaussian W.
    Price: d_v E[f]=E[d_W² f]/2; d_m E[f]=E[d_W f]. For log softmax,
    r=e_y-p and diag(H)=-p(1-p). For f=p_y, multiply by p_y and
    replace diag(H) by diag(H)+r². EP divides averaged dZ by averaged Z
    AFTER all samples. The hidden features need not have a Gaussian law.
    Returns per-network values and ordinary (mean, variance) adjoints.
    """
    logp=jax.nn.log_softmax(logits,-1);p=jnp.exp(logp)
    r=jax.nn.one_hot(label,p.shape[-1],dtype=p.dtype)-p;diagonal=-p*(1-p)
    if method=='mfvi':value=logp[:,label];scale=jnp.ones_like(value)
    elif method=='ep':
        value=p[:,label];scale=value;diagonal=diagonal+r*r
    else:raise ValueError(method)
    h=features[:,:,None];a=scale[:,None,None]
    return value,Moments(a*h*r[:,None,:],.5*a*h*h*diagonal[:,None,:])


def gaussian_head(q,y,noise,method='ep'):
    if method=='ep':
        S=q.variance+noise;e=y-q.mean
        value=-.5*jnp.sum(jnp.log(2*jnp.pi*S)+e*e/S)
        gradient=(e/S,.5*(e*e/S**2-1/S))
    elif method=='mfvi':
        e=y-q.mean
        value=-.5*jnp.sum(jnp.log(2*jnp.pi*noise)+(e*e+q.variance)/noise)
        gradient=(e/noise,jnp.full_like(q.variance,-.5)/noise)
    else:raise ValueError(method)
    return value,encode(q,gradient)


def categorical_nodes(q):
    if isinstance(q,(FullMoments,RootMoments)):
        k=q.mean.shape[-1];D=jnp.concatenate((jnp.eye(k-1,dtype=q.mean.dtype),-jnp.ones((k-1,1),dtype=q.mean.dtype)),1)
        # Contrast subtraction can cancel large, correlated entries. TF32
        # matmul rounding can turn a valid covariance indefinite; retain full
        # float32 multiplication precision here, without adding model jitter.
        m=q.mean@D.T
        if isinstance(q,RootMoments):
            # If (D R)'=Q T, then D C D'=T'T. Choosing positive diagonal
            # gives the SAME Cholesky factor/node rule, without squaring the
            # condition number in the numerical factorization.
            A=jnp.einsum('ij,bjr->bir',D,q.root,precision=jax.lax.Precision.HIGHEST)
            _,T=jnp.linalg.qr(jnp.swapaxes(A,-1,-2),mode='reduced')
            sign=jnp.where(jnp.diagonal(T,axis1=-2,axis2=-1)>=0,1.,-1.).astype(T.dtype)
            L=jnp.swapaxes(T,-1,-2)*sign[:,None,:]
        else:
            C=jnp.einsum("ij,bjk,lk->bil",D,q.covariance,D,precision=jax.lax.Precision.HIGHEST)
            L=jnp.linalg.cholesky(C)
        delta=jnp.sqrt(k-1)*jnp.moveaxis(L,-1,0)
        nodes=jnp.concatenate((m[None]+delta,m[None]-delta),0)
        return jnp.pad(nodes,((0,0),(0,0),(0,1)))
    k=q.mean.shape[-1];offset=jnp.sqrt(k*q.variance)[None,...]*jnp.eye(k,dtype=q.mean.dtype)[:,None,:]
    return jnp.concatenate((q.mean[None]+offset,q.mean[None]-offset),0)


def categorical_objective(nodes,labels,method='mfvi',power=1.):
    logp=jax.nn.log_softmax(nodes,-1)
    selected=jnp.take_along_axis(logp,labels[None,:,None],-1)[...,0]
    if method=='mfvi':return selected.mean(0).sum()
    if method=='ep':return ((logsumexp(power*selected,axis=0)-math.log(nodes.shape[0]))/power).sum()
    raise ValueError(method)


def categorical_head(q,labels,method='mfvi',power=1.):
    f=lambda q:categorical_objective(categorical_nodes(q),labels,method,power)
    value,g=jax.value_and_grad(f)(q)
    return value,encode(q,g)


def probabilities(q):
    return jax.nn.softmax(categorical_nodes(q),-1).mean(0)


def full_covariance_probabilities(mean,covariance):
    # Softmax is invariant to a common shift; integrate C-1 contrasts.
    k=mean.shape[-1];D=jnp.concatenate((jnp.eye(k-1,dtype=mean.dtype),-jnp.ones((k-1,1),dtype=mean.dtype)),1)
    a=mean@D.T;C=jnp.einsum('ij,bjk,lk->bil',D,covariance,D,precision=jax.lax.Precision.HIGHEST)
    # PSD eigenfactor supports singular contrasts without artificial jitter.
    values,vectors=jnp.linalg.eigh(C);L=vectors*jnp.sqrt(jnp.maximum(values,0))[:,None,:]
    delta=jnp.sqrt(k-1)*jnp.moveaxis(L,-1,0)
    nodes=jnp.concatenate((a[None]+delta,a[None]-delta),0)
    return jax.nn.softmax(jnp.pad(nodes,((0,0),(0,0),(0,1))),-1).mean(0)


def binary_gaussian(q,labels=None,method='mfvi',order=20):
    """Integrate a Gaussian logit CONTRAST, as in the original scalar head.

    MFVI sums E log sigmoid(s z); EP sums log E sigmoid(s z). Prediction
    integrates sigmoid before averaging. Nodes are scalar quadrature points.
    """
    import numpy as np
    t,w=np.polynomial.hermite.hermgauss(order)
    t=jnp.asarray(t*math.sqrt(2),q.mean.dtype);w=jnp.asarray(w/math.sqrt(math.pi),q.mean.dtype)
    z=q.mean[:,None]+jnp.sqrt(q.variance[:,None])*t
    if labels is None:
        p=(jax.nn.sigmoid(z)*w).sum(-1);return jnp.stack((1-p,p),-1)
    logp=jax.nn.log_sigmoid((2*labels[:,None]-1)*z)
    return (logp*w).sum() if method=='mfvi' else logsumexp(logp+jnp.log(w),axis=-1).sum()
