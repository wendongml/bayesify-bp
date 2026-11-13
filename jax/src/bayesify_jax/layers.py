"""Shared moment kernels. Arrays use NHWC images and [in,out] dense weights.

The affine rules require independent inputs and weights with finite second
moments, not Gaussian input data. ReLU/max-pool introduce declared Gaussian
closures. EP and MFVI use these identical maps with different heads/updates.
"""
import math
from functools import partial
import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import ndtr
from .gaussian import Moments,FullMoments,encode,decode


def dense(x,w,b):
    # E[Y]=mM+b; Var[Y]=v M²+(m²+v)V+v_b (diagonal activation closure).
    return Moments(x.mean@w.mean+b.mean,
                   x.variance@(w.mean**2)+(x.mean**2+x.variance)@w.variance+b.variance)


def dense_backward(x,w,b,signal):
    a,c=decode(dense(x,w,b),signal)
    gx=Moments(a@w.mean.T+2*x.mean*(c@w.variance.T),c@(w.mean**2+w.variance).T)
    flat=lambda z:z.reshape(-1,z.shape[-1])
    a,c,m,v=map(flat,(a,c,x.mean,x.variance))
    gw=Moments(m.T@a+2*w.mean*(v.T@c),(m*m+v).T@c)
    return tuple(encode(q,g) for q,g in zip((x,w,b),(gx,gw,Moments(a.sum(0),c.sum(0)))))


def conv(x,w,stride=1,padding=0):
    pad=((padding,padding),(padding,padding)) if isinstance(padding,int) else padding
    return lax.conv_general_dilated(x,w,(stride,stride),pad,dimension_numbers=('NHWC','HWIO','NHWC'))


def convolution(x,w,b,stride=1,padding=0):
    return Moments(conv(x.mean,w.mean,stride,padding)+b.mean,
        conv(x.variance,w.mean**2,stride,padding)+conv(x.mean**2+x.variance,w.variance,stride,padding)+b.variance)


def convolution_backward(x,w,b,signal,stride=1,padding=0):
    # Transposes of the *linear convolution*, not autodiff of the moment map:
    # g_m=C_A* a+2m C_V* c; g_v=C_(A²+V)* c, and analogous filter adjoints.
    a,c=decode(convolution(x,w,b,stride,padding),signal)
    ix=lambda k,g:jax.linear_transpose(lambda z:conv(z,k,stride,padding),jnp.zeros_like(x.mean))(g)[0]
    iw=lambda z,g:jax.linear_transpose(lambda k:conv(z,k,stride,padding),jnp.zeros_like(w.mean))(g)[0]
    gx=Moments(ix(w.mean,a)+2*x.mean*ix(w.variance,c),ix(w.mean**2+w.variance,c))
    gw=Moments(iw(x.mean,a)+2*w.mean*iw(x.variance,c),iw(x.mean**2+x.variance,c))
    gb=Moments(a.sum((0,1,2)),c.sum((0,1,2)))
    return tuple(encode(q,g) for q,g in zip((x,w,b),(gx,gw,gb)))


def relu_terms(x):
    positive=x.variance>0;s=jnp.sqrt(jnp.where(positive,x.variance,1))
    t=x.mean/s;P=ndtr(t);p=jnp.exp(-t*t/2)/math.sqrt(2*math.pi)
    m=x.mean*P+s*p;second=(x.mean*x.mean+x.variance)*P+x.mean*s*p
    v=second-m*m
    return Moments(jnp.where(positive,m,jnp.maximum(x.mean,0)),jnp.where(positive,jnp.maximum(v,0),0)),(positive,s,P,p,m,v)


def relu(x):
    return relu_terms(x)[0]


def relu_backward(x,signal):
    y,(pos,s,P,p,m,v)=relu_terms(x);a,b=decode(y,signal)
    # Price/Stein: dm/dm=P, dm/dv=p/(2s); differentiate E[X_+²]-m_+².
    dm=jnp.where(pos,P,(x.mean>0).astype(x.mean.dtype))
    dv=jnp.where(pos,p/(2*s),0)
    vm=jnp.where(pos&(v>0),2*m*(1-P),0)
    vv=jnp.where(pos&(v>0),P-m*p/s,0)
    return (encode(x,Moments(a*dm+b*vm,a*dv+b*vv)),)


def max_pair(a,b):
    pos=a.variance+b.variance>0;s=jnp.sqrt(jnp.where(pos,a.variance+b.variance,1))
    t=(a.mean-b.mean)/s;P=ndtr(t);p=jnp.exp(-t*t/2)/math.sqrt(2*math.pi)
    m=a.mean*P+b.mean*(1-P)+s*p
    second=(a.variance+a.mean**2)*P+(b.variance+b.mean**2)*(1-P)+(a.mean+b.mean)*s*p
    return Moments(jnp.where(pos,m,jnp.maximum(a.mean,b.mean)),jnp.where(pos,jnp.maximum(second-m*m,0),jnp.where(a.mean>=b.mean,a.variance,b.variance)))


def max_pair_adjoint(a,b,g):
    pos=a.variance+b.variance>0;s=jnp.sqrt(jnp.where(pos,a.variance+b.variance,1))
    t=(a.mean-b.mean)/s;P=ndtr(t);p=jnp.exp(-t*t/2)/math.sqrt(2*math.pi)
    m=a.mean*P+b.mean*(1-P)+s*p
    second=(a.variance+a.mean*a.mean)*P+(b.variance+b.mean*b.mean)*(1-P)+(a.mean+b.mean)*s*p
    gv=g.variance*(second-m*m>0);gm=g.mean-2*m*gv
    common=p*((a.mean+b.mean)/(2*s)-(a.mean-b.mean)*(a.variance-b.variance)/(2*s**3))
    ga=Moments(gm*P+gv*(2*a.mean*P+2*a.variance*p/s),gm*p/(2*s)+gv*(P+common))
    gb=Moments(gm*(1-P)+gv*(2*b.mean*(1-P)+2*b.variance*p/s),gm*p/(2*s)+gv*(1-P+common))
    result=[]
    for h,mask,mm in ((ga,a.mean>=b.mean,(a.mean>b.mean)+.5*(a.mean==b.mean)),(gb,b.mean>a.mean,(b.mean>a.mean)+.5*(a.mean==b.mean))):
        result.append(Moments(jnp.where(pos,h.mean,g.mean*mm),jnp.where(pos,h.variance,g.variance*mask)))
    return tuple(result)


def pool_parts(x):
    h,w=x.mean.shape[1:3];h-=h%2;w-=w%2
    return [Moments(x.mean[:,i:h:2,j:w:2,:],x.variance[:,i:h:2,j:w:2,:]) for i,j in ((0,0),(0,1),(1,0),(1,1))]


def max_pool(x):
    a,b,c,d=pool_parts(x)
    return max_pair(max_pair(a,b),max_pair(c,d))


def max_pool_backward(x,signal):
    a,b,c,d=pool_parts(x);left,right=max_pair(a,b),max_pair(c,d)
    gl,gr=max_pair_adjoint(left,right,decode(max_pair(left,right),signal))
    grads=max_pair_adjoint(a,b,gl)+max_pair_adjoint(c,d,gr)
    gm,gv=jnp.zeros_like(x.mean),jnp.zeros_like(x.variance);h,w=x.mean.shape[1:3];h-=h%2;w-=w%2
    for g,(i,j) in zip(grads,((0,0),(0,1),(1,0),(1,1))):
        gm=gm.at[:,i:h:2,j:w:2,:].set(g.mean);gv=gv.at[:,i:h:2,j:w:2,:].set(g.variance)
    return (encode(x,Moments(gm,gv)),)


def average(x,k=2):
    return lax.reduce_window(x,0.,lax.add,(1,k,k,1),(1,k,k,1),'VALID')/(k*k)


def avg_pool(x,k=2):
    return Moments(average(x.mean,k),average(x.variance,k)/(k*k))


def avg_pool_backward(x,signal,k=2):
    a,b=decode(avg_pool(x,k),signal)
    transpose=lambda g:jax.linear_transpose(lambda z:average(z,k),jnp.zeros_like(x.mean))(g)[0]
    return (encode(x,Moments(transpose(a),transpose(b)/(k*k))),)


def flatten(x):
    # Transpose before flatten to preserve the original NCHW dense ordering.
    flatten_array=lambda z:z.transpose(0,3,1,2).reshape(z.shape[0],-1) if z.ndim==4 else z.reshape(z.shape[0],-1)
    return Moments(flatten_array(x.mean),flatten_array(x.variance))


def add(x,y):
    return Moments(x.mean+y.mean,x.variance+y.variance)


def duplicate(x):
    return Moments(2*x.mean,4*x.variance)


def concatenate(x,y):
    return Moments(jnp.concatenate((x.mean,y.mean),-1),jnp.concatenate((x.variance,y.variance),-1))


def global_pool(x):
    area=x.mean.shape[1]*x.mean.shape[2]
    return Moments(x.mean.mean((1,2)),x.variance.mean((1,2))/area)


def shortcut(x):
    pad=x.mean.shape[-1]//2
    return Moments(*(jnp.pad(z[:,::2,::2,:],((0,0),(0,0),(0,0),(pad,pad))) for z in x))


def normalization(x,gamma,beta,mean,variance,eps=1e-5):
    # Conditional on frozen batch statistics, normalization is affine.
    inv=lax.rsqrt(variance+eps);a=(x.mean-mean)*inv;t=x.variance*inv**2
    return Moments(a*gamma.mean+beta.mean,t*gamma.mean**2+(a*a+t)*gamma.variance+beta.variance)


def normalization_backward(x,gamma,beta,signal,mean,variance,eps=1e-5):
    a,c=decode(normalization(x,gamma,beta,mean,variance,eps),signal)
    inv=lax.rsqrt(variance+eps);m=(x.mean-mean)*inv;v=x.variance*inv**2;axes=tuple(range(x.mean.ndim-1))
    gx=Moments((a*gamma.mean+2*m*c*gamma.variance)*inv,c*(gamma.mean**2+gamma.variance)*inv**2)
    gg=Moments((a*m+2*c*v*gamma.mean).sum(axes),(c*(m*m+v)).sum(axes))
    gb=Moments(a.sum(axes),c.sum(axes))
    return tuple(encode(q,g) for q,g in zip((x,gamma,beta),(gx,gg,gb)))


def projected(forward,parents,signal):
    # The derivative is Euclidean. Coordinate encoding is explicitly separate.
    y,pullback=jax.vjp(forward,*parents)
    ordinary=pullback(decode(y,signal))
    return tuple(encode(q,g) for q,g in zip(parents,ordinary))


def linear_literal(forward,parents,signal):
    # Used only for fixed linear reshape/sum/concatenation maps.
    y=forward(*parents);g=decode(y,signal)
    gradients=jax.linear_transpose(forward,*parents)(g)
    return tuple(encode(q,a) for q,a in zip(parents,gradients))


def readout(x,w,b):
    """Exact output covariance under diagonal input and independent weights."""
    mean=x.mean@w.mean+b.mean
    C=jnp.einsum('bi,ic,id->bcd',x.variance,w.mean,w.mean)
    noise=(x.mean*x.mean+x.variance)@w.variance+b.variance
    return FullMoments(mean,C+jax.vmap(jnp.diag)(noise))


def readout_backward(x,w,b,signal):
    a,G=decode(readout(x,w,b),signal);G=(G+jnp.swapaxes(G,-1,-2))/2;d=jnp.diagonal(G,axis1=-2,axis2=-1)
    gx=Moments(a@w.mean.T+2*x.mean*(d@w.variance.T),jnp.einsum('ic,bcd,id->bi',w.mean,G,w.mean)+d@w.variance.T)
    gw=Moments(x.mean.T@a+2*jnp.einsum('bi,id,bdc->ic',x.variance,w.mean,G),(x.mean*x.mean+x.variance).T@d)
    gb=Moments(a.sum(0),d.sum(0))
    return tuple(encode(q,g) for q,g in zip((x,w,b),(gx,gw,gb)))


def correlated_readout(x,first,pre,w,b,w2,basis):
    # L_b=diag(Phi(pre_m/pre_sd)) W2^T diag(sqrt(v1)) Q.
    sd=jnp.sqrt(jnp.where(first.variance>0,first.variance,1))* (first.variance>0)
    scale=jnp.sqrt(jnp.where(pre.variance>0,pre.variance,1));gate=jnp.where(pre.variance>0,ndtr(pre.mean/scale),(pre.mean>0).astype(pre.mean.dtype))
    U=gate[:,:,None]*jnp.einsum('ih,bir->bhr',w2.mean,sd[:,:,None]*basis[None])
    residual=x.variance-jnp.sum(U*U,-1)
    # Roundoff at zero is allowed; a materially negative residual is diagnosed.
    residual=jnp.maximum(residual,0)
    output=jnp.einsum('ic,bir->bcr',w.mean,U)
    C=jnp.einsum('bi,ic,id->bcd',residual,w.mean,w.mean)+output@jnp.swapaxes(output,-1,-2)
    noise=(x.mean*x.mean+x.variance)@w.variance+b.variance
    return FullMoments(x.mean@w.mean+b.mean,C+jax.vmap(jnp.diag)(noise))


def literal_autodiff(forward,reverse):
    """JAX custom VJP implementing the literal local message rule.

    JAX cotangents are Euclidean. Encode the child's cotangent, apply the
    independent analytic message rule, then decode each parent before handing
    it to JAX. Returning a natural signal as an ordinary gradient is incorrect.
    The explicit graph engine exposes the intermediate signals for inspection.
    """
    @jax.custom_vjp
    def operation(*parents):return forward(*parents)
    def fwd(*parents):
        y=forward(*parents);return y,(parents,y)
    def bwd(cache,adjoint):
        parents,y=cache;signal=encode(y,adjoint)
        return tuple(decode(q,d) for q,d in zip(parents,reverse(*parents,signal)))
    operation.defvjp(fwd,bwd)
    return operation
