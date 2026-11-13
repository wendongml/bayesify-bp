"""Shared activation directions and correlated Gaussian weights.

H=m+U epsilon/sqrt(R). A direction is shared across branches and examples.
D+LLᵀ weight covariance is a different object from a finite sketch of diagonal
weights. We expose both and never infer one from the other.
"""
from typing import NamedTuple
import jax
import jax.numpy as jnp
from .gaussian import Moments
from . import layers as L
from .messages import local_map

class Shared(NamedTuple):
    mean:object
    factors:object
    @property
    def variance(self):return jnp.mean(self.factors**2,0)
    @property
    def rank(self):return self.factors.shape[0]

class StructuredGaussian(NamedTuple):
    mean:object
    diagonal:object
    factors:object
    def sample(self,key,samples):
        k1,k2=jax.random.split(key)
        return self.mean+jnp.sqrt(self.diagonal)*jax.random.normal(k1,(samples,)+self.mean.shape,dtype=self.mean.dtype)+jax.random.normal(k2,(samples,self.factors.shape[0]),dtype=self.mean.dtype)@self.factors
    def covariance(self):return jnp.diag(self.diagonal)+self.factors.T@self.factors
    def logdet(self):
        # Matrix determinant lemma: |D+LLᵀ|=|D| |I+LᵀD⁻¹L|.
        A=jnp.eye(self.factors.shape[0])+(self.factors/self.diagonal)@self.factors.T
        return jnp.log(self.diagonal).sum()+jnp.linalg.slogdet(A)[1]
    def solve(self,b):
        # Woodbury, with small rank×rank solve. Requires strictly positive D.
        U=self.factors;A=jnp.eye(U.shape[0])+(U/self.diagonal)@U.T
        return b/self.diagonal-(U.T@jnp.linalg.solve(A,U@(b/self.diagonal)))/self.diagonal


def directions(q,key,rank):
    return jnp.sqrt(q.variance)*jax.random.rademacher(key,(rank,)+q.mean.shape,dtype=q.mean.dtype)


def full_covariance(q):
    U=q.factors.reshape(q.rank,-1)
    return U.T@U/q.rank


@jax.custom_jvp
def relu_shared(q):
    y,terms=L.relu_terms(Moments(q.mean,q.variance));positive,s,P,*_=terms
    gate=jnp.where(positive,P,(q.mean>0).astype(q.mean.dtype))
    # First Hermite projection; omitted residual is an explicit approximation.
    return Shared(y.mean,q.factors*gate)


@relu_shared.defjvp
def relu_shared_jvp(primals,tangents):
    # s²=mean(U²), ds=mean((U/s)dU), t=m/s. Differentiate the
    # Gaussian integrals BEFORE evaluating: dE[ReLU]=Phi(t)dm+phi(t)ds,
    # dPhi=(phi(t)dm-t phi(t)ds)/s. This avoids an overflowing 1/s³
    # intermediate times an underflowed Gaussian density in float32 tails.
    q,=primals;dq,=tangents;y=relu_shared(q)
    _,(positive,s,P,p,_,_)=L.relu_terms(Moments(q.mean,q.variance))
    t=q.mean/s;tp=jnp.where(jnp.isfinite(t),t,0)*p
    ds=jnp.mean((q.factors/s)*dq.factors,0)
    gate=jnp.where(positive,P,(q.mean>0).astype(q.mean.dtype))
    mean_gate=jnp.where(positive,P,(1+jnp.sign(q.mean))/2)
    dgate=jnp.where(positive,(p*dq.mean-tp*ds)/s,0)
    dm=mean_gate*dq.mean+jnp.where(positive,p*ds,0)
    return y,Shared(dm,gate*dq.factors+q.factors*dgate)


def push(f,parents):
    means=tuple(q.mean for q in parents);factors=tuple(q.factors for q in parents)
    mean=f(*means)
    U=jax.vmap(lambda *us:jax.jvp(f,means,us)[1])(*factors)
    return Shared(mean,U)


def shared_forward(nodes,beliefs,x,key,rank,statistics=None,return_states=False):
    keys=jax.random.split(key,len(beliefs));weights={n:Shared(q.mean,directions(q,k,rank)) for (n,q),k in zip(sorted(beliefs.items()),keys)}
    states={'input':Shared(x,jnp.zeros((rank,)+x.shape,dtype=x.dtype))}
    for node in nodes:
        parents=tuple(states[n] for n in node.parents)+tuple(weights[n] for n in node.parameters)
        if node.kind=='relu':y=relu_shared(parents[0])
        elif node.kind=='maxpool':
            # Tangent of deterministic max, intentionally separate from pairwise Gaussian closure.
            f=lambda x:jax.lax.reduce_window(x,-jnp.inf,jax.lax.max,(1,2,2,1),(1,2,2,1),'VALID')
            y=push(f,parents)
        else:
            mf=local_map(node,statistics)
            f=lambda *values:mf(*(Moments(z,jnp.zeros_like(z)) for z in values)).mean
            y=push(f,parents)
        states[node.name]=y
    return states if return_states else states[nodes[-1].name]


def residual_integral(f,surrogate,gaussian_value,epsilon):
    # Exact control-variate identity E f = E surrogate + E(f-surrogate).
    # Coefficients/proposal are fixed, but parameter dependence through f remains.
    return gaussian_value+jax.vmap(lambda e:f(e)-surrogate(e))(epsilon).mean(0)


def spherical_carrier(plus,minus,key):
    # E[Y]=m, Cov[Y]=mean(odd oddᵀ + even evenᵀ), conditional on the nodes.
    rank=plus.shape[0];mean=((plus+minus)/2).mean(0)
    odd=(plus-minus)/2;even=(plus+minus)/2-mean
    a,b=jax.random.normal(key,(2,rank),dtype=mean.dtype);a=a/jnp.linalg.norm(a);b=b/jnp.linalg.norm(b)
    shape=(rank,)+(1,)*mean.ndim
    return mean+(a.reshape(shape)*odd+b.reshape(shape)*even).sum(0)


def shared_local(node,parents,beliefs,noise,statistics=None,training=False):
    """One closure per layer; the same weight noise IDs are used on every branch."""
    from .gaussian import RootMoments
    if node.kind=='relu':return relu_shared(parents[0])
    if node.kind=='readout':
        x=parents[0];w,b=beliefs;U=x.factors@w.mean
        noise_variance=(x.mean*x.mean+x.variance)@w.variance+b.variance
        # Preserve the Gram factor plus independent diagonal noise. Forming a
        # dense, nearly rank-one C first can erase its small eigenvalues.
        root=jnp.concatenate((U.transpose(1,2,0)/jnp.sqrt(x.rank),jax.vmap(jnp.diag)(jnp.sqrt(noise_variance))),-1)
        return RootMoments(x.mean@w.mean+b.mean,root)
    qs=tuple(Shared(q.mean,jnp.sqrt(q.variance)*e) for q,e in zip(beliefs,noise))
    if node.kind=='norm' and training:
        def f(x,g,b):
            m=x.mean(tuple(range(x.ndim-1)));v=((x-m)**2).mean(tuple(range(x.ndim-1)))
            return (x-m)*jax.lax.rsqrt(v+1e-5)*g+b
    elif node.kind=='maxpool':
        f=lambda x:jax.lax.reduce_window(x,-jnp.inf,jax.lax.max,(1,2,2,1),(1,2,2,1),'VALID')
    else:
        f=lambda *values:local_map(node,statistics)(*(Moments(z,jnp.zeros_like(z)) for z in values)).mean
    return push(f,parents+qs)


def shared_records(nodes,beliefs,x,key,rank,statistics=None,training=False,noise=None):
    # Explicit directions permit the same finite approximation to be compared
    # across frameworks. Different PRNGs are not implementation discrepancies.
    if noise is None:
        keys=jax.random.split(key,len(beliefs))
        noise={n:jax.random.rademacher(k,(rank,)+q.mean.shape,dtype=q.mean.dtype) for (n,q),k in zip(sorted(beliefs.items()),keys)}
    states={'input':Shared(x,jnp.zeros((rank,)+x.shape,dtype=x.dtype))};records=[]
    for node in nodes:
        ps=tuple(states[n] for n in node.parents);ws=tuple(beliefs[n] for n in node.parameters)
        es=tuple(noise[n] for n in node.parameters)
        f=lambda ps,ws,node=node,es=es:shared_local(node,ps,ws,es,statistics,training)
        out=f(ps,ws);states[node.name]=out;records.append((node,ps,ws,f))
    return states,records


def shared_likelihood(nodes,beliefs,x,y,key,rank,statistics=None,method='mfvi',power=1.,training=False,noise=None):
    """Explicit local reverse, including shared-parent/weight accumulation.

    A finite rank factor is not a full-rank Gaussian Fisher coordinate. Its
    Euclidean adjoint is transported directly; persistent weights are encoded
    in Gaussian moment coordinates at the boundary.
    """
    from .gaussian import encode
    from .heads import categorical_nodes,categorical_objective
    states,records=shared_records(nodes,beliefs,x,key,rank,statistics,training,noise)
    objective=lambda q:categorical_objective(categorical_nodes(q),y,method,power)
    value,d=jax.value_and_grad(objective)(states[nodes[-1].name]);pending={nodes[-1].name:d};weight={}
    def accumulate(table,name,value):
        table[name]=value if name not in table else jax.tree_util.tree_map(lambda a,b:a+b,table[name],value)
    for node,ps,ws,f in reversed(records):
        _,pull=jax.vjp(f,ps,ws);dp,dw=pull(pending[node.name])
        for n,g in zip(node.parents,dp):accumulate(pending,n,g)
        for n,g in zip(node.parameters,dw):accumulate(weight,n,g)
    return value,weight,pending,{n:encode(beliefs[n],g) for n,g in weight.items()}


def structured_kl(q,prior_mean,prior_variance):
    # KL(N(m,D+UUᵀ)||N(m0,diag(v0))) with O(P r²+r³), no dense P² matrix.
    trace=jnp.sum((q.diagonal+jnp.sum(q.factors*q.factors,0))/prior_variance)
    quadratic=jnp.sum((q.mean-prior_mean)**2/prior_variance)
    return .5*(trace+quadratic-q.mean.size+jnp.log(prior_variance).sum()-q.logdet())


def gaussian_residual_moments(mean,C,labels,order=32):
    """E[r], E[rr^T] for r_i=sigmoid(z_i)-y_i, z~N(mean,C).

    Two-dimensional Gauss-Hermite integration retains cross-example terms.
    Squaring E[r] would lose the sampled gradient's normalization statistic.
    """
    import numpy as np
    t,w=np.polynomial.hermite.hermgauss(order)
    t=jnp.asarray(t*jnp.sqrt(2.),mean.dtype);w=jnp.asarray(w/jnp.sqrt(jnp.pi),mean.dtype)
    sd=jnp.sqrt(jnp.maximum(jnp.diag(C),0));outer=sd[:,None]*sd[None,:]
    corr=jnp.where(outer>0,C/jnp.maximum(outer,1e-30),0);corr=jnp.clip(corr,-1,1)
    r=jax.nn.sigmoid(mean[:,None]+sd[:,None]*t)-labels[:,None]
    zj=mean[None,:,None,None]+sd[None,:,None,None]*(corr[:,:,None,None]*t[None,None,:,None]+jnp.sqrt(jnp.maximum(1-corr*corr,0))[:,:,None,None]*t[None,None,None,:])
    rj=jax.nn.sigmoid(zj)-labels[None,:,None,None]
    second=jnp.sum(r[:,None,:,None]*rj*w[None,None,:,None]*w[None,None,None,:],axis=(2,3))
    second=(second+second.T)/2;second=second.at[jnp.diag_indices(len(mean))].set((r*r*w).sum(-1))
    return (r*w).sum(-1),second


def frozen_gradient_moments(mean,J,variances,labels,order=32):
    """For fixed logit Jacobian J, E[g], E[g²], not the VI variance derivative.

    J has B×P shape; covariance is only B×B. Shared weight draws induce C.
    The returned quantities may feed an optimizer state, but do not define EP.
    """
    C=(J*variances)@J.T;r,rr=gaussian_residual_moments(mean,C,labels,order);B=len(mean)
    return J.T@r/B,jnp.einsum('bi,bc,ci->i',J,rr,J)/(B*B),C
