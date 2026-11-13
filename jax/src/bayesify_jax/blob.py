"""BLoB conditional-Gaussian LoRA and its source-coupled moment mirror.

A = mu + rho² * epsilon, B is a point parameter. This is a variational
Gaussian-A model, not EP. The local 2K-node moment rule is a stochastic hybrid.
The one-node control reproduces the reference output AND derivatives exactly
for common primitive randomness. Larger K changes higher-order moments.

Reference: Wang-ML-Lab/bayesian-peft, commit 7d960082, modelwrappers/blob.py.
The split update follows its actual sequence, including the post-Adam KL
mean correction. Optimizer choices do not change the local moment identity.
"""
import math
import jax
import jax.numpy as jnp
import optax
from . import transformers as T


def coupled_nodes(xi,a,key):
    # Xi_i=a_i xi+Z_i-a_i sum_j a_j Z_j; Cov(Xi_i,Xi_j)=delta_ij I.
    Z=jax.random.normal(key,(len(a),)+xi.shape,dtype=xi.dtype)
    av=a.reshape((len(a),)+(1,)*xi.ndim)
    return av*xi+Z-av*(av*Z).sum(0)


def transition(h,base,means,rho,mask,cfg,index,key,auxiliary,pairs=2,reference=False):
    k1,k2,k3=jax.random.split(auxiliary,3)
    a=jax.random.normal(k1,(pairs,),dtype=h.dtype);a=a/jnp.linalg.norm(a)
    b=jax.random.normal(k2,(pairs,),dtype=h.dtype);b=b/jnp.linalg.norm(b)
    samples={};signs={};prefix=f'block{index}.'
    names=[prefix+n for n in ['query','value']]
    for n,ks,ka in zip(names,jax.random.split(key,len(names)),jax.random.split(k3,len(names))):
        r,t,e=jax.random.split(ks,3);A=means[n+'.A']
        signs[n]=(jax.random.rademacher(r,h.shape,dtype=h.dtype),jax.random.rademacher(t,h.shape[:-1]+(cfg.rank,),dtype=h.dtype))
        xi=jax.random.normal(e,A.shape,dtype=h.dtype)
        if reference:samples[n]=xi[None]
        else:
            z=coupled_nodes(xi,a,ka);samples[n]=jnp.concatenate((z,-z),0)
    def one(noise):
        def linear(x,name):
            y=T.affine(x,base,name,means,cfg.alpha/cfg.rank)
            if name in noise:
                r,t=signs[name]
                y=y+cfg.alpha/cfg.rank*((((x*r)@(noise[name]*rho[name+'.A']**2))*t)@means[name+'.B'])
            return y
        return T.block(h,mask,base,means,cfg,index,linear_fn=linear)
    nodes=jax.vmap(one)(samples)
    if reference:return nodes[0]
    plus,minus=nodes[:pairs],nodes[pairs:];mean=((plus+minus)/2).mean(0)
    odd=(plus-minus)/2;even=(plus+minus)/2-mean;shape=(pairs,)+(1,)*mean.ndim
    return mean+(a.reshape(shape)*odd+b.reshape(shape)*even).sum(0)


def forward(base,means,rho,ids,mask,cfg,target_ids,key,auxiliary,pairs=2,reference=False,rematerialize=True):
    # LoRA randomness is block-local; PRNG inputs make checkpoint replay exact.
    h=T.embeddings(base,ids,mask,cfg);keys=jax.random.split(key,cfg.layers);aux=jax.random.split(auxiliary,cfg.layers)
    for i in range(cfg.layers):
        f=lambda h,i=i:transition(h,base,means,rho,mask,cfg,i,keys[i],aux[i],pairs,reference)
        h=jax.checkpoint(f)(h) if rematerialize else f(h)
    return T.readout(h,base,means,mask,cfg,target_ids)


def source_kl(means,rho,prior_std=.2,epsilon=1e-6):
    # Source-stabilized Gaussian KL. eps=0 recovers the exact Gaussian formula.
    return sum(jnp.sum(jnp.log(prior_std+epsilon)-jnp.log(r**2+epsilon)+
                      (r**4+means[n]**2)/(2*prior_std**2+epsilon)-.5) for n,r in rho.items())


def cycle_weight(step,cycle):
    i=step%cycle+1
    # Integer powers of two are exact binary scalings. CUDA's approximate
    # exp2 can drift by one float32 unit even here; ldexp avoids that drift.
    one=jnp.float32(1)
    return jnp.ldexp(one,jnp.asarray(i-cycle-1,jnp.int32))/(one-jnp.ldexp(one,jnp.int32(-cycle-1)))


def split_update(means,rho,g_mean,g_rho,state,optimizer,lr,variance_lr,step,cycle,decay=0.,prior_std=.2):
    # 1. AdamW likelihood update of means/point parameters; rho is excluded.
    u,state=optimizer.update(g_mean,state,means)
    next_mean={n:m-lr*u[n]-lr*(decay[n] if isinstance(decay,dict) else decay)*m for n,m in means.items()}
    # 2. Evaluate KL at updated means, retaining rho's OLD likelihood adjoint.
    pi=cycle_weight(step,cycle);value,(dm,dr)=jax.value_and_grad(source_kl,argnums=(0,1))(next_mean,rho,prior_std)
    next_mean=jax.tree_util.tree_map(lambda m,d:m-variance_lr*pi*d,next_mean,dm)
    next_rho={n:r-variance_lr*(g_rho[n]+pi*dr[n]) for n,r in rho.items()}
    return next_mean,next_rho,state,value,pi


def initial_rho(means,key,epsilon=.05):
    names=[n for n in means if n.endswith('.A')];keys=jax.random.split(key,len(names))
    return {n:jax.random.uniform(k,means[n].shape,minval=epsilon/math.sqrt(2),maxval=epsilon,dtype=means[n].dtype) for n,k in zip(names,keys)}
