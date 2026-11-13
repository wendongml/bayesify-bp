"""Exact affine-chain pseudo-backpropagation, independent of the reference."""
import jax
import jax.numpy as jnp
from .gaussian import FullMoments,encode_full,decode_full


def evidence(q,H,y,R):
    S=H@q.covariance@H.T+R;e=y-H@q.mean
    return -.5*(len(y)*jnp.log(2*jnp.pi)+jnp.linalg.slogdet(S)[1]+e@jnp.linalg.solve(S,e))


def candidate(layers,weights,initial,H,y,R):
    states=[initial]
    for (A,B,c),q in zip(layers,weights):
        x=states[-1];states.append(FullMoments(A@x.mean+B@q.mean+c,A@x.covariance@A.T+B@q.covariance@B.T))
    head=jax.grad(evidence)(states[-1],H,y,R);signal=encode_full(states[-1],head)
    activation=[None]*len(states);weight=[None]*len(weights);activation[-1]=signal
    for i in reversed(range(len(layers))):
        A,B,c=layers[i]
        def local(x,w):return FullMoments(A@x.mean+B@w.mean+c,A@x.covariance@A.T+B@w.covariance@B.T)
        _,pull=jax.vjp(local,states[i],weights[i]);gx,gw=pull(FullMoments(*decode_full(states[i+1],signal)))
        weight[i]=encode_full(weights[i],gw);signal=encode_full(states[i],gx);activation[i]=signal
    install=lambda q,d:FullMoments(q.mean+d[0],q.covariance+d[1]-jnp.outer(q.mean,d[0])-jnp.outer(d[0],q.mean)-jnp.outer(d[0],d[0]))
    return states,[install(q,d) for q,d in zip(states,activation)],[install(q,d) for q,d in zip(weights,weight)]


def frozen_mlp(seed=41):
    """64->12->8->10 tanh checkpoint, frozen local weight/input Jacobians.

    States are perturbations: the observation is target-f(checkpoint), not the
    raw one-hot target. Shared parameter blocks contain biases as well as weights.
    """
    import numpy as np
    rng=np.random.default_rng(seed);h=jnp.asarray(rng.normal(size=64));layers=[];weights=[];base_activations=[h]
    for i,(a,b) in enumerate([(64,12),(12,8),(8,10)]):
        theta=jnp.asarray(rng.normal(size=(a+1)*b)*.1)
        def layer(x,w,i=i,a=a,b=b):
            W=w[:a*b].reshape(a,b);bias=w[a*b:];z=x@W+bias
            return jnp.tanh(z) if i<2 else z
        A,B=jax.jacrev(layer,argnums=(0,1))(h,theta);h=layer(h,theta);base_activations.append(h)
        layers.append((A,B,jnp.zeros(b)))
        weights.append(FullMoments(jnp.asarray(rng.normal(size=theta.shape)*.003),jnp.eye(theta.size)*.002))
    initial=FullMoments(jnp.zeros(64),jnp.zeros((64,64)));target=jax.nn.one_hot(3,10)-h
    return layers,weights,initial,jnp.eye(10),target,jnp.eye(10)*.2,base_activations
