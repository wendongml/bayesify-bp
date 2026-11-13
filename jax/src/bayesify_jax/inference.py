"""Finite inference updates. EP sites, VI factors and optimizer state are distinct.

Heads maximize a log likelihood/normalizer. Training routines minimize its
negative, with the KL/external field counted once globally. All reverse messages
use the same forward snapshot; no belief is mutated during differentiation.
"""
import jax
import jax.numpy as jnp
import optax
from .gaussian import Moments,FullMoments,encode_full,decode,kl_diagonal,mfvi_update
from .heads import categorical_head,categorical_objective,categorical_nodes,probabilities
from .messages import forward,backward,point_forward
from .gaussian_chain import evidence


def power_cavity(beliefs,priors,observations,alpha):
    k=alpha/observations;cavity={};logA=0.
    for n,q in beliefs.items():
        p=priors[n];e=k*(q.variance/p.variance-1);d=1+e
        cavity[n]=Moments(p.mean+(1-k)*(q.mean-p.mean)/d,q.variance/d)
        logA-=.5*jnp.sum(jnp.log1p(e)-k*jnp.log(q.variance/p.variance)+k*(1-k)*(q.mean-p.mean)**2/(p.variance*d))
    return cavity,logA


def kl(beliefs,priors):return sum(kl_diagonal(q,priors[n]) for n,q in beliefs.items())


def moment_likelihood(nodes,beliefs,x,y,statistics=None,method='mfvi',power=1.,literal=False):
    states,records=forward(nodes,beliefs,Moments(x,jnp.zeros_like(x)),statistics)
    value,d=categorical_head(states[nodes[-1].name],y,method,power)
    _,activations,signals=backward(records,d,statistics,literal)
    gradients={n:decode(q,signals[n]) for n,q in beliefs.items()}
    return value,gradients,activations,signals


def sample_weights(q,key,samples):
    keys=jax.random.split(key,len(q))
    return {n:p.mean+jnp.sqrt(p.variance)*jax.random.normal(k,(samples,)+p.mean.shape,dtype=p.mean.dtype) for (n,p),k in zip(sorted(q.items()),keys)}


def sampled_objective(nodes,beliefs,x,y,key,samples,statistics,method,power):
    weights=sample_weights(beliefs,key,samples)
    logits=jax.vmap(lambda w:point_forward(nodes,w,x,statistics)[0])(weights)
    return categorical_objective(logits,y,method,power)


def objective_and_grad(nodes,beliefs,priors,x,y,key,observations,temperature=1.,
                       method='mfvi',mode='moment',samples=1,power=1.,statistics=None,literal=False):
    """Actual moment reverse, or complete-network MC, then the declared field.

    MFVI: -mean E log p + (tau/N) KL. EP tied energy:
    -mean log E_c p^power /power - log A /power, k=tau*power/N.
    The tied energy is explicitly an EP-style approximation for nonlinear models.
    """
    if method=='ep':
        (cavity,logA),pull=jax.vjp(lambda q:power_cavity(q,priors,observations,temperature*power),beliefs)
    else:cavity=beliefs;logA=0.
    if mode=='moment':
        value,g,_,_=moment_likelihood(nodes,cavity,x,y,statistics,method,power,literal)
    elif mode=='sampled':
        value,g=jax.value_and_grad(lambda q:sampled_objective(nodes,q,x,y,key,samples,statistics,method,power))(cavity)
    else:raise ValueError(mode)
    g=jax.tree_util.tree_map(lambda a:-a/x.shape[0],g)
    if method=='ep':
        result=pull((g,jnp.asarray(-1/power,dtype=value.dtype)))[0]
        return -value/x.shape[0]-logA/power,result
    penalty,gkl=jax.value_and_grad(lambda q:kl(q,priors))(beliefs)
    result=jax.tree_util.tree_map(lambda a,b:a+temperature/observations*b,g,gkl)
    return -value/x.shape[0]+temperature/observations*penalty,result


def adam_parameters(beliefs):return {n:Moments(q.mean,jnp.log(q.variance)) for n,q in beliefs.items()}
def from_adam(parameters):return {n:Moments(q.mean,jnp.exp(q.variance)) for n,q in parameters.items()}


def adam_step(beliefs,gradient,state,optimizer):
    # Variance parameterization a=log v => grad_a L = v grad_v L.
    p=adam_parameters(beliefs);g={n:Moments(gradient[n].mean,q.variance*gradient[n].variance) for n,q in beliefs.items()}
    updates,state=optimizer.update(g,state,p)
    return from_adam(optax.apply_updates(p,updates)),state


def vadam_init(beliefs,priors,N,tau,likelihood_normalization=False):
    Neff=N/tau
    state=(jnp.array(0),{n:jnp.zeros_like(q.mean) for n,q in beliefs.items()},
            {n:(1/q.variance-1/priors[n].variance)/Neff for n,q in beliefs.items()})
    # Some source transformer runs normalize means with a separate likelihood
    # EMA initialized at zero, while retaining the variance-implied curvature.
    return state+({n:jnp.zeros_like(q.mean) for n,q in beliefs.items()},) if likelihood_normalization else state


def vadam_step(beliefs,priors,g,g2,state,N,tau,lr,betas=(.9,.99),decay=0.):
    # Variance is inverse curvature; EMA of E[g²] is not (E[g])².
    step,r,s=state[:3];step=step+1;b1,b2=betas;Neff=N/tau;new={};nr={};ns={};nu={}
    for n,q in beliefs.items():
        p=priors[n];lam=1/(p.variance*Neff)
        nr[n]=b1*r[n]+(1-b1)*(g[n]+lam*(q.mean-p.mean))
        ns[n]=b2*s[n]+(1-b2)*g2[n]
        normalization=ns[n]
        if len(state)==4:
            nu[n]=b2*state[3][n]+(1-b2)*g2[n];normalization=nu[n]
        m=q.mean*(1-lr*decay)-lr*(nr[n]/(1-b1**step))/(jnp.sqrt(normalization/(1-b2**step))+lam)
        new[n]=Moments(m,1/(Neff*ns[n]+1/p.variance))
    return new,((step,nr,ns,nu) if len(state)==4 else (step,nr,ns))


def gradient_second(nodes,beliefs,x,y,key,statistics=None,samples=1,mode='sampled'):
    # Actual per-draw gradient moments for MC; delta-method H V Hᵀ for moments.
    loss=lambda w:-categorical_objective(point_forward(nodes,w,x,statistics)[0][None],y)/len(y)
    if mode=='sampled':
        gs=jax.vmap(jax.grad(loss))(sample_weights(beliefs,key,samples))
        return ({n:a.mean(0) for n,a in gs.items()},{n:(a*a).mean(0) for n,a in gs.items()})
    _,g,_,_=moment_likelihood(nodes,beliefs,x,y,statistics,'mfvi')
    means={n:q.mean for n,q in beliefs.items()};keys=jax.random.split(key,len(beliefs))
    direction={n:jnp.sqrt(q.variance)*jax.random.rademacher(k,q.mean.shape,dtype=q.mean.dtype) for (n,q),k in zip(sorted(beliefs.items()),keys)}
    _,hvp=jax.jvp(jax.grad(loss),(means,),(direction,))
    gm={n:-a.mean/len(y) for n,a in g.items()}
    return gm,{n:gm[n]**2+hvp[n]**2 for n in gm}


def predict(nodes,beliefs,x,key=None,samples=32,statistics=None,mode='moment'):
    if mode=='moment':
        out=forward(nodes,beliefs,Moments(x,jnp.zeros_like(x)),statistics)[0][nodes[-1].name]
        return probabilities(out)
    if mode=='point':return jax.nn.softmax(point_forward(nodes,{n:q.mean for n,q in beliefs.items()},x,statistics)[0],-1)
    weights=sample_weights(beliefs,key,samples)
    return jax.vmap(lambda w:jax.nn.softmax(point_forward(nodes,w,x,statistics)[0],-1))(weights).mean(0)


def gaussian_tilt(q,H,y,R):
    # Log Z derivatives -> finite raw-moment update, no conditioning oracle.
    adj=jax.grad(evidence)(q,H,y,R);dm,dM=encode_full(q,adj);m=q.mean+dm
    C=q.covariance+dM-jnp.outer(q.mean,dm)-jnp.outer(dm,q.mean)-jnp.outer(dm,dm)
    return FullMoments(m,(C+C.T)/2)


def ep_replace(global_information,site,H,y,R,diagonal=False,damping=1.):
    precision,information=global_information;siteP,siteh=site
    Pc,hc=precision-siteP,information-siteh;Cc=jnp.linalg.solve(Pc,jnp.eye(len(hc)));mc=Cc@hc
    tilted=gaussian_tilt(FullMoments(mc,Cc),H,y,R)
    Ct=jnp.diag(jnp.diag(tilted.covariance)) if diagonal else tilted.covariance
    Pt=jnp.linalg.solve(Ct,jnp.eye(len(hc)));ht=Pt@tilted.mean
    nextP=(1-damping)*siteP+damping*(Pt-Pc);nexth=(1-damping)*siteh+damping*(ht-hc)
    return (Pc+nextP,hc+nexth),(nextP,nexth),tilted
