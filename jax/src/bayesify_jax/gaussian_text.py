"""The original Gaussian EP/MFVI transformer family, sharing one message graph.

LoRA: independent Gaussian A and B. Full RoBERTa: Gaussian embeddings, affine
maps, normalization parameters, and classifier. Observed token IDs remain fixed.
Adam acts on means/log variances AFTER each complete moment backward snapshot.
The canonical MFVI option uses the decoded likelihood adjoints and one global
prior, not Adam's gradient-square state. EP uses the tied cavity/external field
and is labeled an energy-optimization variant, not literal site replacement.
"""
from dataclasses import replace
import jax,jax.numpy as j
from . import transformers as T,inference as I,heads,covariance
from .gaussian import Moments,mfvi_update,encode
from .training import adam_recurrence


def initialize(cfg,base,seed=41,full_weight=False,optimizer='adam',N=1,temperature=1.,likelihood_normalization=False):
    cfg=replace(cfg,family='full' if full_weight else 'ab')
    q=T.full_weight_beliefs(base) if full_weight else T.initial_beliefs(cfg,seed)
    # Full-weight prior is centered at the supplied pretrained checkpoint.
    prior={n:Moments(p.mean if full_weight else j.zeros_like(p.mean),j.full_like(p.variance,1e-6 if full_weight else 1/2500)) for n,p in q.items()}
    zero={n:j.zeros_like(p.mean) for n,p in q.items()}
    opt=I.vadam_init(q,prior,N,temperature,likelihood_normalization) if optimizer=='vadam' else (j.array(0),zero,zero,zero,zero)
    return cfg,prior,{'belief':q,'optimizer':opt,'key':jax.random.PRNGKey(seed),'updates':j.array(0)}


def frozen_statistics(cfg,base,targets,q,ids,mask,labels,key,rank=4):
    """The R frozen-sensitivity closures; J is refreshed at every update.

    Binary: integrate E[r] and E[rr'] under the full minibatch Gaussian,
    then g=J'E[r]/B and E[g²]=diag(J'E[rr']J)/B².
    Multiclass: g=J' E[softmax(z)-y]/B and a shared-direction GGN correction
    sum_r (J' H_loss U_r/sqrt(R))². Neither is the VI variance derivative.
    """
    means={n:p.mean for n,p in q.items()};B=len(labels)
    f=lambda w:T.point(base,w,ids,mask,cfg,targets)
    if cfg.classes==2:
        scalar,J=T.scalar_moments(base,q,ids,mask,cfg,targets);m=scalar.mean
        names=tuple(sorted(q));flat=j.concatenate([J[n].reshape(B,-1) for n in names],1)
        v=j.concatenate([q[n].variance.reshape(-1) for n in names])
        gm,g2,C=covariance.frozen_gradient_moments(m,flat,v,labels,order=20)
        first={};second={};offset=0
        for n in names:
            count=q[n].mean.size;first[n]=gm[offset:offset+count].reshape(q[n].mean.shape);second[n]=g2[offset:offset+count].reshape(q[n].mean.shape);offset+=count
        return first,second
    out=T.tangent(base,q,ids,mask,cfg,targets,key,rank)
    nodes=j.concatenate((out.mean[None]+out.factors,out.mean[None]-out.factors),0)
    adj=(jax.nn.softmax(nodes,-1).mean(0)-jax.nn.one_hot(labels,cfg.classes,dtype=out.mean.dtype))/B
    _,pull=jax.vjp(f,means);first=pull(adj)[0];p=jax.nn.softmax(out.mean,-1)
    variation=p*(out.factors-(p*out.factors).sum(-1,keepdims=True))/(B*j.sqrt(rank))
    directions=jax.vmap(lambda a:pull(a)[0])(variation)
    return first,{n:first[n]**2+(directions[n]**2).sum(0) for n in q}


def make_step(cfg,base,targets,prior,N,method='mfvi',mode='moment',rank=4,samples=4,learning_rate=1e-4,temperature=1.,optimizer='adam',damping=.01):
    if optimizer in ('canonical','vadam') and method!='mfvi':raise ValueError('This MFVI update is not an EP site update.')
    def step(state,ids,mask,labels):
        q=state['belief'];key,k=jax.random.split(state['key']);B=len(ids)
        if optimizer=='vadam':
            point_loss=lambda w:-heads.categorical_objective(T.point(base,w,ids,mask,cfg,targets)[None],labels)/B
            if mode=='sampled':
                losses,g=jax.vmap(jax.value_and_grad(point_loss))(I.sample_weights(q,k,samples))
                gm={n:a.mean(0) for n,a in g.items()};g2={n:(a*a).mean(0) for n,a in g.items()};loss=losses.mean()
            else:
                gm,g2=frozen_statistics(cfg,base,targets,q,ids,mask,labels,k,rank)
                value,_,_,_=T.moment_likelihood(base,q,ids,mask,cfg,targets,k,labels,rank,'mfvi');loss=-value/B
            new,opt=I.vadam_step(q,prior,gm,g2,state['optimizer'],N,temperature,learning_rate)
            return {'belief':new,'optimizer':opt,'key':key,'updates':state['updates']+1},loss
        if method=='ep':(cavity,A),pull=jax.vjp(lambda p:I.power_cavity(p,prior,N,temperature),q)
        else:cavity=q
        if mode=='moment' and cfg.classes==2:value,g=T.scalar_likelihood(base,cavity,ids,mask,cfg,targets,labels,method)
        elif mode=='moment':value,g,_,_=T.moment_likelihood(base,cavity,ids,mask,cfg,targets,k,labels,rank,method)
        else:
            objective=lambda p:heads.categorical_objective(T.sampled(base,p,ids,mask,cfg,targets,k,samples),labels,method)
            value,g=jax.value_and_grad(objective)(cavity)
        if optimizer=='canonical':
            # Maximize scaled likelihood / temperature - KL. Independent of
            # the sampled-gradient normalization state used by Vadam.
            like=jax.tree_util.tree_map(lambda a:a*N/(B*temperature),g)
            new={n:mfvi_update(q[n],encode(q[n],like[n]),prior[n],damping) for n in q};opt=state['optimizer'];loss=-value/B
        else:
            loss=-value/B;g=jax.tree_util.tree_map(lambda a:-a/B,g)
            if method=='ep':loss=loss-A;g=pull((g,j.asarray(-1.,dtype=A.dtype)))[0]
            else:
                K,gK=jax.value_and_grad(lambda p:I.kl(p,prior))(q);loss+=temperature/N*K
                g=jax.tree_util.tree_map(lambda a,b:a+temperature/N*b,g,gK)
            new,opt=adam_recurrence(q,g,state['optimizer'],learning_rate,learning_rate)
        return {'belief':new,'optimizer':opt,'key':key,'updates':state['updates']+1},loss
    return jax.jit(step)


def predictor(cfg,base,targets,mode='moment',rank=4,samples=32):
    def predict(q,ids,mask,key):
        if mode=='moment' and cfg.classes==2:return heads.binary_gaussian(T.scalar_moments(base,q,ids,mask,cfg,targets)[0])
        if mode=='moment':return T.gaussian_prediction(T.tangent(base,q,ids,mask,cfg,targets,key,rank))
        # Sequential sampling can replace this vmap when memory is constrained.
        return jax.nn.softmax(T.sampled(base,q,ids,mask,cfg,targets,key,samples),-1).mean(0)
    return jax.jit(predict)
