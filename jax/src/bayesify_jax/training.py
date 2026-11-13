"""Complete image training with immutable snapshots and explicit local messages.

A compiled update contains forward moments, local reverse messages, and the
finite optimizer recurrence. A checkpoint contains the belief, optimizer,
curvature cache, normalization state, random key and completed-update count.
There is no posterior oracle. Pre-run compilation and evaluation are recorded
separately; any remaining first-use compilation is included in training time.
"""
from dataclasses import dataclass,asdict
from pathlib import Path
import json,time,hashlib
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import serialization
from .gaussian import Moments
from . import inference as I,models,heads,messages,covariance,data
from .uncertainty import uncertainty_metrics,ood_metrics

@dataclass(frozen=True)
class Protocol:
    architecture:str='mlp'
    method:str='moment_mfvi'
    seed:int=41
    epochs:int=40
    batch:int=128
    learning_rate:float=.001
    variance_rate:float=.001
    temperature:float=.01
    samples:int=1
    prediction_samples:int=20
    optimizer:str='vadam'
    weight_decay:float=0.
    rank:int=1
    curvature_refresh:int=4
    correlated_readout:bool=False
    limit_train:int=0
    limit_validation:int=0
    dataset:str="clean"


def protocol(architecture,method,seed=41):
    image=architecture in ('resnet20','densenet40');point=method=='ordinary';mfvi='mfvi' in method
    lr=.003 if image else (3e-4 if architecture=='mlp' and method in ('ordinary','sampled_ep') else .001)
    return Protocol(architecture,method,seed,100 if image else (40 if architecture=='mlp' else 30),125 if image else 128,
        lr,.003 if method=='sampled_ep' and architecture=='mlp' else lr,
        (.01 if method=='moment_ep' else .001) if image else .01,
        4 if method=='sampled_ep' else 1,20,'adamw' if point else ('vadam' if mfvi else 'adam'),
        .005 if image and point else 0.,1,1 if image else (4 if architecture=='mlp' else 2),
        architecture=='mlp' and method=='moment_ep')


def initialize(cfg,N):
    model=getattr(models,cfg.architecture)(seed=cfg.seed)
    if cfg.correlated_readout:model=models.correlated_mlp(model)
    q=model.snapshot();prior=model.priors();stats=model.statistics()
    if cfg.correlated_readout:stats['readout']=models.readout_basis(q)
    mean={n:p.mean for n,p in q.items()};zero=jax.tree_util.tree_map(jnp.zeros_like,mean)
    opt=I.vadam_init(q,prior,N,cfg.temperature) if cfg.optimizer=='vadam' else (
        jnp.array(0),zero,zero,zero,zero)
    state={'belief':q,'statistics':stats,'optimizer':opt,'curvature':zero,'key':jax.random.PRNGKey(cfg.seed),'updates':jnp.array(0)}
    # Fixed probes are part of the declared quadrature, not refreshed MC networks.
    keys=jax.random.split(jax.random.PRNGKey(cfg.seed+703),len(q))
    draw=jax.random.rademacher if cfg.architecture=='lenet' else jax.random.normal
    probes={n:draw(k,p.mean.shape,dtype=p.mean.dtype) for (n,p),k in zip(sorted(q.items()),keys)}
    return model,prior,state,probes


def adam_recurrence(q,g,state,lr,vlr,decay=0.,point=False,beta2=.999):
    t,r,s,rv,sv=state;t=t+1;new={};nr={};ns={};nrv={};nsv={}
    for n,p in q.items():
        a=g[n].mean;nr[n]=.9*r[n]+.1*a;ns[n]=beta2*s[n]+(1-beta2)*a*a
        mean=p.mean*(1-lr*decay)-lr*(nr[n]/(1-.9**t))/(jnp.sqrt(ns[n]/(1-beta2**t))+1e-8)
        b=p.variance*g[n].variance;nrv[n]=.9*rv[n]+.1*b;nsv[n]=beta2*sv[n]+(1-beta2)*b*b
        logv=jnp.log(p.variance)-vlr*(nrv[n]/(1-.9**t))/(jnp.sqrt(nsv[n]/(1-beta2**t))+1e-8)
        new[n]=Moments(mean,p.variance if point else jnp.exp(logv))
    return new,(t,nr,ns,nrv,nsv)


def make_step(model,prior,N,cfg,probes):
    nodes=model.nodes;point=cfg.method=='ordinary';method='ep' if 'ep' in cfg.method else 'mfvi'
    mode='sampled' if cfg.method.startswith('sampled') else 'moment'
    shared=cfg.architecture in ('resnet20','densenet40');direction_key=jax.random.PRNGKey(cfg.seed+7919)
    def likelihood(q,x,y,stats):
        if shared:return covariance.shared_likelihood(nodes,q,x,y,direction_key,cfg.rank,stats,method,training=True)[:2]
        return I.moment_likelihood(nodes,q,x,y,stats,method)[:2]
    def point_loss(w,x,y,stats):
        z,updated=messages.point_forward(nodes,w,x,stats,training=shared)
        return -heads.categorical_objective(z[None],y)/len(y),updated
    def step(state,x,y,rate):
        q=state['belief'];stats=state['statistics'];key,k=jax.random.split(state['key']);weights={n:p.mean for n,p in q.items()}
        cache=state['curvature'];newstats={}
        if point:
            (loss,newstats),g=jax.value_and_grad(point_loss,has_aux=True)(weights,x,y,stats)
            grad={n:Moments(a,jnp.zeros_like(a)) for n,a in g.items()}
        elif method=='mfvi' and cfg.optimizer=='vadam':
            if mode=='sampled':
                ws=I.sample_weights(q,k,cfg.samples)
                (losses,ns),grads=jax.vmap(jax.value_and_grad(point_loss,has_aux=True),in_axes=(0,None,None,None))(ws,x,y,stats)
                loss=losses.mean();g={n:a.mean(0) for n,a in grads.items()};g2={n:(a*a).mean(0) for n,a in grads.items()}
                newstats=jax.tree_util.tree_map(lambda a:a.mean(0),ns)
            else:
                value,gradient=likelihood(q,x,y,stats);loss=-value/len(y);g={n:-a.mean/len(y) for n,a in gradient.items()}
                def curvature(_):
                    d={n:jnp.sqrt(p.variance)*probes[n] for n,p in q.items()}
                    hvp=jax.jvp(jax.grad(lambda w:point_loss(w,x,y,stats)[0]),(weights,),(d,))[1]
                    return jax.tree_util.tree_map(jnp.square,hvp)
                cache=jax.lax.cond(state['updates']%cfg.curvature_refresh==0,curvature,lambda _:cache,operand=None)
                g2={n:a*a+cache[n] for n,a in g.items()}
            q,opt=I.vadam_step(q,prior,g,g2,state['optimizer'],N,cfg.temperature,cfg.learning_rate*rate)
        else:
            if method=='ep':(cavity,A),pull=jax.vjp(lambda q:I.power_cavity(q,prior,N,cfg.temperature),q)
            else:cavity=q;A=0.
            if mode=='moment':value,grad=likelihood(cavity,x,y,stats);grad=jax.tree_util.tree_map(lambda a:-a/len(y),grad);loss=-value/len(y)
            else:
                def sample_loss(q):
                    ws=I.sample_weights(q,k,cfg.samples)
                    logits,ns=jax.vmap(lambda w:messages.point_forward(nodes,w,x,stats,shared))(ws)
                    return -heads.categorical_objective(logits,y,method)/len(y),ns
                (loss,ns),grad=jax.value_and_grad(sample_loss,has_aux=True)(cavity)
                newstats=jax.tree_util.tree_map(lambda a:a.mean(0),ns)
            if method=='ep':loss=loss-A;grad=pull((grad,jnp.asarray(-1.,dtype=A.dtype)))[0]
            else:
                K,dK=jax.value_and_grad(lambda q:I.kl(q,prior))(q);loss+=cfg.temperature/N*K
                grad=jax.tree_util.tree_map(lambda a,b:a+cfg.temperature/N*b,grad,dK)
        if cfg.optimizer!='vadam':
            q,opt=adam_recurrence(q,grad,state['optimizer'],cfg.learning_rate*rate,cfg.variance_rate*rate,cfg.weight_decay,point,
                                 .99 if point and not shared else .999)
        if shared:
            if mode=='moment' and not point:
                states,_=covariance.shared_records(nodes,cavity if method=='ep' else state['belief'],x,direction_key,cfg.rank,stats,training=True)
                for node in nodes:
                    if node.kind=='norm':
                        h=states[node.parents[0]].mean;axes=tuple(range(h.ndim-1));newstats[node.name]=(h.mean(axes),h.var(axes)*(h.size/h.shape[-1])/(h.size/h.shape[-1]-1))
            stats={n:(.9*stats[n][0]+.1*m,.9*stats[n][1]+.1*v) for n,(m,v) in newstats.items()}
        return {'belief':q,'statistics':stats,'optimizer':opt,'curvature':cache,'key':key,'updates':state['updates']+1},loss
    return jax.jit(step)


def make_predictor(model,cfg,mode=None,samples=None):
    mode=mode or ('point' if cfg.method=='ordinary' else ('sampled' if cfg.method.startswith('sampled') else 'moment'))
    samples=samples or cfg.prediction_samples
    def predict(state,x,key):
        if mode=='moment' and cfg.architecture in ('resnet20','densenet40'):
            states,_=covariance.shared_records(model.nodes,state['belief'],x,jax.random.PRNGKey(cfg.seed+7919),cfg.rank,state['statistics'])
            return heads.probabilities(states[model.nodes[-1].name])
        return I.predict(model.nodes,state['belief'],x,key,samples,state['statistics'],mode)
    return jax.jit(predict)


def predict_array(predict,state,x,batch=256,seed=192):
    out=[];key=jax.random.PRNGKey(seed)
    for i in range(0,len(x),batch):
        key,k=jax.random.split(key);out.append(np.asarray(predict(state,jnp.asarray(x[i:i+batch]),k)))
    return np.concatenate(out)


def save_checkpoint(path,state,epoch):
    payload=serialization.to_bytes({'state':jax.device_get(state),'epoch':epoch})
    path=Path(path);temporary=path.with_suffix('.partial');temporary.write_bytes(payload);temporary.replace(path)


def train(cfg,root='data',output='results'):
    folder=Path(output)/(f'{cfg.architecture}_{cfg.method}_seed{cfg.seed}'+('_dirty' if cfg.dataset=='dirty' else ''));folder.mkdir(parents=True,exist_ok=True)
    (folder/'config.json').write_text(json.dumps(asdict(cfg),indent=2))
    ds=(data.dirty_mnist(root) if cfg.dataset=='dirty' else data.mnist(root)) if cfg.architecture in ('mlp','lenet') else data.cifar10(root)
    x,y=ds['train'];xv,yv=ds['validation']
    if cfg.limit_train:x,y=x[:cfg.limit_train],y[:cfg.limit_train]
    if cfg.limit_validation:xv,yv=xv[:cfg.limit_validation],yv[:cfg.limit_validation]
    model,prior,state,probes=initialize(cfg,len(y));step=make_step(model,prior,len(y),cfg,probes);prediction=make_predictor(model,cfg)
    @jax.jit
    def checked_step(state,x,y,rate):
        updated,loss=step(state,x,y,rate)
        finite=jnp.isfinite(loss)&jnp.all(jnp.stack([jnp.isfinite(a).all() for a in jax.tree_util.tree_leaves(updated)]))
        return updated,loss,finite
    history=[];start_epoch=0;train_seconds=0.;best=float('inf');checkpoint=folder/'recovery.msgpack'
    if checkpoint.exists():
        saved=serialization.from_bytes({'state':state,'epoch':0},checkpoint.read_bytes());state=saved['state'];start_epoch=int(saved['epoch'])
        history=json.loads((folder/'curves.json').read_text());train_seconds=history[-1]['training_seconds'];best=min(row['validation_nll'] for row in history)
    begin=time.perf_counter()
    for size in sorted(set([cfg.batch,len(y)%cfg.batch])-{0}):
        compiled=checked_step(state,jnp.asarray(x[:size]),jnp.asarray(y[:size]),jnp.float32(1.));jax.block_until_ready(compiled)
    for size in sorted(set([min(256,len(yv)),len(yv)%256])-{0}):
        prediction(state,jnp.asarray(xv[:size]),jax.random.PRNGKey(192)).block_until_ready()
    compile_seconds=time.perf_counter()-begin
    print('COMPILED',folder.name,round(compile_seconds,2),flush=True)
    for epoch in range(start_epoch,cfg.epochs):
        if cfg.correlated_readout:state['statistics']['readout']=models.readout_basis(state['belief'])
        order=np.random.default_rng(cfg.seed*1000+epoch+1).permutation(len(y));losses=[];begin=time.perf_counter()
        rate=jnp.float32((.03 if cfg.architecture in ('mlp','lenet') else 0.)+( .97 if cfg.architecture in ('mlp','lenet') else 1.)*.5*(1+np.cos(np.pi*epoch/max(1,cfg.epochs-1))))
        for i in range(0,len(y),cfg.batch):
            ids=order[i:i+cfg.batch];xx=x[ids]
            if cfg.architecture in ('resnet20','densenet40'):xx=augment(xx,np.random.default_rng(cfg.seed*100000+epoch*1000+i))
            before=state;state,loss,finite=checked_step(state,jnp.asarray(xx),jnp.asarray(y[ids]),rate)
            if not bool(finite):
                save_checkpoint(folder/'diagnostic_before.msgpack',before,epoch)
                save_checkpoint(folder/'failed.msgpack',state,epoch+1)
                np.savez_compressed(folder/'failed_minibatch.npz',x=xx,y=y[ids],epoch=epoch+1,offset=i,rate=float(rate))
                raise FloatingPointError(f'First nonfinite update: epoch {epoch+1}, batch {i//cfg.batch}; pre-update belief and batch preserved')
            losses.append((loss,len(ids)))
        jax.block_until_ready(state);train_seconds+=time.perf_counter()-begin
        loss=sum(float(l)*n for l,n in losses)/len(y)
        if not np.isfinite(loss):
            save_checkpoint(folder/'failed.msgpack',state,epoch+1);raise FloatingPointError(f'Nonfinite training objective at epoch {epoch+1}')
        begin=time.perf_counter();p=predict_array(prediction,state,xv);evaluation_seconds=time.perf_counter()-begin
        if not np.isfinite(p).all():
            save_checkpoint(folder/'failed_prediction.msgpack',state,epoch+1)
            np.savez_compressed(folder/'failed_prediction.npz',probabilities=p,labels=yv)
            raise FloatingPointError(f'Nonfinite predictive probabilities at epoch {epoch+1}; belief preserved')
        scores=uncertainty_metrics(p,yv)
        row={'epoch':epoch+1,'training_objective':loss,'training_seconds':train_seconds,'evaluation_seconds':evaluation_seconds,'learning_rate':cfg.learning_rate*float(rate),**{'validation_'+k:v for k,v in scores.items()}}
        history.append(row);(folder/'curves.json').write_text(json.dumps(history,indent=2))
        save_checkpoint(checkpoint,state,epoch+1)
        if scores['nll']<best:
            best=scores['nll'];save_checkpoint(folder/'best.msgpack',state,epoch+1);np.savez_compressed(folder/'validation_predictions.npz',probabilities=p,labels=yv)
        print('EPOCH',folder.name,epoch+1,'NLL',round(scores['nll'],5),'accuracy',round(scores['accuracy'],5),'train_s',round(train_seconds,1),flush=True)
    receipt={'framework':'jax','versions':{'jax':jax.__version__},'devices':[str(x) for x in jax.devices()],
        'completed_epochs':cfg.epochs,'training_seconds':train_seconds,'compile_seconds':compile_seconds,'status':'subset_execution' if cfg.limit_train else 'full_fixed_budget',
        'observations':len(y),'parameters':models.parameter_count(model),'split_sha256':hashlib.sha256(np.asarray(y).tobytes()).hexdigest()}
    (folder/'receipt.json').write_text(json.dumps(receipt,indent=2))
    best_state=serialization.from_bytes({'state':state,'epoch':0},(folder/'best.msgpack').read_bytes())
    selected=best_state['state'];xt,yt=ds['test']
    if cfg.limit_validation:xt,yt=xt[:cfg.limit_validation],yt[:cfg.limit_validation]
    # Compile both batch shapes outside inference timing.
    for size in sorted(set([min(256,len(yt)),len(yt)%256])-{0}):prediction(selected,jnp.asarray(xt[:size]),jax.random.PRNGKey(192)).block_until_ready()
    begin=time.perf_counter();p=predict_array(prediction,selected,xt);seconds=time.perf_counter()-begin
    scores=uncertainty_metrics(p,yt);scores.update(inference_seconds=seconds,selected_epoch=int(best_state['epoch']),training_seconds=train_seconds)
    np.savez_compressed(folder/'test_predictions.npz',probabilities=p,labels=yt)
    if cfg.architecture in ('mlp','lenet'):
        xo,_=data.fashion(root)
        if cfg.limit_validation:xo=xo[:cfg.limit_validation]
        po=predict_array(prediction,selected,xo);scores.update(ood_metrics(p,po));np.savez_compressed(folder/'ood_predictions.npz',probabilities=po)
        if cfg.dataset=='dirty':
            # Ten observed labels per ambiguous image enter proper scores;
            # count each image once for ID/OOD rankings.
            xa,ya=ds['ambiguous_test']
            if cfg.limit_validation:xa,ya=xa[:cfg.limit_validation],ya[:cfg.limit_validation]
            pa=predict_array(prediction,selected,xa)
            np.savez_compressed(folder/'ambiguous_predictions.npz',probabilities=pa,labels=ya)
            combined=uncertainty_metrics(np.concatenate((p,np.repeat(pa,10,axis=0))),np.concatenate((yt,ya.ravel())))
            combined.update(ood_metrics(np.concatenate((p,pa)),po))
            scores.update(combined)
    (folder/'metrics.json').write_text(json.dumps(scores,indent=2));return history


def augment(x,rng):
    padded=np.pad(x,((0,0),(4,4),(4,4),(0,0)),mode='constant');offset=rng.integers(0,9,(len(x),2));result=np.empty_like(x)
    for i,(a,b) in enumerate(offset):result[i]=padded[i,a:a+32,b:b+32]
    flip=rng.random(len(x))<.5;result[flip]=result[flip,:,::-1];return result
