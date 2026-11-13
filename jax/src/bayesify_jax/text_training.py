"""The fixed BLoB transfer protocol, ported to JAX without a tuning search.

The mean likelihood step uses AdamW; rho's retained likelihood adjoint and
post-mean KL derivatives use the source SGD step. Gaussian-A/point-B LoRA and
the full point classifier are separate parameter groups. Predictions average
probabilities. 'hybrid' is conditional moment propagation, not deterministic EP.
"""
from pathlib import Path
from dataclasses import replace,asdict
import json,time,math
import numpy as np
import jax
jax.config.update('jax_default_matmul_precision','highest')
import jax.numpy as jnp
import optax
from flax import serialization
from . import transformers as T,blob,data
from .uncertainty import uncertainty_metrics,ood_metrics


def make_state(cfg,base,seed=41):
    means={n:q.mean for n,q in T.initial_beliefs(cfg,seed).items()}
    if cfg.architecture=='roberta':means.update({n:a for n,a in base.items() if n.startswith('classifier.')})
    rho=blob.initial_rho(means,jax.random.PRNGKey(seed+1));optimizer=optax.scale_by_adam(b1=.9,b2=.999,eps=1e-8)
    return {'means':means,'rho':rho,'optimizer':optimizer.init(means),'key':jax.random.PRNGKey(seed),'updates':jnp.array(0)},optimizer


def make_step(cfg,base,target_ids,method,N,epochs,optimizer,batch=32,pairs=2):
    total=epochs*math.ceil(N/batch);warm=max(1,round((.05 if cfg.architecture=='gpt2' else .06)*total));cycle=int(100*N**(math.pi/8)/batch)
    learning_rate=3e-4 if cfg.architecture=='gpt2' else .001
    def step(state,ids,mask,labels):
        key,source,aux=jax.random.split(state['key'],3)
        def objective(means,rho):
            logits=T.point(base,means,ids,mask,cfg,target_ids) if method=='ordinary' else blob.forward(base,means,rho,ids,mask,cfg,target_ids,source,aux,pairs,reference=method=='sampled')
            return -jnp.take_along_axis(jax.nn.log_softmax(logits,-1),labels[:,None],-1).mean()
        value,(gm,gr)=jax.value_and_grad(objective,argnums=(0,1))(state['means'],state['rho'])
        step=state['updates'];t=step+1;u=jnp.maximum(t-warm,0)/max(total-warm,1)
        cool=.1+.9*(1+jnp.cos(jnp.pi*u))/2 if cfg.architecture=='gpt2' else jnp.maximum(0.,1-u)
        rate=jnp.where(t<=warm,t/warm,cool);lr=learning_rate*rate
        # Head weights share the ordinary decay; head biases have none.
        decay={n:(.01 if (method=='ordinary' or n.startswith('classifier.')) and not n.endswith('.bias') else 0.) for n in state['means']}
        if method=='ordinary':
            updates,opt=optimizer.update(gm,state['optimizer'],state['means'])
            means={n:m-lr*(updates[n]+decay[n]*m) for n,m in state['means'].items()};rho=state['rho'];K=0.;pi=0.
        else:means,rho,opt,K,pi=blob.split_update(state['means'],state['rho'],gm,gr,state['optimizer'],optimizer,lr,.002*rate,step,cycle,decay)
        return {'means':means,'rho':rho,'optimizer':opt,'key':key,'updates':t},(value,K,pi,lr)
    return jax.jit(step)


def prediction(cfg,base,target_ids,method='sampled',draws=10,pairs=2):
    def one(state,ids,mask,key):
        if method=='ordinary':return jax.nn.softmax(T.point(base,state['means'],ids,mask,cfg,target_ids),-1)
        keys=jax.random.split(key,draws)
        # Sequential scan bounds activation memory at one predictor, not S networks.
        def body(total,k):
            source,aux=jax.random.split(k)
            if method=='global':
                names=tuple(sorted(state['rho']));keys=jax.random.split(source,len(names));w=dict(state['means'])
                for n,draw in zip(names,keys):w[n]=w[n]+state['rho'][n]**2*jax.random.normal(draw,w[n].shape,dtype=w[n].dtype)
                z=T.point(base,w,ids,mask,cfg,target_ids)
            else:z=blob.forward(base,state['means'],state['rho'],ids,mask,cfg,target_ids,source,aux,pairs,reference=method=='sampled',rematerialize=False)
            return total+jax.nn.softmax(z,-1),None
        initial=jnp.zeros((len(ids),cfg.classes),dtype=state['means'][next(iter(state['means']))].dtype)
        total,_=jax.lax.scan(body,initial,keys);return total/draws
    return jax.jit(one)


def evaluate(fn,state,ds,split,batch=32,seed=7119):
    ids=ds[split+'_ids'];mask=ds[split+'_mask'];y=ds[split+'_labels'];out=[];key=jax.random.PRNGKey(seed)
    for i in range(0,len(y),batch):
        key,k=jax.random.split(key);out.append(np.asarray(fn(state,jnp.asarray(ids[i:i+batch]),jnp.asarray(mask[i:i+batch]),k)))
    p=np.concatenate(out);return p,uncertainty_metrics(p,y) if np.all(y>=0) else {}


def train(task,method='hybrid',seed=41,root='data',output='results',limit_train=0):
    ds=data.text_data(task,root);cfg,base=T.load_pretrained('gpt2' if task=='sst2' else 'roberta');cfg=replace(cfg,family='a')
    state,optimizer=make_state(cfg,base,seed);N=len(ds['train_labels'])
    if limit_train:N=min(N,limit_train)
    epochs=3 if task=='sst2' else 6;batch=32;step=make_step(cfg,base,ds['target_ids'],method,N,epochs,optimizer)
    predict=prediction(cfg,base,ds['target_ids'],'ordinary' if method=='ordinary' else 'sampled',1 if method=='ordinary' else 10)
    folder=Path(output)/f'{task}_{method}_seed{seed}';folder.mkdir(parents=True,exist_ok=True)
    (folder/'config.json').write_text(json.dumps({'task':task,'method':method,'seed':seed,'model':asdict(cfg),'epochs':epochs,'batch':batch,'observations':N,'limit_train':limit_train,'prediction_samples':10 if method!='ordinary' else 1,'prediction_mode':'point' if method=='ordinary' else 'sampled probability average','precision':'float32 highest matmul','selection':'fixed epoch3' if task=='sst2' else 'training-only stopping set'},indent=2))
    history=[];start_epoch=0;seconds=0.;best=float('inf');checkpoint=folder/'recovery.msgpack'
    if checkpoint.exists():
        saved=serialization.from_bytes({'state':state,'epoch':0},checkpoint.read_bytes());state=saved['state'];start_epoch=saved['epoch'];history=json.loads((folder/'curves.json').read_text());seconds=history[-1]['training_seconds'];best=min(row['selection_nll'] for row in history)
    def save(name,epoch):
        p=folder/name;tmp=p.with_suffix('.partial');tmp.write_bytes(serialization.to_bytes({'state':jax.device_get(state),'epoch':epoch}));tmp.replace(p)
    begin=time.perf_counter();jax.block_until_ready(step(state,jnp.asarray(ds['train_ids'][:batch]),jnp.asarray(ds['train_mask'][:batch]),jnp.asarray(ds['train_labels'][:batch])));compile_seconds=time.perf_counter()-begin
    # Embedded numerical evidence has receipts but no checkpoints. A fresh
    # reproduction must not inherit the earlier run's compilation time.
    if checkpoint.exists() and (folder/'receipt.json').exists():compile_seconds+=json.loads((folder/'receipt.json').read_text()).get('compile_seconds',0.)
    for epoch in range(start_epoch,epochs):
        order=np.random.default_rng(90000+seed*101+epoch).permutation(N);begin=time.perf_counter();loss=[]
        for i in range(0,N,batch):
            take=order[i:i+batch];state,stats=step(state,jnp.asarray(ds['train_ids'][take]),jnp.asarray(ds['train_mask'][take]),jnp.asarray(ds['train_labels'][take]));loss.append((stats[0],len(take)))
            if (i//batch)%25==0:
                jax.block_until_ready(state);print(task,method,'epoch',epoch+1,'examples',i+len(take),'loss',float(stats[0]),flush=True)
        jax.block_until_ready(state);seconds+=time.perf_counter()-begin
        if not np.isfinite(float(stats[0])):raise FloatingPointError('Nonfinite text objective')
        begin=time.perf_counter();split='report' if task=='sst2' else 'stop';p,scores=evaluate(predict,state,ds,split,seed=7119+seed*97+epoch);ev=time.perf_counter()-begin
        row={'epoch':epoch+1,'training_seconds':seconds,'evaluation_seconds':ev,'training_loss':sum(float(a)*n for a,n in loss)/N,**{'selection_'+k:v for k,v in scores.items()}}
        history.append(row);(folder/'curves.json').write_text(json.dumps(history,indent=2));save('recovery.msgpack',epoch+1)
        if task=='sst2' or scores['nll']<best:best=scores['nll'];save('selected.msgpack',epoch+1)
        np.savez_compressed(folder/f'epoch{epoch+1}_selection_predictions.npz',probabilities=p,labels=ds[split+'_labels'])
        print(row,flush=True)
    (folder/'receipt.json').write_text(json.dumps({'framework':'jax','devices':[str(d) for d in jax.devices()],'training_seconds':seconds,'compile_seconds':compile_seconds,'completed_epochs':epochs,'status':'subset_execution' if limit_train else 'full_fixed_budget'},indent=2))
    # Test labels never influence the CLINC checkpoint. SST-2 is fixed epoch 3.
    selected=serialization.from_bytes({'state':state,'epoch':0},(folder/'selected.msgpack').read_bytes())
    state=selected['state'];split='report'
    for size in sorted({min(batch,len(ds[split+'_labels'])),len(ds[split+'_labels'])%batch}-{0}):
        predict(state,jnp.asarray(ds[split+'_ids'][:size]),jnp.asarray(ds[split+'_mask'][:size]),jax.random.PRNGKey(7119)).block_until_ready()
    start=time.perf_counter();p,scores=evaluate(predict,state,ds,split);inference_seconds=time.perf_counter()-start
    np.savez_compressed(folder/'test_predictions.npz',probabilities=p,labels=ds[split+'_labels'])
    scores.update(inference_seconds=inference_seconds,training_seconds=seconds,selected_epoch=int(selected['epoch']))
    if 'ood_ids' in ds:
        po,_=evaluate(predict,state,ds,'ood');scores.update(ood_metrics(p,po))
        np.savez_compressed(folder/'ood_predictions.npz',probabilities=po)
    (folder/'metrics.json').write_text(json.dumps(scores,indent=2))
    return history
