"""Train the original Gaussian A+B or full-weight EP/MFVI family.

This entry point deliberately does not call the conditional BLoB optimizer.
All updates are computed by gaussian_text.make_step. The CLINC stopping split
is separate from its reporting split. SST-2 reports the predeclared last epoch.
"""
import os
os.environ.setdefault('XLA_FLAGS','--xla_disable_hlo_passes=constant_folding')
import argparse,json,time
from pathlib import Path
from dataclasses import asdict
import numpy as np
import jax
jax.config.update('jax_default_matmul_precision','highest')
import jax.numpy as j
from flax import serialization
from bayesify_jax import data,transformers as T,gaussian_text as G
from bayesify_jax.text_training import evaluate
from bayesify_jax.uncertainty import ood_metrics


def train(task='sst2',method='mfvi',mode='moment',full_weight=False,epochs=1,
          rank=4,samples=4,learning_rate=1e-4,temperature=.01,seed=41,limit_train=0,optimizer='adam',likelihood_normalization=False):
    ds=data.text_data(task);cfg,base=T.load_pretrained('gpt2' if task=='sst2' else 'roberta')
    N=len(ds['train_labels']);batch=8 if full_weight else 16
    if limit_train:N=min(N,limit_train)
    cfg,prior,state=G.initialize(cfg,base,seed,full_weight,optimizer,N,temperature,likelihood_normalization)
    step=G.make_step(cfg,base,ds['target_ids'],prior,N,method,mode,rank,samples,learning_rate,temperature,optimizer)
    predict0=G.predictor(cfg,base,ds['target_ids'],mode,rank,samples=32)
    predict=lambda state,ids,mask,key:predict0(state['belief'],ids,mask,key)
    folder=Path('results')/f'{task}_{mode}_{method}_{cfg.family}_{optimizer}_seed{seed}';folder.mkdir(parents=True,exist_ok=True)
    (folder/'config.json').write_text(json.dumps({'task':task,'method':mode+'_'+method,'family':cfg.family,'seed':seed,'model':asdict(cfg),'epochs':epochs,'batch':batch,'rank':rank,'samples':samples,'learning_rate':learning_rate,'temperature':temperature,'optimizer':optimizer,'likelihood_normalization':likelihood_normalization,'prediction_samples':32 if mode=='sampled' else 0,'selection':'fixed last epoch' if task=='sst2' else 'training-only stopping NLL','limit_train':limit_train},indent=2))
    history=[];start_epoch=0;seconds=0.;best=float('inf')
    recovery=folder/'recovery.msgpack'
    if recovery.exists():
        saved=serialization.from_bytes({'state':state,'epoch':0},recovery.read_bytes());state=saved['state'];start_epoch=saved['epoch'];history=json.loads((folder/'curves.json').read_text());seconds=history[-1]['training_seconds'];best=min(r['selection_nll'] for r in history)
    def save(name,epoch):
        path=folder/name;part=path.with_suffix('.partial');part.write_bytes(serialization.to_bytes({'state':jax.device_get(state),'epoch':epoch}));part.replace(path)
    start=time.perf_counter()
    for size in sorted({batch,N%batch}-{0}):jax.block_until_ready(step(state,j.asarray(ds['train_ids'][:size]),j.asarray(ds['train_mask'][:size]),j.asarray(ds['train_labels'][:size])))
    compile_seconds=time.perf_counter()-start
    for epoch in range(start_epoch,epochs):
        order=np.random.default_rng(seed*1000+epoch).permutation(N);start=time.perf_counter();losses=[]
        for i in range(0,N,batch):
            take=order[i:i+batch];state,loss=step(state,j.asarray(ds['train_ids'][take]),j.asarray(ds['train_mask'][take]),j.asarray(ds['train_labels'][take]));losses.append((loss,len(take)))
        jax.block_until_ready(state);seconds+=time.perf_counter()-start
        loss=sum(float(a)*n for a,n in losses)/N
        if not np.isfinite(loss):save('failed.msgpack',epoch+1);raise FloatingPointError('Nonfinite Gaussian transformer objective')
        split='report' if task=='sst2' else 'stop';start=time.perf_counter();p,scores=evaluate(predict,state,ds,split,batch=batch)
        row={'epoch':epoch+1,'training_seconds':seconds,'evaluation_seconds':time.perf_counter()-start,'training_loss':loss,**{'selection_'+k:v for k,v in scores.items()}}
        history.append(row);(folder/'curves.json').write_text(json.dumps(history,indent=2));save('recovery.msgpack',epoch+1)
        if task=='sst2' or scores['nll']<best:best=scores['nll'];save('selected.msgpack',epoch+1)
        print(row,flush=True)
    selected=serialization.from_bytes({'state':state,'epoch':0},(folder/'selected.msgpack').read_bytes());state=selected['state']
    for size in sorted({batch,len(ds['report_labels'])%batch}-{0}):predict(state,j.asarray(ds['report_ids'][:size]),j.asarray(ds['report_mask'][:size]),jax.random.PRNGKey(1)).block_until_ready()
    start=time.perf_counter();p,scores=evaluate(predict,state,ds,'report',batch=batch);scores.update(inference_seconds=time.perf_counter()-start,training_seconds=seconds,selected_epoch=selected['epoch'])
    np.savez_compressed(folder/'test_predictions.npz',probabilities=p,labels=ds['report_labels'])
    if 'ood_ids' in ds:
        po,_=evaluate(predict,state,ds,'ood',batch=batch);scores.update(ood_metrics(p,po));np.savez_compressed(folder/'ood_predictions.npz',probabilities=po)
    (folder/'metrics.json').write_text(json.dumps(scores,indent=2));(folder/'receipt.json').write_text(json.dumps({'framework':'jax','status':'subset_execution' if limit_train else 'full_fixed_budget','completed_epochs':epochs,'training_seconds':seconds,'compile_seconds':compile_seconds,'devices':[str(x) for x in jax.devices()]},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['sst2','clinc'],default='sst2');p.add_argument('--method',choices=['mfvi','ep'],default='mfvi');p.add_argument('--mode',choices=['moment','sampled'],default='moment');p.add_argument('--full-weight',action='store_true');p.add_argument('--epochs',type=int,default=1);p.add_argument('--rank',type=int,default=4);p.add_argument('--samples',type=int,default=4);p.add_argument('--learning-rate',type=float,default=1e-4);p.add_argument('--temperature',type=float,default=.01);p.add_argument('--seed',type=int,default=41);p.add_argument('--limit-train',type=int,default=0)
    p.add_argument('--optimizer',choices=['adam','canonical','vadam'],default='adam');p.add_argument('--likelihood-normalization',action='store_true')
    train(**vars(p.parse_args()))
