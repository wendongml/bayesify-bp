"""Same learned adapter beliefs, sampled versus conditional-moment prediction.

A predictor depends on architecture and integration rule, not learned values.
Compile it once and reuse it for both training methods. Warm-up uses exactly
TT.evaluate's array and PRNG-key types; compilation is outside inference time.
"""
import os
os.environ.setdefault('XLA_FLAGS','--xla_disable_hlo_passes=constant_folding')
from pathlib import Path
from dataclasses import replace
import json,time,gc
import numpy as np
import jax
import jax.numpy as jnp
from flax import serialization
from bayesify_jax import data,transformers as T,text_training as TT
from bayesify_jax.uncertainty import ood_metrics

rows=[]
for task in ['clinc','sst2']:
 ds=data.text_data(task)
 cfg,base=T.load_pretrained('roberta' if task=='clinc' else 'gpt2')
 cfg=replace(cfg,family='a');states={};saved={}
 for training in ['sampled','hybrid']:
  folder=Path('results')/f'{task}_{training}_seed41'
  if not (folder/'selected.msgpack').exists():continue
  state,_=TT.make_state(cfg,base)
  states[training]=serialization.from_bytes({'state':state,'epoch':0},(folder/'selected.msgpack').read_bytes())['state']
  saved[training]=[]
 for mode,draws in [('sampled',1),('sampled',4),('sampled',10),('hybrid',1),('hybrid',4),('global',10)]:
  fn=TT.prediction(cfg,base,ds['target_ids'],mode,draws,pairs=2)
  for index,(training,state) in enumerate(states.items()):
   folder=Path('results')/f'{task}_{training}_seed41'
   start=time.perf_counter()
   # Match evaluate() exactly: NumPy/device arrays and typed/legacy random
   # keys have different JIT cache entries even when numerical values agree.
   for size in sorted({32,len(ds['report_labels'])%32}-{0}):
    fn(state,jnp.asarray(ds['report_ids'][:size]),jnp.asarray(ds['report_mask'][:size]),jax.random.PRNGKey(7119)).block_until_ready()
   cs=time.perf_counter()-start;compiled=fn._cache_size()
   start=time.perf_counter();p,m=TT.evaluate(fn,state,ds,'report');seconds=time.perf_counter()-start
   assert fn._cache_size()==compiled, 'Inference timing included a new JIT specialization.'
   arrays={'probabilities':p,'labels':ds['report_labels']}
   if 'ood_ids' in ds:
    po,_=TT.evaluate(fn,state,ds,'ood');m.update(ood_metrics(p,po));arrays['ood_probabilities']=po
   row={'training':training,'task':task,'prediction':mode,'draws':draws,
        'pairs_per_block':2 if mode=='hybrid' else 0,'inference_seconds':seconds,
        'compile_seconds':cs,'compilation_reused':index>0,**m}
   rows.append(row);saved[training].append(row)
   np.savez_compressed(folder/f'serving_{mode}_{draws}_predictions.npz',**arrays)
   (folder/'serving.json').write_text(json.dumps(saved[training],indent=2));print(row,flush=True)
  del fn;jax.clear_caches();gc.collect()
 Path('results/text_serving.json').write_text(json.dumps(rows,indent=2))
 del base,states;gc.collect()
