"""Native pretrained-backbone parity plus real Gaussian adapter updates.

PyTorch/Transformers is used ONLY as an independent public-model reference in
this gate; Bayesify's reader, training, prediction and figures use JAX.
"""
import os,time,json,gc
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
import numpy as np,jax,jax.numpy as j
jax.config.update('jax_default_matmul_precision','highest')
from dataclasses import replace
from pathlib import Path
from bayesify_jax import transformers as T,text_training as TT
from transformers import AutoModel,AutoModelForCausalLM,AutoTokenizer
import torch
rows=[]
for name in ['gpt2','roberta']:
 cfg,base=T.load_pretrained(name);cfg=replace(cfg,family='a')
 repo='openai-community/gpt2' if name=='gpt2' else 'FacebookAI/roberta-base'
 revision='607a30d783dfa663caf39e06633721c8d4cfcd7e' if name=='gpt2' else 'e2da8e2f811d1448a5b465c236feacd80ffbac7b'
 tok=AutoTokenizer.from_pretrained(repo,revision=revision);tok.pad_token=tok.eos_token if name=='gpt2' else tok.pad_token
 encoded=tok(['A useful example.','Another sentence.'],padding=True,return_tensors='np');ids=encoded['input_ids'];mask=encoded['attention_mask']
 targets=j.array([tok.encode(s,add_special_tokens=False)[0] for s in [' negative',' positive']])
 means={n:q.mean for n,q in T.initial_beliefs(cfg).items()}
 if name=='gpt2':
  ref=AutoModelForCausalLM.from_pretrained(repo,revision=revision,attn_implementation='eager').eval()
  with torch.no_grad():z=ref(input_ids=torch.from_numpy(ids),attention_mask=torch.from_numpy(mask)).logits.numpy()
  expected=z[np.arange(len(ids)),mask.sum(-1)-1][:,np.asarray(targets)]
  actual=np.asarray(jax.jit(lambda x,m:T.point(base,means,x,m,cfg,targets))(j.asarray(ids),j.asarray(mask)))
 else:
  ref=AutoModel.from_pretrained(repo,revision=revision,attn_implementation='eager').eval()
  with torch.no_grad():expected=ref(input_ids=torch.from_numpy(ids),attention_mask=torch.from_numpy(mask)).last_hidden_state.numpy()
  def encode(ids,mask):
   h=T.embeddings(base,ids,mask,cfg)
   for i in range(cfg.layers):h=T.block(h,mask,base,means,cfg,i)
   return h
  actual=np.asarray(jax.jit(encode)(j.asarray(ids),j.asarray(mask)))
 error=float(np.max(abs(actual-expected)));np.testing.assert_allclose(actual,expected,atol=3e-4,rtol=2e-4)
 rows.append({'architecture':name,'check':'pretrained backbone','maximum_absolute_error':error,'status':'PASS'});print(rows[-1],flush=True)
 del ref;gc.collect();jax.clear_caches()
 for method in ['ordinary','sampled','hybrid']:
  state,opt=TT.make_state(cfg,base);step=TT.make_step(cfg,base,targets,method,100,3,opt,pairs=2)
  start=time.perf_counter();updated,stats=step(state,j.asarray(ids[:1]),j.asarray(mask[:1]),j.array([0]));jax.block_until_ready(updated)
  assert np.isfinite(float(stats[0]));change=sum(float(j.linalg.norm(updated['means'][n]-v)) for n,v in state['means'].items());assert change>0
  rows.append({'architecture':name,'check':method+' full pretrained update','loss':float(stats[0]),'mean_change_l2_sum':change,'seconds_with_compile':time.perf_counter()-start,'status':'PASS'});print(rows[-1],flush=True)
  Path('results/text_gate.json').write_text(json.dumps(rows,indent=2));del state,updated,step;gc.collect();jax.clear_caches()
 del base;gc.collect();jax.clear_caches()
print('PASS text gates',flush=True)
