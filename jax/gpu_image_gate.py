"""Full-size CIFAR models: real updates before the full training schedule."""
import json,time,gc
from pathlib import Path
import jax
import jax.numpy as j
from bayesify_jax.training import protocol,initialize,make_step,make_predictor
results=json.loads(Path('image_gate.json').read_text()) if Path('image_gate.json').exists() else []
for arch in ['resnet20','densenet40']:
 for method in ['ordinary','sampled_mfvi','moment_mfvi','sampled_ep','moment_ep']:
  if any(r['architecture']==arch and r['method']==method for r in results):continue
  cfg=protocol(arch,method);model,prior,state,probes=initialize(cfg,45000)
  x=jax.random.normal(jax.random.PRNGKey(3),(4,32,32,3),dtype=j.float32);y=j.array([0,1,2,3]);step=make_step(model,prior,45000,cfg,probes)
  start=time.perf_counter();state,loss=step(state,x,y,j.float32(1.));jax.block_until_ready(state)
  assert j.isfinite(loss), (arch,method,loss)
  p=make_predictor(model,cfg)(state,x,jax.random.PRNGKey(4));p.block_until_ready();assert j.isfinite(p).all(), (arch,method)
  results.append({'architecture':arch,'method':method,'loss':float(loss),'elapsed':time.perf_counter()-start});print(results[-1],flush=True)
  with open('image_gate.json','w') as f:json.dump(results,f,indent=2)
  del state,model,step;gc.collect();jax.clear_caches()
print('PASS all image update paths',flush=True)
