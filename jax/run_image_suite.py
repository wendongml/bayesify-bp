"""Full paired protocol, no tuning or cached foreign predictions."""
import os
os.environ.setdefault('XLA_FLAGS','--xla_disable_hlo_passes=constant_folding')

import argparse,json,time
from dataclasses import replace
from pathlib import Path
from bayesify_jax.training import protocol,train
p=argparse.ArgumentParser();p.add_argument('--architectures',nargs='+',default=['mlp','lenet']);p.add_argument('--methods',nargs='+',default=['ordinary','sampled_mfvi','moment_mfvi','sampled_ep','moment_ep']);p.add_argument('--seed',type=int,default=41)
p.add_argument('--dataset',choices=['clean','dirty'],default='clean')
p.add_argument('--epochs',type=int)
a=p.parse_args()
for arch in a.architectures:
 for method in a.methods:
  cfg=protocol(arch,method,a.seed)
  cfg=replace(cfg,dataset=a.dataset,epochs=a.epochs or cfg.epochs)
  Path('progress.json').write_text(json.dumps({'architecture':arch,'method':method,'seed':a.seed,'status':'running'}))
  train(cfg)
  import jax,gc
  jax.clear_caches();gc.collect()
Path('progress.json').write_text(json.dumps({'status':'complete','architectures':a.architectures,'methods':a.methods}))
